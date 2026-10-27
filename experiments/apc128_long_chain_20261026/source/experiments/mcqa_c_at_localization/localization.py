"""One swap bank shared by full-vector ranking and canonical PLOT."""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch

from . import backend
from .data import from_record, effect
from .lens import load_lens, decode, decoding_metrics
from .matching import signature_cost, one_sided_uot, rank_sites
from .protocol import COLORS, LABELS
from .storage import read, write, file_hash, save_torch, checked_torch, digest


def boundary(root):
    from .runtime import active, guard
    if active(root):
        guard()
        return
    if (Path(root) / "pause.request").exists():
        raise RuntimeError("Paused at a saved artifact boundary")
    root = Path(root)
    thermal_path = root / "engineering/thermal_state.json"
    previous = read(thermal_path) if thermal_path.exists() else None
    if previous is None or time.time() - previous["time"] >= 60:
        from .profiling import thermals
        current = thermals()
        write(thermal_path, current)
        if previous and previous["active"] and current["active"]:
            write(root / "pause.request", {"reason": "thermal throttling at consecutive observations at least 60 seconds apart"})
            raise RuntimeError("Persistent thermal throttling; paused at saved boundary")


def states(model, tokenizer, prompts, layers, batch, path):
    path = Path(path)
    identity = {"prompt_ids": [p["id"] for p in prompts], "layers": list(layers), "batch": batch,
                "backend_sha256": file_hash(Path(backend.__file__))}
    manifest = path.with_suffix(".json")
    if manifest.exists():
        receipt = read(manifest)
        if receipt["identity"] != identity:
            raise ValueError("Activation cache identity differs")
        return checked_torch(path, receipt["sha256"])
    ids = backend.tokenize(tokenizer, [p["text"] for p in prompts])
    collected = {layer: [] for layer in layers}
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            _, hidden = backend.forward(model, ids[start:start+batch].to("cuda"), capture=layers)
            for layer in layers:
                collected[layer].append(hidden[layer].detach().cpu())
    result = {layer: torch.cat(values) for layer, values in collected.items()}
    save_torch(path, result)
    write(manifest, {"identity": identity, "sha256": file_hash(path)}, immutable=True)
    return result


def equal_cell_score(correct, rows):
    cells = defaultdict(list)
    for value, row in zip(correct, rows):
        cells[row["family"], row["condition"]].append(float(value))
    if len(cells) != 16:
        raise ValueError("Missing family/condition cell")
    return float(np.mean([np.mean(v) for v in cells.values()]))


def balanced_decoding(correct, values, conditions):
    cells = defaultdict(list)
    for score, value, condition in zip(correct, values, conditions):
        cells[condition, value].append(float(score))
    if len(cells) != 8:
        raise ValueError("Missing condition/class in lens scoring")
    return float(np.mean([np.mean(v) for v in cells.values()]))


def run(model, tokenizer, root, batch):
    root = Path(root)
    rows = read(root / "banks/localization.json")
    token_map = read(root / "token_map.json")
    sites, label_ids = token_map["sites"], token_map["label_ids"]
    sources = states(model, tokenizer, [r["source"] for r in rows], range(26), batch, root / "cache/localization_sources.pt")
    bases = states(model, tokenizer, [r["base"] for r in rows], range(26), batch, root / "cache/localization_bases.pt")
    ids = backend.tokenize(tokenizer, [r["base"]["text"] for r in rows])
    baselines = []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            logits, _ = backend.forward(model, ids[start:start+batch].to("cuda"))
            baselines.append(backend.summarize_logits(logits, label_ids))
    baseline = {key: sum((r[key] for r in baselines), []) for key in baselines[0]}
    write(root / "localization/clean_baseline.json", baseline, immutable=True)
    neural, scores = [], {"C": [], "AT": []}
    for site in sites:
        layer, pos = site["layer"], site["position"]
        path = root / "localization/swaps" / f"L{layer:02d}_P{pos:02d}.json"
        if path.exists():
            measured = read(path)
            if measured["bank_sha256"] != file_hash(root / "banks/localization.json"):
                raise ValueError("Swap bank differs")
        else:
            chunks = []
            with torch.no_grad():
                for start in range(0, len(ids), batch):
                    patch = backend.Patch(layer, pos, sources[layer][start:start+batch, pos])
                    logits, _ = backend.forward(model, ids[start:start+batch].to("cuda"), [patch])
                    chunks.append(backend.summarize_logits(logits, label_ids))
            measured = {key: sum((r[key] for r in chunks), []) for key in chunks[0]}
            measured["delta_probabilities"] = (np.array(measured["label_probabilities"]) - np.array(baseline["label_probabilities"])).tolist()
            measured.update(site=site, bank_sha256=file_hash(root / "banks/localization.json"), batch=batch)
            write(path, measured, immutable=True)
        neural.append(measured["delta_probabilities"])
        for var in ("C", "AT"):
            targets = [label_ids[LABELS.index(r["targets"][var])] for r in rows]
            scores[var].append(equal_cell_score(np.array(measured["prediction"]) == targets, rows))
        boundary(root)
    abstract = [[effect(from_record(r["base"]), from_record(r["source"]), var) for r in rows] for var in ("C", "AT")]
    cost = signature_cost(abstract, neural)
    coupling, convergence = one_sided_uot(cost)
    write(root / "localization/plot.json", {"cost": cost.tolist(), "coupling": coupling.tolist(), "convergence": convergence}, immutable=True)
    lens, lens_metadata = load_lens()
    lens_scores = {var: [] for var in ("C", "AT")}
    lens_mrr = {var: [] for var in ("C", "AT")}
    lens_results = []
    for layer in range(26):
        path = root / "localization/lens" / f"L{layer:02d}.json"
        if path.exists():
            layer_result = read(path)
        else:
            correct = {var: [] for var in ("C", "AT")}
            mrr = {var: [] for var in ("C", "AT")}
            irrelevant = []
            with torch.no_grad():
                for start in range(0, len(rows), batch):
                    chunk = rows[start:start+batch]
                    hidden = bases[layer][start:start+batch].to("cuda")
                    logits = decode(model, hidden, layer, lens)
                    for var, names, candidates in (("C", COLORS, token_map["color_ids"]), ("AT", LABELS, label_ids)):
                        target = torch.tensor([candidates[names.index(r["base"][var])] for r in chunk], device="cuda")[:, None].expand(logits.shape[:2])
                        acc, rr = decoding_metrics(logits, target, candidates)
                        correct[var].extend(acc.cpu().tolist())
                        mrr[var].extend(rr.cpu().tolist())
                    colors = torch.tensor(token_map["color_ids"], device="cuda")[logits[..., token_map["color_ids"]].argmax(-1)]
                    for r, guesses in zip(chunk, colors.cpu().tolist()):
                        p = from_record(r["base"])
                        other = p.facts[1-p.query]
                        other_id = token_map["color_ids"][COLORS.index(other)] if other in COLORS else None
                        irrelevant.append([int(g == other_id) if other_id is not None else None for g in guesses])
            layer_result = {"correct": correct, "reciprocal_rank": mrr, "irrelevant_color_correct": irrelevant, "metadata": lens_metadata}
            write(path, layer_result, immutable=True)
        for pos in range(1, ids.shape[1]):
            for var in ("C", "AT"):
                lens_scores[var].append(balanced_decoding([r[pos] for r in layer_result["correct"][var]],
                    [r["base"][var] for r in rows], [r["condition"] for r in rows]))
                lens_mrr[var].append(float(np.mean([r[pos] for r in layer_result["reciprocal_rank"][var]])))
        boundary(root)
    rankings, shortlists = {}, {}
    for variable_index, var in enumerate(("C", "AT")):
        rankings[var], shortlists[var] = {}, {}
        for method, values in (("full_vector", scores[var]), ("plot", coupling[variable_index]), ("j_lens", lens_scores[var])):
            order = rank_sites(sites, values, lens_mrr[var] if method == "j_lens" else None)
            rankings[var][method] = [{**sites[i], "score": float(values[i]), "rank": n+1} for n, i in enumerate(order)]
            shortlists[var][method] = [sites[i] for i in order[:2]]
    write(root / "localization/rankings.json", rankings, immutable=True)
    write(root / "selections/shortlists.json", {"shortlists": shortlists, "rankings_sha256": file_hash(root / "localization/rankings.json"),
          "registration": file_hash(root / "registration.json")}, immutable=True)
    return shortlists
