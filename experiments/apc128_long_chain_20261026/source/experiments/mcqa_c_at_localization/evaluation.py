"""Heldout direct behavior, AT replay controls, and ordered structural interventions."""
from collections import defaultdict
from pathlib import Path
from itertools import product
import numpy as np
import torch

from . import backend, readouts
from .protocol import COLORS, LABELS, order_key, allocate_groups
from .storage import read, write, checked_torch, file_hash, digest
from .localization import states, boundary
from .metrics import breakdown, paired_group_bootstrap, discrete_gate
from .preparation import require_final_lock
from .training import candidate_id


def handle_name(c):
    return candidate_id(c["variable"], c["layer"], c["position"], c["rank"], c["seed"])


def fixed_libraries(rows):
    prompts = {r[side]["id"]: r[side] for r in rows for side in ("base", "source")}
    ordered = sorted(prompts.values(), key=lambda p: order_key(p["id"]))
    assignments = []
    for row in rows:
        target = row["targets"]["C"]
        different = [p for p in ordered if tuple(p["objects"]) != tuple(row["group"])]
        same = next((p for p in different if p["AT"] == target), None)
        changed = next((p for p in different if p["AT"] != target and same is not None
                        and p["objects"] != same["objects"]), None)
        excluded = {tuple(row["group"])}
        if same is not None:
            excluded.add(tuple(same["objects"]))
        if changed is not None:
            excluded.add(tuple(changed["objects"]))
        donor = next((p for p in ordered if p["AT"] == target and tuple(p["objects"]) not in excluded), None)
        if same is None or changed is None or donor is None:
            raise ValueError("Insufficient clean heldout donor/recipient coverage")
        assignments.append({"pair_id": row["id"], "target": target, "recipient_preserve": same["id"],
                            "recipient_change": changed["id"], "donor": donor["id"]})
    structural = []
    for group, condition, query in product(allocate_groups()["test"], ("one_color", "two_colors"), (0, 1)):
        base = next((p for p in ordered if list(p["objects"]) == group and p["condition"] == condition and p["query"] == query), None)
        if base is None:
            raise ValueError(f"Missing structural base coverage: {group}/{condition}/query{query}")
        for color, letter in product(COLORS, LABELS):
            c_donor = next(p for p in ordered if p["C"] == color and p["objects"] != base["objects"])
            at_donor = next(p for p in ordered if p["AT"] == letter and p["objects"] != base["objects"])
            structural.append({"base": base["id"], "color": color, "letter": letter,
                               "C_donor": c_donor["id"], "AT_donor": at_donor["id"]})
    return {"prompts": ordered, "replay": assignments, "structural": structural,
            "selection": "fixed hashes and clean states only; never intervention outcomes"}


def direct(model, tokenizer, root, rows, chosen, batch, readout, namespace="direct"):
    root = Path(root)
    name = handle_name(chosen)
    path = root / "evaluation" / namespace / (name + ".json")
    identity = {"bank": digest(rows), "checkpoint": chosen["sha256"], "physical_batch": batch,
                "readout": file_hash(root / "readouts" / (name + ".json"))}
    if path.exists():
        saved = read(path)
        if saved["identity"] != identity:
            raise ValueError("Direct evaluation resume identity differs")
        return saved
    layer, pos, variable = chosen["layer"], chosen["position"], chosen["variable"]
    basis = checked_torch(root / chosen["checkpoint"], chosen["sha256"])["basis"].to("cuda")
    source = states(model, tokenizer, [r["source"] for r in rows], [layer], batch,
                    root / "cache" / f"{namespace}_test_sources_L{layer:02d}.pt")[layer][:, pos]
    ids = backend.tokenize(tokenizer, [r["base"]["text"] for r in rows])
    label_ids = read(root / "token_map.json")["label_ids"]
    target_ids = [label_ids[LABELS.index(r["targets"][variable])] for r in rows]
    predictions, decoded, label_logits, label_probs, clean_predictions = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            chunk = ids[start:start+batch].to("cuda")
            logits, hidden = backend.forward(model, chunk, [backend.Patch(layer, pos, source[start:start+batch], basis)], capture=[layer])
            measured = backend.summarize_logits(logits, label_ids)
            predictions.extend(measured["prediction"])
            label_logits.extend(measured["label_logits"])
            label_probs.extend(measured["label_probabilities"])
            decoded.extend(readouts.predict(hidden[layer][:, pos], basis, readout).cpu().tolist())
            baseline, _ = backend.forward(model, chunk)
            clean_predictions.extend(baseline.argmax(-1).cpu().tolist())
    correct = (np.array(predictions) == target_ids).tolist()
    summary = breakdown(rows, correct, variable)
    categories = []
    for row, prediction in zip(rows, predictions):
        categories.append([kind for kind, label in row["targets"].items() if label_ids[LABELS.index(label)] == prediction]
                          or ["other_label" if prediction in label_ids else "other_token"])
    names = COLORS if variable == "C" else LABELS
    decoded_targets = [names.index(r["source"][variable]) for r in rows]
    result = {"identity": identity, "handle": chosen, "pair_ids": [r["id"] for r in rows], "predictions": predictions,
              "targets": target_ids, "correct": correct, "decoded_variable": decoded,
              "decoded_targets": decoded_targets, "decoded_classes": list(names),
              "decoded_summary": breakdown(rows, np.array(decoded) == decoded_targets, variable),
              "label_logits": label_logits, "label_probabilities": label_probs,
              "clean_predictions": clean_predictions, "categories": categories, "summary": summary,
              "gate": discrete_gate(summary), "bootstrap": paired_group_bootstrap([r["group"] for r in rows], correct)}
    changed = np.array([r[variable + "_changed"] for r in rows])
    result["uncertainty"] = {}
    for kind, mask in (("changing", changed), ("preservation", ~changed)):
        if mask.any():
            result["uncertainty"][kind] = paired_group_bootstrap([r["group"] for r, yes in zip(rows, mask) if yes], np.array(correct)[mask])
    write(path, result, immutable=True)
    return result


def wording_check(model, tokenizer, root, rows, chosen, heads, batch):
    from dataclasses import replace
    from .data import from_record, pair_record
    root = Path(root)
    cells = defaultdict(dict)
    for row in rows:
        key = tuple(sorted((row["base"]["id"], row["source"]["id"])))
        cells[row["family"], row["condition"]].setdefault(key, []).append(row)
    subset = []
    for cell, pairs in sorted(cells.items()):
        keys = sorted(pairs, key=lambda k: order_key("wording:" + str(k)))[:6]
        for key in keys:
            if len(pairs[key]) != 2:
                raise ValueError("Wording subset requires exact reciprocal pairs")
            subset.extend(pairs[key])
    if len(subset) != 192:
        raise ValueError("Wrong wording subset size")
    changed = [pair_record(replace(from_record(r["base"]), wording="Which"),
                           replace(from_record(r["source"]), wording="Which"), r["family"], r["slot"]) for r in subset]
    write(root / "evaluation/wording_candidates.json", changed, immutable=True)
    prompt_map = {r[s]["id"]: r[s] for r in changed for s in ("base", "source")}
    prompts = sorted(prompt_map.values(), key=lambda p: order_key(p["id"]))
    label_ids = read(root / "token_map.json")["label_ids"]
    clean, predictions = {}, {}
    with torch.no_grad():
        for start in range(0, len(prompts), batch):
            chunk = prompts[start:start+batch]
            ids = backend.tokenize(tokenizer, [p["text"] for p in chunk]).to("cuda")
            if ids.shape[1] != read(root / "token_map.json")["tokens_including_bos"]:
                raise ValueError("Wording alignment differs")
            logits, _ = backend.forward(model, ids)
            for p, prediction in zip(chunk, logits.argmax(-1).cpu().tolist()):
                predictions[p["id"]] = prediction
                clean[p["id"]] = prediction == label_ids[LABELS.index(p["AT"])]
    retained = [r for r in changed if clean[r["base"]["id"]] and clean[r["source"]["id"]]]
    write(root / "evaluation/wording_filter.json", {"predictions": predictions, "retained_ids": [r["id"] for r in retained],
          "raw_pairs": 192, "retained_pairs": len(retained), "replaced": False,
          "balance": "12 directed pairs per family/condition before separate filtering; exact reversals retained"}, immutable=True)
    if retained:
        for name, candidate in chosen.items():
            direct(model, tokenizer, root, retained, candidate, batch, heads[name], namespace="wording")


def chain(model, tokenizer, root, rows, pair, batch, heads, library):
    root = Path(root)
    c, at = pair["C"], pair["AT"]
    path = root / "evaluation" / f"chain_seed_{c['seed']}.json"
    if path.exists():
        return read(path)
    if c["layer"] >= at["layer"] or c["position"] > at["position"]:
        raise ValueError("Incompatible causal order")
    qc = checked_torch(root / c["checkpoint"], c["sha256"])["basis"].to("cuda")
    qa = checked_torch(root / at["checkpoint"], at["sha256"])["basis"].to("cuda")
    label_ids = read(root / "token_map.json")["label_ids"]
    source = states(model, tokenizer, [r["source"] for r in rows], [c["layer"]], batch,
                    root / "cache" / f"test_sources_L{c['layer']:02d}.pt")[c["layer"]][:, c["position"]]
    prompts = library["prompts"]
    index = {p["id"]: i for i, p in enumerate(prompts)}
    lib_ids = backend.tokenize(tokenizer, [p["text"] for p in prompts])
    lib_states = states(model, tokenizer, prompts, sorted({c["layer"], at["layer"]}), batch,
                        root / "cache" / f"library_C{c['layer']}_AT{at['layer']}.pt")
    ids = backend.tokenize(tokenizer, [r["base"]["text"] for r in rows])
    outputs = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start+batch]
            base = ids[start:start+batch].to("cuda")
            logits, induced = backend.forward(model, base, [backend.Patch(c["layer"], c["position"], source[start:start+batch], qc)], capture=[at["layer"]])
            induced_at = induced[at["layer"]][:, at["position"]].detach()
            outputs["direct"].extend(logits.argmax(-1).cpu().tolist())
            outputs["AT_readout"].extend(readouts.predict(induced_at, qa, heads["AT"]).cpu().tolist())
            replay, _ = backend.forward(model, base, [backend.Patch(at["layer"], at["position"], induced_at, qa)])
            outputs["same_base_replay"].extend(replay.argmax(-1).cpu().tolist())
            assignments = library["replay"][start:start+batch]
            donor_index = [index[r["donor"]] for r in assignments]
            donor = lib_states[at["layer"]][donor_index, at["position"]]
            for kind in ("recipient_preserve", "recipient_change"):
                receiver_index = [index[r[kind]] for r in assignments]
                receiver = lib_ids[receiver_index].to("cuda")
                replay, _ = backend.forward(model, receiver, [backend.Patch(at["layer"], at["position"], induced_at, qa)])
                control, _ = backend.forward(model, receiver, [backend.Patch(at["layer"], at["position"], donor, qa)])
                outputs[kind].extend(replay.argmax(-1).cpu().tolist())
                outputs[kind + "_clean_donor"].extend(control.argmax(-1).cpu().tolist())
    targets = np.array([label_ids[LABELS.index(r["targets"]["C"])] for r in rows])
    scores = {k: (np.array(v) == targets).tolist() for k, v in outputs.items() if k != "AT_readout"}
    scores["AT_readout"] = [v == LABELS.index(r["targets"]["C"]) for v, r in zip(outputs["AT_readout"], rows)]
    scores["direct_and_same_base_replay"] = (np.array(scores["direct"]) & scores["same_base_replay"]).tolist()
    structural = []
    cases = library["structural"]
    with torch.no_grad():
        for start in range(0, len(cases), batch):
            chunk = cases[start:start+batch]
            bs = [index[r["base"]] for r in chunk]
            cs = [index[r["C_donor"]] for r in chunk]
            ats = [index[r["AT_donor"]] for r in chunk]
            base = lib_ids[bs].to("cuda")
            _, factual = backend.forward(model, base, capture=[c["layer"], at["layer"]])
            factual_at = factual[at["layer"]][:, at["position"]].detach()
            cp = backend.Patch(c["layer"], c["position"], lib_states[c["layer"]][cs, c["position"]], qc)
            ap = backend.Patch(at["layer"], at["position"], lib_states[at["layer"]][ats, at["position"]], qa)
            joint, _ = backend.forward(model, base, [cp, ap])
            reverse, upstream = backend.forward(model, base, [ap], capture=[c["layer"]])
            restoration, restored = backend.forward(model, base, [cp, backend.Patch(at["layer"], at["position"], factual_at, qa)], capture=[c["layer"], at["layer"]])
            predicted_c = readouts.predict(upstream[c["layer"]][:, c["position"]], qc, heads["C"]).cpu().tolist()
            restored_c = readouts.predict(restored[c["layer"]][:, c["position"]], qc, heads["C"]).cpu().tolist()
            restored_at = readouts.predict(restored[at["layer"]][:, at["position"]], qa, heads["AT"]).cpu().tolist()
            coordinate_residual = ((restored[at["layer"]][:, at["position"]].float() @ qa)
                - (factual_at.float() @ qa)).abs().amax(-1).cpu().tolist()
            upstream_difference = (upstream[c["layer"]][:, c["position"]].float() - factual[c["layer"]][:, c["position"]].float()).abs().amax(-1).cpu().tolist()
            for i, r in enumerate(chunk):
                original = prompts[bs[i]]
                structural.append({**r, "group": original["objects"],
                    "joint_prediction": int(joint[i].argmax()), "joint_correct": int(joint[i].argmax()) == label_ids[LABELS.index(r["letter"])],
                    "AT_only_correct": int(reverse[i].argmax()) == label_ids[LABELS.index(r["letter"])],
                    "upstream_max_abs_change": upstream_difference[i], "upstream_decoded_C": COLORS[predicted_c[i]],
                    "restoration_correct": int(restoration[i].argmax()) == label_ids[LABELS.index(original["AT"])],
                    "base_C": original["C"], "base_AT": original["AT"],
                    "restored_AT_coordinate_max_abs_residual": coordinate_residual[i],
                    "restored_decoded_C": COLORS[restored_c[i]], "restored_decoded_AT": LABELS[restored_at[i]]})
    result = {"pair": pair, "outputs": dict(outputs), "correct": scores,
              "summary": {k: breakdown(rows, v, "C") for k, v in scores.items()}, "structural": structural,
              "library_sha256": file_hash(root / "evaluation/libraries.json"),
              "structural_reference": "fresh clean forward on the identical physical base batch, not library-cache activations",
              "interpretation": "Replay and restoration require reliable AT donor controls; failure does not prove a bypass"}
    changed = np.array([r["C_changed"] for r in rows])
    result["changing_uncertainty"] = {name: paired_group_bootstrap([r["group"] for r, yes in zip(rows, changed) if yes], np.array(values)[changed])
                                      for name, values in scores.items()}
    write(path, result, immutable=True)
    return result


def run(model, tokenizer, root, batch):
    root = Path(root)
    require_final_lock(root)
    rows = read(root / "banks/test.json")
    library = fixed_libraries(rows)
    write(root / "evaluation/libraries.json", library, immutable=True)
    selection = read(root / "selections/seed1.json")
    chains = read(root / "selections/chain_seeds.json")
    chosen = {handle_name(c): c for methods in selection["method_winners"].values() for c in methods.values()}
    chosen.update({handle_name(c): c for pair in chains.values() if pair is not None for c in pair.values()})
    heads, reliability = {}, {}
    for name, candidate in chosen.items():
        if not (root / "readouts" / (name + ".json")).exists():
            raise ValueError("Measurement classifiers must be frozen before opening final data")
        heads[name], reliability[name] = readouts.fit(model, tokenizer, root, candidate, batch)
        direct(model, tokenizer, root, rows, candidate, batch, heads[name])
        boundary(root)
    write(root / "evaluation/readout_reliability.json", reliability, immutable=True)
    for seed, pair in chains.items():
        if pair is not None:
            chain(model, tokenizer, root, rows, pair, batch, {v: heads[handle_name(c)] for v, c in pair.items()}, library)
            boundary(root)
    wording_check(model, tokenizer, root, rows, chosen, heads, batch)
    return {"direct_handles": len(chosen), "chain_seeds": [s for s, p in chains.items() if p is not None]}
