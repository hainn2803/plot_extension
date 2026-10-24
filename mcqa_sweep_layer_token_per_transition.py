import argparse, csv, random
from pathlib import Path

import torch

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_c import create_bank, load_generated_dataset, predict_correct, rows_to_pairs
from mcqa_gradual_das import answer_label_ids, answer_logits, output_iia, output_label_ids, set_seed

TRANSITIONS = [(i, j) for i in range(4) for j in range(4) if i != j]


def build_transition_banks(model, tokenizer, data_path, split, per_transition, batch_size, seed):
    ds = load_generated_dataset(data_path)
    ids = list(range(len(ds[split])))
    random.Random(seed).shuffle(ids)
    groups = {transition: [] for transition in TRANSITIONS}
    cache = {}
    batch_size = 64

    for start in range(0, len(ids), batch_size):
        candidates = rows_to_pairs(ds[split], ids[start:start + batch_size], split)
        candidates = [p for p in candidates if p["pair_type"] == "AP_only"]
        candidates = predict_correct(model, tokenizer, candidates, batch_size, cache)

        for pair in candidates:
            groups[(pair["base_AP"], pair["source_AP"])].append(pair)

        if per_transition is not None and all(len(groups[t]) >= per_transition for t in TRANSITIONS):
            break

    print("\nAvailable correct AP-only pairs:")
    for transition in TRANSITIONS:
        print(f"{transition[0]}->{transition[1]}: {len(groups[transition])}")

    n = min(len(groups[t]) for t in TRANSITIONS) if per_transition is None else per_transition
    if n <= 0:
        raise RuntimeError("At least one AP transition has no examples.")

    banks = {}
    for k, transition in enumerate(TRANSITIONS):
        if len(groups[transition]) < n:
            raise ValueError(f"Transition {transition[0]}->{transition[1]} has only {len(groups[transition])} examples, but requested {n}")
        pairs = list(groups[transition])
        random.Random(seed + 1000 + k).shuffle(pairs)
        banks[transition] = create_bank(pairs[:n], tokenizer, size=None, seed=seed + k)

    print(f"\nBalanced dataset: {n} examples per transition")
    return banks, n


def real_lengths(mask):
    return mask.sum(dim=1).long()


def common_token_count(bank):
    base_min = int(real_lengths(bank["base_attention_mask"]).min())
    source_min = int(real_lengths(bank["source_attention_mask"]).min())
    return min(base_min, source_min)


def token_positions(mask, token_index, mode):
    lengths = real_lengths(mask)
    pad = (mask == 0).sum(dim=1).long()

    if mode == "end":
        raw = lengths - 1 - token_index
    else:
        raw = torch.full_like(lengths, token_index)

    return pad + raw


def example_token(bank, tokenizer, token_index, mode, side):
    ids = bank[f"{side}_input_ids"][0]
    mask = bank[f"{side}_attention_mask"][0]
    real_ids = ids[mask.bool()]

    if mode == "end":
        token_id = int(real_ids[-1 - token_index])
    else:
        token_id = int(real_ids[token_index])

    return tokenizer.decode([token_id]).replace("\n", "\\n")


@torch.no_grad()
def collect_source_all_layers(model, bank, layers, token_index, mode, batch_size):
    device = next(model.parameters()).device
    collected = {layer: [] for layer in layers}

    for start in range(0, len(bank["pairs"]), batch_size):
        end = min(start + batch_size, len(bank["pairs"]))
        ids = bank["source_input_ids"][start:end].to(device)
        mask = bank["source_attention_mask"][start:end].to(device)
        pos = token_positions(mask, token_index, mode).to(device)
        rows = torch.arange(len(ids), device=device)
        handles = []

        def make_hook(layer):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                collected[layer].append(hidden[rows, pos].detach().float().cpu())
            return hook

        for layer in layers:
            handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))

        try:
            model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()

    return {layer: torch.cat(collected[layer], dim=0) for layer in layers}


@torch.no_grad()
def run_full_intervention(model, tokenizer, bank, source_states, layer, token_index, mode, batch_size, output_space):
    device = next(model.parameters()).device
    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)
    all_logits = []

    for start in range(0, len(bank["pairs"]), batch_size):
        end = min(start + batch_size, len(bank["pairs"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_states[start:end].to(device)
        pos = token_positions(mask, token_index, mode).to(device)
        rows = torch.arange(len(ids), device=device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            hidden_new[rows, pos] = source.to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            all_logits.append(answer_logits(model, outputs, mask, output_ids).cpu())
        finally:
            handle.remove()

    logits = torch.cat(all_logits, dim=0)
    target = bank["intervention_targets"]["Y_after_AP_intervene"]
    return output_iia(logits, target, tokenizer, output_space)


def save_transition(rows, matrix, transition, token_labels, output_dir, mode):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{transition[0]}_to_{transition[1]}"
    csv_path = output_dir / f"ap_{name}_full_layer_token_sweep.csv"
    pt_path = output_dir / f"ap_{name}_full_layer_token_sweep.pt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["base_AP", "source_AP", "layer", "token_index", "base_token", "source_token", "iia"])
        writer.writeheader()
        writer.writerows(rows)

    torch.save({"transition": transition, "mode": mode, "iia": matrix, "token_labels": token_labels}, pt_path)
    print(f"saved csv = {csv_path}")
    print(f"saved pt  = {pt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="randomized_copycolors")
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--per-transition", type=int, default=80)
    p.add_argument("--layers", default="all")
    p.add_argument("--transition", default="all", help="all or one transition such as 0->1")
    p.add_argument("--mode", choices=["end", "start"], default="end")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-space", choices=["full", "az"], default="full")
    p.add_argument("--output-dir", default="results_ap_transition_token_sweep")
    a = p.parse_args()

    set_seed(a.seed)
    model, tokenizer = load_gemma_model()
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    layers = list(range(model.config.num_hidden_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",") if x.strip()]
    banks, n = build_transition_banks(model, tokenizer, a.data_path, a.split, a.per_transition, a.batch_size, a.seed)
    transitions = TRANSITIONS if a.transition == "all" else [tuple(int(x) for x in a.transition.replace("->", ",").split(","))]
    if any(t not in TRANSITIONS for t in transitions):
        raise ValueError("--transition must be all or one of 0->1, 0->2, ..., 3->2")

    print(f"\nsplit          = {a.split}")
    print(f"per transition = {n}")
    print(f"layers         = {layers}")
    print(f"mode           = {a.mode}")
    print(f"output space   = {a.output_space}")

    best_rows = []

    for transition in TRANSITIONS:
        bank = banks[transition]
        n_tokens = common_token_count(bank)
        matrix = torch.zeros(len(layers), n_tokens)
        rows = []
        token_labels = []

        print(f"\n================ AP {transition[0]}->{transition[1]} | N={len(bank['pairs'])} | TOKENS={n_tokens} ================")

        for token_index in range(n_tokens):
            base_token = example_token(bank, tokenizer, token_index, a.mode, "base")
            source_token = example_token(bank, tokenizer, token_index, a.mode, "source")
            token_labels.append({"token_index": token_index, "base_token": base_token, "source_token": source_token})
            print(f"\n=== TOKEN {token_index} | base={base_token!r} source={source_token!r} ===")

            source_by_layer = collect_source_all_layers(model, bank, layers, token_index, a.mode, a.batch_size)

            for layer_idx, layer in enumerate(layers):
                iia = run_full_intervention(model, tokenizer, bank, source_by_layer[layer], layer, token_index, a.mode, a.batch_size, a.output_space)
                matrix[layer_idx, token_index] = iia
                rows.append({"base_AP": transition[0], "source_AP": transition[1], "layer": layer, "token_index": token_index, "base_token": base_token, "source_token": source_token, "iia": iia})
                print(f"layer={layer:02d} iia={iia:.4f}")

        flat_idx = int(matrix.argmax())
        best_layer_idx = flat_idx // n_tokens
        best_token_idx = flat_idx % n_tokens
        best = {"transition": f"{transition[0]}->{transition[1]}", "layer": layers[best_layer_idx], "token_index": best_token_idx, "base_token": token_labels[best_token_idx]["base_token"], "source_token": token_labels[best_token_idx]["source_token"], "iia": float(matrix[best_layer_idx, best_token_idx])}
        best_rows.append(best)

        print(f"\nBEST {best['transition']}: layer={best['layer']} token={best['token_index']} base={best['base_token']!r} source={best['source_token']!r} iia={best['iia']:.4f}")
        save_transition(rows, matrix, transition, token_labels, a.output_dir, a.mode)

    summary_path = Path(a.output_dir) / "best_by_transition.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transition", "layer", "token_index", "base_token", "source_token", "iia"])
        writer.writeheader()
        writer.writerows(best_rows)

    print(f"\nsaved summary = {summary_path}")


if __name__ == "__main__":
    main()
