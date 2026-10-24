import argparse, csv, random
from pathlib import Path

import torch

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_c import build_pair_bank, create_bank, symbol_position


def balance_ap_transitions(bank, tokenizer, seed, per_transition=None):
    """Sample the same number of examples from each of the 12 ordered AP transitions."""
    groups = {(base_ap, source_ap): [] for base_ap in range(4) for source_ap in range(4) if base_ap != source_ap}

    for pair in bank["pairs"]:
        groups[(pair["base_AP"], pair["source_AP"])].append(pair)

    min_count = min(len(group) for group in groups.values())
    n = min_count if per_transition is None else per_transition

    for transition, group in groups.items():
        if len(group) < n:
            raise ValueError(f"Transition {transition[0]}->{transition[1]} has only {len(group)} examples, but requested {n}")

    rng = random.Random(seed)
    selected = []

    for transition in groups:
        group = groups[transition]
        rng.shuffle(group)
        selected.extend(group[:n])

    rng.shuffle(selected)
    print(f"Balanced AP transitions: {n} examples per transition, {len(selected)} total")
    return create_bank(selected, tokenizer, size=None, seed=seed), n


def add_ap_positions(bank, tokenizer):
    """Add semantic token positions for AP-only interventions."""
    base_source_slot = []
    source_source_slot = []
    base_golden = []
    source_golden = []

    for pair in bank["pairs"]:
        ap = pair["source_AP"]
        base_letter_at_source_ap = pair["base_choices"]["label"][ap]
        source_letter_at_source_ap = pair["source_choices"]["label"][ap]

        base_source_slot.append(symbol_position(pair["base_prompt"], base_letter_at_source_ap, tokenizer))
        source_source_slot.append(symbol_position(pair["source_prompt"], source_letter_at_source_ap, tokenizer))
        base_golden.append(symbol_position(pair["base_prompt"], pair["base_AT"], tokenizer))
        source_golden.append(symbol_position(pair["source_prompt"], pair["source_AT"], tokenizer))

    bank["base_position_by_id"]["source_ap_slot_label"] = torch.tensor(base_source_slot)
    bank["source_position_by_id"]["source_ap_slot_label"] = torch.tensor(source_source_slot)
    bank["base_position_by_id"]["golden_letter"] = torch.tensor(base_golden)
    bank["source_position_by_id"]["golden_letter"] = torch.tensor(source_golden)


@torch.no_grad()
def collect_source_states(model, bank, layer, token_position, batch_size):
    """Collect source hidden states at one layer and semantic token position."""
    device = next(model.parameters()).device
    states = []

    for start in range(0, len(bank["pairs"]), batch_size):
        end = min(start + batch_size, len(bank["pairs"]))
        input_ids = bank["source_input_ids"][start:end].to(device)
        attention_mask = bank["source_attention_mask"][start:end].to(device)
        positions = bank["source_position_by_id"][token_position][start:end].to(device)

        pad_offset = (attention_mask == 0).sum(dim=1)
        positions = pad_offset + positions
        rows = torch.arange(end - start, device=device)
        captured = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["state"] = hidden[rows, positions].detach().cpu()

        handle = model.model.layers[layer].register_forward_hook(hook)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        handle.remove()
        states.append(captured["state"])

    return torch.cat(states, dim=0)


@torch.no_grad()
def run_full_intervention(model, tokenizer, bank, source_states, layer, token_position, batch_size):
    """Patch source states into base and return the model's predicted answer token."""
    device = next(model.parameters()).device
    predictions = []

    for start in range(0, len(bank["pairs"]), batch_size):
        end = min(start + batch_size, len(bank["pairs"]))
        input_ids = bank["base_input_ids"][start:end].to(device)
        attention_mask = bank["base_attention_mask"][start:end].to(device)
        positions = bank["base_position_by_id"][token_position][start:end].to(device)
        source = source_states[start:end].to(device)

        pad_offset = (attention_mask == 0).sum(dim=1)
        positions = pad_offset + positions
        rows = torch.arange(end - start, device=device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            hidden_new[rows, positions] = source.to(hidden.dtype)
            if isinstance(output, tuple): return (hidden_new,) + output[1:]
            return hidden_new

        handle = model.model.layers[layer].register_forward_hook(hook)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

        try:
            logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, logits_to_keep=1).logits[:, -1]
        except TypeError:
            logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits[:, -1]

        handle.remove()
        token_ids = logits.argmax(dim=-1).tolist()
        predictions.extend(tokenizer.decode([token]).strip() for token in token_ids)

    return predictions


def compute_transition_stats(bank, predictions):
    stats = {(base_ap, source_ap): {"n": 0, "cf_correct": 0, "base_retained": 0} for base_ap in range(4) for source_ap in range(4) if base_ap != source_ap}
    rows = []

    for pair, prediction in zip(bank["pairs"], predictions):
        transition = (pair["base_AP"], pair["source_AP"])
        cf_target = pair["Y_after_AP_intervene"]
        base_answer = pair["base_Y"]
        cf_correct = prediction == cf_target
        base_retained = prediction == base_answer

        stats[transition]["n"] += 1
        stats[transition]["cf_correct"] += int(cf_correct)
        stats[transition]["base_retained"] += int(base_retained)

        rows.append({"base_AP": pair["base_AP"], "source_AP": pair["source_AP"], "transition": f"{pair['base_AP']}->{pair['source_AP']}", "base_answer": base_answer, "counterfactual_target": cf_target, "prediction": prediction, "cf_correct": int(cf_correct), "base_retained": int(base_retained)})

    return stats, rows


def summarize_group(bank, predictions, key):
    result = {}

    for value in range(4):
        indices = [i for i, pair in enumerate(bank["pairs"]) if pair[key] == value]
        n = len(indices)
        cf_correct = 0
        base_retained = 0

        for i in indices:
            pair = bank["pairs"][i]
            prediction = predictions[i]
            cf_correct += int(prediction == pair["Y_after_AP_intervene"])
            base_retained += int(prediction == pair["base_Y"])

        result[value] = {"n": n, "cf_accuracy": cf_correct / n if n else 0.0, "base_retention": base_retained / n if n else 0.0}

    return result


def print_transition_table(stats):
    print("\n================ 12 ORDERED AP TRANSITIONS ================")
    print(f"{'transition':>10} {'N':>6} {'CF acc':>10} {'base retain':>12}")

    for base_ap in range(4):
        for source_ap in range(4):
            if base_ap == source_ap: continue
            s = stats[(base_ap, source_ap)]
            cf_accuracy = s["cf_correct"] / s["n"] if s["n"] else 0.0
            base_retention = s["base_retained"] / s["n"] if s["n"] else 0.0
            print(f"{base_ap}->{source_ap:1d} {s['n']:>10d} {cf_accuracy:>10.4f} {base_retention:>12.4f}")


def print_group_table(title, stats):
    print(f"\n================ {title} ================")
    print(f"{'AP':>4} {'N':>6} {'CF acc':>10} {'base retain':>12}")

    for ap in range(4):
        s = stats[ap]
        print(f"{ap:>4d} {s['n']:>6d} {s['cf_accuracy']:>10.4f} {s['base_retention']:>12.4f}")


def save_results(output_dir, layer, token_position, per_transition, transition_stats, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"ap_transitions_layer{layer}_{token_position}_balanced{per_transition}"
    summary_path = output_dir / f"{stem}_summary.csv"
    rows_path = output_dir / f"{stem}_pairs.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base_AP", "source_AP", "transition", "N", "counterfactual_accuracy", "base_retention"])

        for base_ap in range(4):
            for source_ap in range(4):
                if base_ap == source_ap: continue
                s = transition_stats[(base_ap, source_ap)]
                cf_accuracy = s["cf_correct"] / s["n"] if s["n"] else 0.0
                base_retention = s["base_retained"] / s["n"] if s["n"] else 0.0
                writer.writerow([base_ap, source_ap, f"{base_ap}->{source_ap}", s["n"], cf_accuracy, base_retention])

    with open(rows_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nsaved summary = {summary_path}")
    print(f"saved pairs   = {rows_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pool-size", type=int, default=400)
    parser.add_argument("--per-transition", type=int, default=None)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--token-position", choices=["last_token", "color", "source_ap_slot_label", "golden_letter"], default="last_token")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results_ap_transitions")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    model.eval()

    bank = build_pair_bank(model, tokenizer, args.data_path, args.split, "AP_only", args.pool_size, args.batch_size, args.seed)
    bank, per_transition = balance_ap_transitions(bank, tokenizer, args.seed, args.per_transition)
    add_ap_positions(bank, tokenizer)

    print(f"pair_type      = AP_only")
    print(f"split          = {args.split}")
    print(f"pool_size      = {args.pool_size}")
    print(f"per_transition = {per_transition}")
    print(f"balanced_size  = {len(bank['pairs'])}")
    print(f"layer          = {args.layer}")
    print(f"token_position = {args.token_position}")

    source_states = collect_source_states(model, bank, args.layer, args.token_position, args.batch_size)
    predictions = run_full_intervention(model, tokenizer, bank, source_states, args.layer, args.token_position, args.batch_size)

    transition_stats, rows = compute_transition_stats(bank, predictions)
    from_stats = summarize_group(bank, predictions, "base_AP")
    to_stats = summarize_group(bank, predictions, "source_AP")

    print_transition_table(transition_stats)
    print_group_table("GROUPED BY WHERE AP MOVES FROM", from_stats)
    print_group_table("GROUPED BY WHERE AP MOVES TO", to_stats)
    save_results(args.output_dir, args.layer, args.token_position, per_transition, transition_stats, rows)


if __name__ == "__main__":
    main()
