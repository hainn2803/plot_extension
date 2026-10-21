import argparse
import csv
from pathlib import Path

import torch

from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import set_seed, answer_label_ids, output_label_ids, output_iia, collect_all_layer_states, run_full_layer_intervention
from mcqa_load_randomized_ap_at_c import build_pair_bank


PAIR_TARGETS = {
    "AP_only": "Y_after_AP_intervene",
    "AT_only": "Y_after_AT_intervene",
    "C_only": "Y_after_C_intervene",
    "C_AP_AT": "Y_after_C_intervene",
}


def parse_layers(text, num_layers):
    if text == "all":
        return list(range(num_layers))

    layers = []
    for value in text.split(","):
        layers.append(int(value))

    return layers


def add_token_position(bank, token_index, mode):
    key = f"token_{token_index}"

    base_lengths = bank["base_attention_mask"].sum(dim=1).long()
    source_lengths = bank["source_attention_mask"].sum(dim=1).long()

    if mode == "start":
        base_positions = torch.full_like(base_lengths, token_index)
        source_positions = torch.full_like(source_lengths, token_index)
    else:
        base_positions = base_lengths - 1 - token_index
        source_positions = source_lengths - 1 - token_index

    if (base_positions < 0).any() or (source_positions < 0).any():
        raise ValueError(f"Token position {token_index} does not exist for every example")

    bank["base_position_by_id"][key] = base_positions
    bank["source_position_by_id"][key] = source_positions

    return key


def get_num_token_positions(bank):
    base_min_length = int(bank["base_attention_mask"].sum(dim=1).min())
    source_min_length = int(bank["source_attention_mask"].sum(dim=1).min())
    return min(base_min_length, source_min_length)


def get_sample_token(bank, tokenizer, token_index, mode, side):
    input_ids = bank[f"{side}_input_ids"][0]
    attention_mask = bank[f"{side}_attention_mask"][0]

    tokens = input_ids[attention_mask.bool()]

    if mode == "start":
        token_id = tokens[token_index]
    else:
        token_id = tokens[len(tokens) - 1 - token_index]

    return tokenizer.decode([int(token_id)])


@torch.no_grad()
def sweep_token(model, tokenizer, bank, layers, token_index, mode, target, output_space, batch_size):
    token_position = add_token_position(bank, token_index, mode)

    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    source_states = collect_all_layer_states(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        layers,
        token_position,
        batch_size,
    )

    scores = {}

    for layer in layers:
        logits = run_full_layer_intervention(
            model,
            bank,
            source_states[layer],
            layer,
            output_ids,
            token_position,
            batch_size,
        )

        iia = output_iia(logits, bank["intervention_targets"][target], tokenizer, output_space)
        scores[layer] = iia

    return scores


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "token_index", "position_mode", "sample_base_token", "sample_source_token", "iia"])
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors")
    parser.add_argument("--pair-type", choices=PAIR_TARGETS, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--position-mode", choices=["start", "end"], default="end")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-space", choices=["full", "az"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results_token_sweep")
    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_gemma_model()
    model.eval()

    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    target = PAIR_TARGETS[args.pair_type]

    print(f"pair_type = {args.pair_type}")
    print(f"target    = {target}")
    print(f"split     = {args.split}")
    print(f"size      = {args.size}")
    print(f"layers    = {layers}")
    print(f"mode      = {args.position_mode}")

    bank = build_pair_bank(
        model,
        tokenizer,
        args.data_path,
        args.split,
        args.pair_type,
        args.size,
        args.batch_size,
        args.seed,
    )

    num_tokens = get_num_token_positions(bank)

    print(f"\nNumber of pairs: {len(bank['pairs'])}")
    print(f"Token positions shared by all pairs: {num_tokens}")

    rows = []
    matrix = torch.zeros(len(layers), num_tokens)

    for token_index in range(num_tokens):
        base_token = get_sample_token(bank, tokenizer, token_index, args.position_mode, "base")
        source_token = get_sample_token(bank, tokenizer, token_index, args.position_mode, "source")

        print(f"\n=== TOKEN {token_index} | base={base_token!r} source={source_token!r} ===")

        scores = sweep_token(
            model,
            tokenizer,
            bank,
            layers,
            token_index,
            args.position_mode,
            target,
            args.output_space,
            args.batch_size,
        )

        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            iia = scores[layer]
            matrix[layer_idx, token_index] = iia

            print(f"layer={layer:02d} iia={iia:.4f}")

            rows.append({
                "layer": layer,
                "token_index": token_index,
                "position_mode": args.position_mode,
                "sample_base_token": repr(base_token),
                "sample_source_token": repr(source_token),
                "iia": iia,
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = f"{args.pair_type}_{args.split}"
    csv_path = output_dir / f"{name}.csv"
    pt_path = output_dir / f"{name}.pt"

    save_csv(rows, csv_path)

    torch.save({
        "pair_type": args.pair_type,
        "target": target,
        "split": args.split,
        "position_mode": args.position_mode,
        "layers": layers,
        "num_tokens": num_tokens,
        "iia": matrix,
    }, pt_path)

    best_flat = matrix.argmax()
    best_layer_idx = int(best_flat // num_tokens)
    best_token_idx = int(best_flat % num_tokens)

    best_layer = layers[best_layer_idx]
    best_iia = float(matrix[best_layer_idx, best_token_idx])

    print("\n================ BEST ================")
    print(f"layer       = {best_layer}")
    print(f"token_index = {best_token_idx}")
    print(f"iia         = {best_iia:.4f}")
    print(f"saved csv   = {csv_path}")
    print(f"saved pt    = {pt_path}")


if __name__ == "__main__":
    main()