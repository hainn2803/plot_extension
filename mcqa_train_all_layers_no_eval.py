"""Train independent Standard DAS handles and readouts at every layer; no chain evaluation.

AT: last_token, AT_only
AP: color_0/color_1/color_2/color_3 (joint), AP_only
C:  color, C_only
Rerun to skip completed handles/readouts; summary.csv is rebuilt from saved results.
"""

import argparse
import csv
from pathlib import Path

import torch

from mcqa_gradual_das import set_seed
from mcqa_load_randomized_ap_at_c import build_shared_readout_banks
from mcqa_neural_net import load_gemma_model
from mcqa_train_grid_standard_das import (
    clean_readout_accuracy,
    collect_states,
    color_targets,
    fit_one_readout,
    load_ckpt,
    load_readout,
    make_variable_banks,
    readout_prediction,
    train_standard_fixed,
)


SPECS = {
    "AT": {"pair_types": ["AT_only"], "token_position": "last_token", "target": "Y_after_AT_intervene"},
    "AP": {"pair_types": ["AP_only"], "token_position": ["color_0", "color_1", "color_2", "color_3"], "target": "Y_after_AP_intervene"},
    "C": {"pair_types": ["C_only"], "token_position": "color", "target": "Y_after_C_intervene"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at_no_distractor")
    parser.add_argument("--result-dir", default="results_standard_all_layers")
    parser.add_argument("--variables", default="AT,AP,C", help="AT, AP, C or a subset")
    parser.add_argument("--layers", default="all", help="all or comma-separated layer indices")
    parser.add_argument("--ft-size", type=int, default=800)
    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--te-size", type=int, default=400)
    parser.add_argument("--readout-train-size", type=int, default=1300)
    parser.add_argument("--readout-val-size", type=int, default=520)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--at-epochs", type=int, default=10)
    parser.add_argument("--ap-epochs", type=int, default=10)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    variables = [name.strip().upper() for name in args.variables.split(",") if name.strip()]
    if not variables or len(set(variables)) != len(variables) or any(name not in SPECS for name in variables):
        raise ValueError("Use --variables AT,AP,C or a unique subset")

    model, tokenizer = load_gemma_model()
    num_layers = len(model.model.layers)
    layers = list(range(num_layers)) if args.layers.lower() == "all" else [int(x.strip()) for x in args.layers.split(",")]
    if not layers or any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(f"Valid layers: 0 to {num_layers - 1}")

    specs = {}
    for variable in variables:
        specs[variable] = SPECS[variable]

    banks, _ = make_variable_banks(model, tokenizer, args.data_path, args.ft_size, args.cal_size, args.te_size, args.eval_batch_size, args.seed, specs)
    shared_readout = build_shared_readout_banks(model, tokenizer, args.data_path, args.readout_train_size, args.readout_val_size, args.eval_batch_size, args.seed)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.csv"
    fields = ["variable", "layer", "cal_iia", "test_iia", "best_epoch", "readout_val_acc", "clean_base_acc", "clean_source_acc"]
    rows = []
    epochs = {"AT": args.at_epochs, "AP": args.ap_epochs, "C": args.c_epochs}

    for variable in variables:
        spec = SPECS[variable]
        token_position = spec["token_position"]
        test_bank = banks[variable][2][variable]

        for layer in layers:
            handle_dir = result_dir / "trained_handles" / f"{variable}_L{layer}"
            checkpoint_path = handle_dir / f"das_{variable.lower()}.pt"
            readout_path = handle_dir / f"{variable.lower()}_readout_LogisticRegression.pt"

            train_standard_fixed(
                model=model, tokenizer=tokenizer, banks=banks, variable=variable, layer=layer,
                token_position=token_position, target_name=spec["target"],
                subspace_dim=args.dim, epochs=epochs[variable], lr=args.lr,
                train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
                seed=args.seed, checkpoint_path=checkpoint_path,
                output_space=args.output_space, retrain=args.retrain,
            )

            if args.retrain and readout_path.exists():
                readout_path.unlink()
            fit_one_readout(model, checkpoint_path, readout_path, variable, shared_readout, token_position, args.eval_batch_size, args.seed)

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            device = next(model.parameters()).device
            Q, _, _ = load_ckpt(checkpoint_path, device)
            W, b, readout_info = load_readout(readout_path, variable, device)

            val_bank = shared_readout[1]
            states = collect_states(model, val_bank["input_ids"], val_bank["attention_mask"], val_bank["position_by_id"], layer, token_position, args.eval_batch_size)
            prediction = readout_prediction(states, Q, W, b)
            readout_val_acc = float((prediction == val_bank[f"{variable}_target_ids"].cpu()).float().mean())

            if variable == "AT":
                base_target, source_target = test_bank["base_answer_label_ids"], test_bank["source_answer_label_ids"]
            elif variable == "AP":
                base_target, source_target = test_bank["base_answer_pointer_ids"], test_bank["source_answer_pointer_ids"]
            else:
                base_target = color_targets(test_bank, readout_info["color_to_class"], "base")
                source_target = color_targets(test_bank, readout_info["color_to_class"], "source")

            clean_base = clean_readout_accuracy(model, test_bank, Q, layer, token_position, W, b, base_target, args.eval_batch_size, "base")
            clean_source = clean_readout_accuracy(model, test_bank, Q, layer, token_position, W, b, source_target, args.eval_batch_size, "source")

            row = {
                "variable": variable, "layer": layer,
                "cal_iia": checkpoint["cal_iia"], "test_iia": checkpoint["test_iia"],
                "best_epoch": checkpoint["best_epoch"], "readout_val_acc": readout_val_acc,
                "clean_base_acc": clean_base, "clean_source_acc": clean_source,
            }
            rows.append(row)
            with open(summary_path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            print(f"[{variable} L{layer}] cal_iia={row['cal_iia']:.4f} test_iia={row['test_iia']:.4f} readout_val={readout_val_acc:.4f} clean={clean_base:.4f}/{clean_source:.4f}")

    print(f"Done. Saved Q/readouts under {result_dir / 'trained_handles'} and metrics in {summary_path}")


if __name__ == "__main__":
    main()
