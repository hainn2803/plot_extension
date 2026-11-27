"""Train independent Standard DAS handles across selected layers; NO chain eval.

Training: mixed pairs (all five pair types).
Calibration/test: controlled, output-changing pairs only.

AT: last_token; calibration/test on AT_only.
AP: joint color_0..color_3; calibration/test on AP_only.
C: color; calibration/test on C_only.

Data sampling and minibatch seeds use the GDAS v2 aligned bank builder.
Saves summary.csv using the original Standard all-layer schema.
Select layers by calibration IIA (max-5%) in later chain evaluation.
A restart preserves existing summary rows for other variables and layers.
"""

import argparse
import csv
from pathlib import Path

import torch

import mcqa_train_gradual_efficient_v2 as gdas_config
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
    make_gdas_aligned_banks,
    readout_prediction,
    train_standard_fixed,
)

ALL_PAIRS = ["AT_only", "AP_only", "C_only", "AP_AT", "C_AP_AT"]

SPECS = {
    "AT": {"pair_types": list(ALL_PAIRS), "token_position": "last_token", "target": "Y_after_AT_intervene"},
    "AP": {"pair_types": list(ALL_PAIRS), "token_position": ["color_0", "color_1", "color_2", "color_3"], "target": "Y_after_AP_intervene"},
    "C": {"pair_types": list(ALL_PAIRS), "token_position": "color", "target": "Y_after_C_intervene"},
}

SUMMARY_FIELDS = [
    "variable", "layer", "cal_iia", "test_iia", "best_epoch",
    "readout_val_acc", "clean_base_acc", "clean_source_acc",
]


def configure_pair_types():
    # Override GDAS bank-builder settings only in this running process.
    gdas_config.TRAIN_PAIR_TYPES = {
        "AT": list(ALL_PAIRS),
        "AP": list(ALL_PAIRS),
        "C": list(ALL_PAIRS),
    }
    gdas_config.CAL_TEST_PAIR_TYPES = {
        "AT": ["AT_only"],
        "AP": ["AP_only"],
        "C": ["C_only"],
    }
    gdas_config.CHANGING_ONLY_CAL_TEST = True


def save_summary(summary_path, saved):
    order = {"AT": 0, "AP": 1, "C": 2}
    keys = sorted(saved, key=lambda x: (order.get(x[0], 99), x[1]))
    temp = summary_path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for key in keys:
            writer.writerow({field: saved[key].get(field, "") for field in SUMMARY_FIELDS})
    temp.replace(summary_path)


def load_summary(summary_path):
    saved = {}
    if summary_path.is_file():
        with summary_path.open(newline="") as file:
            for row in csv.DictReader(file):
                if row.get("variable") and row.get("layer"):
                    key = (row["variable"], int(row["layer"]))
                    saved[key] = {field: row.get(field, "") for field in SUMMARY_FIELDS}
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at_no_distractor")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--variables", default="AT,AP,C", help="AT, AP, C or unique subset")
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

    if args.result_dir is None:
        args.result_dir = f"results_standard_all_layers_mixed_train_controlled_cal_test_seed{args.seed}_no_distractor"

    configure_pair_types()
    print(f"Train pair types: {gdas_config.TRAIN_PAIR_TYPES}", flush=True)
    print(f"Cal/test pair types: {gdas_config.CAL_TEST_PAIR_TYPES}", flush=True)
    print(f"Changing-only cal/test: {gdas_config.CHANGING_ONLY_CAL_TEST}", flush=True)

    set_seed(args.seed)
    variables = [name.strip().upper() for name in args.variables.split(",") if name.strip()]
    if not variables or len(set(variables)) != len(variables) or any(name not in SPECS for name in variables):
        raise ValueError("Use --variables AT,AP,C or a unique subset")

    model, tokenizer = load_gemma_model()
    num_layers = len(model.model.layers)
    layers = list(range(num_layers)) if args.layers.lower() == "all" else [int(x.strip()) for x in args.layers.split(",")]
    if not layers or any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(f"Valid layers: 0 to {num_layers - 1}")

    specs = {variable: SPECS[variable] for variable in variables}
    banks, _ = make_gdas_aligned_banks(
        model, tokenizer, args.data_path, args.ft_size, args.cal_size,
        args.te_size, args.eval_batch_size, args.seed, specs,
    )
    shared_readout = build_shared_readout_banks(
        model, tokenizer, args.data_path, args.readout_train_size,
        args.readout_val_size, args.eval_batch_size, args.seed,
    )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.csv"
    saved = load_summary(summary_path)
    epochs = {"AT": args.at_epochs, "AP": args.ap_epochs, "C": args.c_epochs}

    for variable in variables:
        token_position = SPECS[variable]["token_position"]
        test_bank = banks[variable][2][variable]
        for layer in layers:
            handle_dir = result_dir / "trained_handles" / f"{variable}_L{layer}"
            checkpoint_path = handle_dir / f"das_{variable.lower()}.pt"
            readout_path = handle_dir / f"{variable.lower()}_readout_LogisticRegression.pt"

            train_standard_fixed(
                model=model, tokenizer=tokenizer, banks=banks,
                variable=variable, layer=layer, token_position=token_position,
                target_name=SPECS[variable]["target"],
                subspace_dim=args.dim, epochs=epochs[variable], lr=args.lr,
                train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
                seed=args.seed, checkpoint_path=checkpoint_path,
                output_space=args.output_space, retrain=args.retrain,
            )
            if args.retrain and readout_path.exists():
                readout_path.unlink()
            fit_one_readout(
                model, checkpoint_path, readout_path, variable, shared_readout,
                token_position, args.eval_batch_size, args.seed,
            )

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            device = next(model.parameters()).device
            Q, _, _ = load_ckpt(checkpoint_path, device)
            W, b, readout_info = load_readout(readout_path, variable, device)

            val_bank = shared_readout[1]
            states = collect_states(
                model, val_bank["input_ids"], val_bank["attention_mask"],
                val_bank["position_by_id"], layer, token_position, args.eval_batch_size,
            )
            prediction = readout_prediction(states, Q, W, b)
            readout_val_acc = float(
                (prediction == val_bank[f"{variable}_target_ids"].cpu()).float().mean()
            )

            if variable == "AT":
                base_target = test_bank["base_answer_label_ids"]
                source_target = test_bank["source_answer_label_ids"]
            elif variable == "AP":
                base_target = test_bank["base_answer_pointer_ids"]
                source_target = test_bank["source_answer_pointer_ids"]
            else:
                color_map = readout_info["color_to_class"]
                base_target = color_targets(test_bank, color_map, "base")
                source_target = color_targets(test_bank, color_map, "source")

            clean_base = clean_readout_accuracy(
                model, test_bank, Q, layer, token_position, W, b,
                base_target, args.eval_batch_size, "base",
            )
            clean_source = clean_readout_accuracy(
                model, test_bank, Q, layer, token_position, W, b,
                source_target, args.eval_batch_size, "source",
            )
            row = {
                "variable": variable,
                "layer": layer,
                "cal_iia": checkpoint["cal_iia"],
                "test_iia": checkpoint.get("test_iia"),
                "best_epoch": checkpoint["best_epoch"],
                "readout_val_acc": readout_val_acc,
                "clean_base_acc": clean_base,
                "clean_source_acc": clean_source,
            }
            saved[(variable, layer)] = row
            save_summary(summary_path, saved)
            print(
                f"[{variable} L{layer}] cal_iia={row['cal_iia']:.4f} "
                f"test_iia={row['test_iia']} readout_val={readout_val_acc:.4f} "
                f"clean={clean_base:.4f}/{clean_source:.4f}"
            )

    print(f"Training complete (no chain evaluation). Saved {summary_path}")
    print(f"Checkpoints/readouts: {result_dir / 'trained_handles'}")


if __name__ == "__main__":
    main()
