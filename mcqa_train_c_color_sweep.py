"""Train C DAS subspace at every layer L1-L25.

Uses the controlled C_only pairs and the existing Standard DAS training code.
Re-running skips completed layer checkpoints (resume by layer, not mid-epoch).
"""

import argparse
import csv
from pathlib import Path

import torch

from mcqa_gradual_das import set_seed
from mcqa_neural_net import load_gemma_model
from mcqa_train_grid_standard_das import make_variable_banks, train_standard_fixed


LAYERS = range(0, 26)  # inclusive: L1 ... L25
TOKEN_POSITIONS = "color"
TRAIN_PAIR_TYPES = ["C_only"]  # edit here to include other pair types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at_no_distractor")
    parser.add_argument("--result-dir", default="results_c_color_sweep")
    parser.add_argument("--ft-size", type=int, default=800)
    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--te-size", type=int, default=400)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_gemma_model()

    specs = {
        "C": {
            "pair_types": TRAIN_PAIR_TYPES,
            "token_position": TOKEN_POSITIONS,
            "target": "Y_after_C_intervene",
        }
    }
    banks, _ = make_variable_banks(model, tokenizer, args.data_path, args.ft_size, args.cal_size, args.te_size, args.eval_batch_size, args.seed, specs)

    rows = []
    for layer in LAYERS:
        if layer >= len(model.model.layers):
            raise ValueError(f"Layer {layer} is invalid; model has {len(model.model.layers)} layers")

        checkpoint_path = result_dir / f"C_L{layer}.pt"
        train_standard_fixed(
            model=model, tokenizer=tokenizer, banks=banks, variable="C", layer=layer,
            token_position=TOKEN_POSITIONS, target_name="Y_after_C_intervene",
            subspace_dim=args.dim, epochs=args.epochs, lr=args.lr,
            train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
            seed=args.seed, checkpoint_path=checkpoint_path,
            output_space=args.output_space, retrain=args.retrain,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        rows.append({
            "layer": layer,
            "cal_iia": checkpoint["cal_iia"],
            "test_iia": checkpoint["test_iia"],
            "best_epoch": checkpoint["best_epoch"],
        })
        with open(result_dir / "summary.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["layer", "cal_iia", "test_iia", "best_epoch"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. Checkpoints and summary: {result_dir}")


if __name__ == "__main__":
    main()
