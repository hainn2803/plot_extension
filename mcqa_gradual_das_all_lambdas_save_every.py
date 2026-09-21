#!/usr/bin/env python3
#SBATCH --job-name=das-lambda-tune
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=das-lambda-tune-%j.out
#SBATCH --error=das-lambda-tune-%j.err

"""
Tune lambda_med and lambda_readout for Gradual DAS on Delta.

Submit from the repo directory with your environment already activated:

    sbatch -A YOUR_DELTA_PROJECT tune_gradual_das_lambdas.py

Useful examples:

    # Default 6x6 grid, one seed: each lambda in {0, 0.1, 0.3, 1, 3, 5, 8, 10}
    sbatch -A YOUR_DELTA_PROJECT tune_gradual_das_lambdas.py

    # Wider grid
    sbatch -A YOUR_DELTA_PROJECT tune_gradual_das_lambdas.py \
        --lambda-med-grid 0,0.03,0.1,0.3,1,3,5,8,10 \
        --lambda-readout-grid 0,0.03,0.1,0.3,1,3,5,8,10

    # Multiple seeds
    sbatch -A YOUR_DELTA_PROJECT tune_gradual_das_lambdas.py \
        --seeds 0,1,2

The script is resumable at the lambda-combination level. Re-submit the same
command and already-finished checkpoints are skipped unless --overwrite is set.

Important:
- Training/CAL/TEST banks are built once and reused for every lambda setting.
- AP layer selection is done once and reused for every lambda setting.
- Every seed gets its own folder, and every lambda config gets a subfolder inside it.
- For a one-seed invocation (recommended with a Slurm array), summary/aggregate/best files also stay inside that seed folder.
- Lightweight epoch checkpoints are saved according to --save-every (default 1).
- Every config also saves best.pt and final.pt.
- Every config writes meta.csv with:
    * max direct CAL + epoch
    * max mediator CAL + epoch
    * max readout CAL + epoch
    * best max-min-3 score + epoch
    * final epoch metrics
- TEST is evaluated only after the best lambda pair has been selected.
"""

import argparse
import csv
import itertools
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from mcqa_data_load_all import build_mcqa_banks
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import (
    LearnedSubspace,
    answer_label_ids,
    collect_all_layer_states,
    collect_layer_states,
    run_full_layer_intervention,
    set_seed,
)
from mcqa_gradual_das_ap import (
    evaluate_ap,
    run_ap_intervention_and_capture_at,
    run_frozen_at_mediator,
)


def parse_float_grid(text):
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Lambda grid cannot be empty.")
    return values


def parse_int_grid(text):
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Seed list cannot be empty.")
    return values


def float_tag(x):
    text = f"{float(x):g}"
    return text.replace("-", "m").replace(".", "p")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def fixed_cal_score(direct_cal, mediator_cal, readout_cal):
    """
    One fixed selection metric for every lambda pair.

    This deliberately does NOT drop mediator/readout from the score when its
    lambda is zero, because doing that would make scores across lambda settings
    incomparable.
    """
    return min(float(direct_cal), float(mediator_cal), float(readout_cal))


def select_ap_layer(
    model,
    cal_bank,
    at_layer,
    labels,
    token_position,
    eval_batch_size,
    layers=None,
):
    if layers is None:
        layers = list(range(at_layer))
    else:
        layers = [int(layer) for layer in layers if int(layer) < at_layer]

    if not layers:
        raise ValueError(f"No AP candidate layers before AT layer {at_layer}.")

    cal_source_by_layer = collect_all_layer_states(
        model,
        cal_bank["source_input_ids"],
        cal_bank["source_attention_mask"],
        cal_bank["source_position_by_id"],
        layers,
        token_position,
        eval_batch_size,
    )

    layer_results = []
    for layer in layers:
        logits = run_full_layer_intervention(
            model,
            cal_bank,
            cal_source_by_layer[layer],
            layer,
            labels,
            token_position,
            eval_batch_size,
        )
        pred = logits[:, :4].argmax(dim=-1)
        target = cal_bank["counterfactual_label_ids"]["answer_pointer"].cpu()
        iia = float((pred == target).float().mean())

        row = {"layer": int(layer), "cal_iia": iia}
        layer_results.append(row)
        print(f"[AP layer] layer={layer} iia={iia:.4f}", flush=True)

    best_row = max(layer_results, key=lambda row: row["cal_iia"])
    ap_layer = int(best_row["layer"])
    print(
        f"[AP layer selected] layer={ap_layer} "
        f"iia={best_row['cal_iia']:.4f}",
        flush=True,
    )

    return ap_layer, best_row, layer_results, cal_source_by_layer[ap_layer]


def train_one_lambda(
    model,
    fit_bank,
    cal_bank,
    ft_source_ap,
    cal_source_ap,
    ap_layer,
    at_layer,
    Q_at,
    W_at,
    b_at,
    labels,
    token_position,
    subspace_dim,
    epochs,
    lr,
    lambda_med,
    lambda_readout,
    train_batch_size,
    eval_batch_size,
    seed,
    epoch_dir,
    save_every,
    checkpoint_common,
):
    """
    Train one AP alignment for one (lambda_med, lambda_readout, seed).

    Keep TWO snapshots:
      1) BEST CAL epoch according to fixed_cal_score(...)
      2) FINAL training epoch

    Both are returned and saved separately by the caller.

    In addition, save a lightweight checkpoint every `save_every` epochs.
    With --save-every 1, every epoch is saved.
    """
    set_seed(seed)
    device = next(model.parameters()).device

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    target_ap_ft = fit_bank["counterfactual_label_ids"]["answer_pointer"]
    target_at_ft = fit_bank["counterfactual_label_ids"]["answer_token"]

    best_score = -float("inf")
    best_basis = None
    best_epoch = None
    best_direct_cal = None
    best_mediator_cal = None
    best_readout_cal = None

    final_basis = None
    final_epoch = None
    final_direct_cal = None
    final_mediator_cal = None
    final_readout_cal = None
    final_score = None

    epoch_history = []

    n_train = len(fit_bank["base_input_ids"])

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_train)

        total_direct_loss = 0.0
        total_med_loss = 0.0
        total_readout_loss = 0.0
        total_count = 0

        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]

            mini_bank = {
                "base_input_ids": fit_bank["base_input_ids"][idx],
                "base_attention_mask": fit_bank["base_attention_mask"][idx],
                "base_position_by_id": {
                    k: v[idx]
                    for k, v in fit_bank["base_position_by_id"].items()
                },
            }

            optimizer.zero_grad(set_to_none=True)
            Q_ap = alignment.basis()

            direct_logits, generated_at = run_ap_intervention_and_capture_at(
                model,
                mini_bank,
                ft_source_ap[idx],
                Q_ap,
                ap_layer,
                at_layer,
                labels,
                token_position,
                len(idx),
                require_grad=True,
            )

            mediator_logits = run_frozen_at_mediator(
                model,
                mini_bank,
                generated_at,
                Q_at,
                at_layer,
                labels,
                token_position,
                len(idx),
                require_grad=True,
            )

            target_ap = target_ap_ft[idx].to(device)
            target_at = target_at_ft[idx].to(device)

            direct_loss = F.cross_entropy(direct_logits[:, :4], target_ap)
            mediator_loss = F.cross_entropy(mediator_logits[:, :4], target_ap)

            readout_logits = (generated_at @ Q_at) @ W_at.T + b_at
            readout_loss = F.cross_entropy(readout_logits, target_at)

            loss = (
                direct_loss
                + lambda_med * mediator_loss
                + lambda_readout * readout_loss
            )

            loss.backward()
            optimizer.step()

            n = len(idx)
            total_direct_loss += direct_loss.item() * n
            total_med_loss += mediator_loss.item() * n
            total_readout_loss += readout_loss.item() * n
            total_count += n

        with torch.no_grad():
            Q_eval = alignment.basis()
            direct_cal, mediator_cal, readout_cal = evaluate_ap(
                model,
                cal_bank,
                cal_source_ap,
                Q_eval,
                ap_layer,
                at_layer,
                Q_at,
                W_at,
                b_at,
                labels,
                token_position,
                eval_batch_size,
            )

        score = fixed_cal_score(direct_cal, mediator_cal, readout_cal)

        epoch_history.append({
            "epoch": int(epoch),
            "direct_cal": float(direct_cal),
            "mediator_cal": float(mediator_cal),
            "readout_cal": float(readout_cal),
            "min_3_score": float(score),
        })

        print(
            f"[lambda_med={lambda_med:g} lambda_readout={lambda_readout:g} "
            f"seed={seed} epoch={epoch:02d}] "
            f"direct_loss={total_direct_loss / total_count:.4f} "
            f"med_loss={total_med_loss / total_count:.4f} "
            f"readout_loss={total_readout_loss / total_count:.4f} "
            f"direct_cal={direct_cal:.4f} "
            f"mediator_cal={mediator_cal:.4f} "
            f"readout_cal={readout_cal:.4f} "
            f"score={score:.4f}",
            flush=True,
        )

        # --------------------------------------------------------------
        # Lightweight per-epoch snapshot.
        # No optimizer state is stored.
        # Always save the final epoch even when epochs % save_every != 0.
        # --------------------------------------------------------------
        if epoch % save_every == 0 or epoch == epochs:
            epoch_basis = Q_eval.detach().cpu().clone()
            epoch_result = {
                **checkpoint_common,
                "snapshot": f"epoch_{epoch:03d}",
                "basis": epoch_basis,
                "ap_basis": epoch_basis,
                "cal": {
                    "selected_epoch": int(epoch),
                    "best_epoch": int(epoch),
                    "selection_method": "epoch_snapshot",
                    "direct_iia": float(direct_cal),
                    "mediator_iia": float(mediator_cal),
                    "readout_accuracy": float(readout_cal),
                    "score": float(score),
                },
                "test": None,
            }

            epoch_path = Path(epoch_dir) / f"epoch_{epoch:03d}.pt"
            torch.save(epoch_result, epoch_path)
            print(f"[saved epoch] {epoch_path}", flush=True)

        # BEST CAL snapshot.
        if score > best_score:
            best_score = float(score)
            best_basis = Q_eval.detach().cpu().clone()
            best_epoch = int(epoch)
            best_direct_cal = float(direct_cal)
            best_mediator_cal = float(mediator_cal)
            best_readout_cal = float(readout_cal)

        # FINAL snapshot: always overwrite with the current epoch.
        final_basis = Q_eval.detach().cpu().clone()
        final_epoch = int(epoch)
        final_direct_cal = float(direct_cal)
        final_mediator_cal = float(mediator_cal)
        final_readout_cal = float(readout_cal)
        final_score = float(score)

    return {
        "best": {
            "basis": best_basis,
            "epoch": best_epoch,
            "selection_method": "best_cal_score",
            "direct_cal": best_direct_cal,
            "mediator_cal": best_mediator_cal,
            "readout_cal": best_readout_cal,
            "score": float(best_score),
        },
        "final": {
            "basis": final_basis,
            "epoch": final_epoch,
            "selection_method": "final_epoch",
            "direct_cal": final_direct_cal,
            "mediator_cal": final_mediator_cal,
            "readout_cal": final_readout_cal,
            "score": final_score,
        },
        "epoch_history": epoch_history,
    }


def aggregate_lambda_rows(rows):
    groups = {}

    for row in rows:
        key = (float(row["lambda_med"]), float(row["lambda_readout"]))
        groups.setdefault(key, []).append(row)

    aggregated = []
    for (lambda_med, lambda_readout), group in groups.items():
        n = len(group)

        def mean(name):
            return sum(float(row[name]) for row in group) / n

        aggregated.append({
            "lambda_med": lambda_med,
            "lambda_readout": lambda_readout,
            "num_seeds": n,
            # Lambda selection remains based on the BEST-CAL snapshot.
            "mean_score": mean("best_score"),
            "mean_direct_cal": mean("best_direct_cal"),
            "mean_mediator_cal": mean("best_mediator_cal"),
            "mean_readout_cal": mean("best_readout_cal"),
            # Also report the FINAL-epoch averages for comparison.
            "mean_final_score": mean("final_score"),
            "mean_final_direct_cal": mean("final_direct_cal"),
            "mean_final_mediator_cal": mean("final_mediator_cal"),
            "mean_final_readout_cal": mean("final_readout_cal"),
        })

    aggregated.sort(
        key=lambda row: (
            row["mean_score"],
            row["mean_readout_cal"],
            row["mean_mediator_cal"],
            row["mean_direct_cal"],
        ),
        reverse=True,
    )
    return aggregated



def build_config_meta_rows(epoch_history):
    """
    One compact per-config meta table.

    For each individual CAL metric, report the epoch where it is maximal.
    Also report the epoch maximizing min(direct, mediator, readout).
    """
    if not epoch_history:
        raise ValueError("epoch_history is empty.")

    best_direct = max(epoch_history, key=lambda row: row["direct_cal"])
    best_mediator = max(epoch_history, key=lambda row: row["mediator_cal"])
    best_readout = max(epoch_history, key=lambda row: row["readout_cal"])
    best_min3 = max(epoch_history, key=lambda row: row["min_3_score"])
    final = epoch_history[-1]

    def make_row(criterion, selected_row, value_key):
        return {
            "criterion": criterion,
            "epoch": int(selected_row["epoch"]),
            "value": float(selected_row[value_key]),
            "direct_cal": float(selected_row["direct_cal"]),
            "mediator_cal": float(selected_row["mediator_cal"]),
            "readout_cal": float(selected_row["readout_cal"]),
            "min_3_score": float(selected_row["min_3_score"]),
        }

    return [
        make_row("max_direct_cal", best_direct, "direct_cal"),
        make_row("max_mediator_cal", best_mediator, "mediator_cal"),
        make_row("max_readout_cal", best_readout, "readout_cal"),
        make_row("max_min_3", best_min3, "min_3_score"),
        make_row("final_epoch", final, "min_3_score"),
    ]


def load_epoch_history_from_folder(config_dir):
    """
    Reconstruct epoch metrics from saved epoch_XXX.pt snapshots.

    Useful when best.pt/final.pt already exist but meta.csv is missing.
    """
    rows = []

    for path in sorted(Path(config_dir).glob("epoch_*.pt")):
        result = torch.load(path, map_location="cpu")
        cal = result["cal"]

        rows.append({
            "epoch": int(cal["selected_epoch"]),
            "direct_cal": float(cal["direct_iia"]),
            "mediator_cal": float(cal["mediator_iia"]),
            "readout_cal": float(cal["readout_accuracy"]),
            "min_3_score": float(cal["score"]),
        })

    rows.sort(key=lambda row: row["epoch"])
    return rows


def write_config_meta_csv(config_dir, epoch_history):
    meta_path = Path(config_dir) / "meta.csv"
    meta_rows = build_config_meta_rows(epoch_history)

    write_csv(
        meta_path,
        meta_rows,
        fieldnames=[
            "criterion",
            "epoch",
            "value",
            "direct_cal",
            "mediator_cal",
            "readout_cal",
            "min_3_score",
        ],
    )

    return meta_path

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--at-checkpoint",
        default="results/das_at.pt",
    )
    parser.add_argument(
        "--at-readout-checkpoint",
        default="results/at_readout_LogisticRegression.pt",
    )

    parser.add_argument(
        "--lambda-med-grid",
        default="0.0,0.1,0.3,1,3,5,8,10",
        help="Comma-separated lambda_med values.",
    )
    parser.add_argument(
        "--lambda-readout-grid",
        default="0.0,0.1,0.3,1,3,5,8,10",
        help="Comma-separated lambda_readout values.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated training seeds. Example: 0,1,2",
    )

    parser.add_argument(
        "--layers",
        default="all",
        help="AP layer candidates, e.g. 17,18,19. Default: all layers before AT.",
    )
    parser.add_argument("--subspace-dim", type=int, default=128)

    parser.add_argument("--ft-size", type=int, default=None)
    parser.add_argument("--cal-size", type=int, default=None)
    parser.add_argument("--te-size", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a lightweight AP checkpoint every N epochs. Default: 1 (save every epoch).",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--token-position", default=None)

    parser.add_argument(
        "--output-dir",
        default="results_save_every_dims_128_seeds_1",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run lambda combinations even if their checkpoint exists.",
    )

    args = parser.parse_args()

    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1.")

    lambda_med_grid = parse_float_grid(args.lambda_med_grid)
    lambda_readout_grid = parse_float_grid(args.lambda_readout_grid)
    seeds = parse_int_grid(args.seeds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If this invocation runs exactly one seed (recommended for a Slurm array),
    # keep ALL outputs for that job inside its seed folder.  If multiple seeds
    # are run in one invocation, per-config checkpoints still live in seed
    # folders, while the cross-seed summary/aggregate stay at output_dir.
    run_output_dir = (
        output_dir / f"seed_{seeds[0]}"
        if len(seeds) == 1
        else output_dir
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print("===== LAMBDA TUNING CONFIG =====", flush=True)
    print(f"lambda_med_grid={lambda_med_grid}", flush=True)
    print(f"lambda_readout_grid={lambda_readout_grid}", flush=True)
    print(f"seeds={seeds}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"run_output_dir={run_output_dir}", flush=True)
    print(f"save_every={args.save_every}", flush=True)
    print(f"cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    # ------------------------------------------------------------------
    # Load shared model/downstream AT objects ONCE.
    # ------------------------------------------------------------------
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    labels = answer_label_ids(tokenizer)

    at_result = torch.load(args.at_checkpoint, map_location="cpu")
    at_layer = int(at_result["layer"])
    Q_at = at_result["basis"].float().to(device)

    readout_result = torch.load(args.at_readout_checkpoint, map_location="cpu")
    W_at = readout_result["W_AT"].float().to(device)
    b_at = readout_result["b_AT"].float().to(device)

    token_position = (
        at_result["token_position"]
        if args.token_position is None
        else args.token_position
    )

    config = at_result["config"]
    ft_size = config["ft_size"] if args.ft_size is None else args.ft_size
    cal_size = config["cal_size"] if args.cal_size is None else args.cal_size
    te_size = config["te_size"] if args.te_size is None else args.te_size

    # ------------------------------------------------------------------
    # Build the data ONCE. Every lambda pair sees exactly the same banks.
    # ------------------------------------------------------------------
    set_seed(args.data_seed)

    fit_bank, cal_banks, te_banks = build_mcqa_banks(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=ft_size,
        cal_size=cal_size,
        te_size=te_size,
        dataset_size=None,
        split="train",
        device=device,
        batch_size=args.eval_batch_size,
        seed=args.data_seed,
    )

    cal_bank = cal_banks["answer_pointer"]
    te_bank = te_banks["answer_pointer"]

    # ------------------------------------------------------------------
    # Select AP layer ONCE. Lambda does not affect this layer-selection step.
    # ------------------------------------------------------------------
    layers = (
        None
        if args.layers == "all"
        else [int(x) for x in args.layers.split(",") if x.strip()]
    )

    ap_layer, best_layer_row, layer_results, cal_source_ap = select_ap_layer(
        model=model,
        cal_bank=cal_bank,
        at_layer=at_layer,
        labels=labels,
        token_position=token_position,
        eval_batch_size=args.eval_batch_size,
        layers=layers,
    )

    ft_source_ap = collect_layer_states(
        model,
        fit_bank["source_input_ids"],
        fit_bank["source_attention_mask"],
        fit_bank["source_position_by_id"],
        ap_layer,
        token_position,
        args.eval_batch_size,
    )

    # ------------------------------------------------------------------
    # Grid search.
    # ------------------------------------------------------------------
    summary_rows = []
    summary_csv = run_output_dir / "gradual_das_lambda_tuning_summary.csv"

    for lambda_med, lambda_readout, seed in itertools.product(
        lambda_med_grid,
        lambda_readout_grid,
        seeds,
    ):
        # One top-level directory per optimization seed.
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # One subdirectory per lambda configuration inside that seed folder.
        tag = (
            f"gradual_das_ap_lambda_med_{float_tag(lambda_med)}_"
            f"lambda_readout_{float_tag(lambda_readout)}"
        )
        config_dir = seed_dir / tag
        config_dir.mkdir(parents=True, exist_ok=True)

        best_checkpoint_path = config_dir / "best.pt"
        final_checkpoint_path = config_dir / "final.pt"

        if (
            best_checkpoint_path.exists()
            and final_checkpoint_path.exists()
            and not args.overwrite
        ):
            meta_path = config_dir / "meta.csv"

            if not meta_path.exists():
                epoch_history = load_epoch_history_from_folder(config_dir)
                if epoch_history:
                    write_config_meta_csv(config_dir, epoch_history)
                    print(f"[resume] rebuilt missing {meta_path}", flush=True)

            print(
                f"[resume] skip existing best+final for {tag}",
                flush=True,
            )

            best_result = torch.load(best_checkpoint_path, map_location="cpu")
            final_result = torch.load(final_checkpoint_path, map_location="cpu")

            best_cal = best_result["cal"]
            final_cal = final_result["cal"]

            summary_rows.append({
                "lambda_med": float(lambda_med),
                "lambda_readout": float(lambda_readout),
                "seed": int(seed),
                "best_epoch": int(best_cal["selected_epoch"]),
                "best_direct_cal": float(best_cal["direct_iia"]),
                "best_mediator_cal": float(best_cal["mediator_iia"]),
                "best_readout_cal": float(best_cal["readout_accuracy"]),
                "best_score": float(best_cal["score"]),
                "final_epoch": int(final_cal["selected_epoch"]),
                "final_direct_cal": float(final_cal["direct_iia"]),
                "final_mediator_cal": float(final_cal["mediator_iia"]),
                "final_readout_cal": float(final_cal["readout_accuracy"]),
                "final_score": float(final_cal["score"]),
                "best_checkpoint": str(best_checkpoint_path),
                "final_checkpoint": str(final_checkpoint_path),
            })
            continue

        # If this config was interrupted before best.pt/final.pt were written,
        # retrain it from scratch and remove stale epoch snapshots.
        if args.overwrite or not (
            best_checkpoint_path.exists() and final_checkpoint_path.exists()
        ):
            stale_epochs = list(config_dir.glob("epoch_*.pt"))
            if stale_epochs:
                print(
                    f"[restart config] removing {len(stale_epochs)} stale epoch snapshots "
                    f"from {config_dir}",
                    flush=True,
                )
                for stale_path in stale_epochs:
                    stale_path.unlink()

        print(
            "\n"
            "============================================================\n"
            f"TRAIN lambda_med={lambda_med:g} "
            f"lambda_readout={lambda_readout:g} seed={seed}\n"
            "============================================================",
            flush=True,
        )

        checkpoint_common = {
            "variable": "answer_pointer",
            "method": "Gradual DAS lambda tuning",
            "layer": ap_layer,
            "ap_layer": ap_layer,
            "at_layer": at_layer,
            "subspace_dim": args.subspace_dim,
            "ap_subspace_dim": args.subspace_dim,
            "at_subspace_dim": int(Q_at.shape[1]),
            "token_position": token_position,
            "layer_selection": {
                "method": "full_layer_interchange_cal_iia",
                "selected_full_layer_cal_iia": float(best_layer_row["cal_iia"]),
                "all_layers": layer_results,
            },
            "config": {
                "ft_size": ft_size,
                "cal_size": cal_size,
                "te_size": te_size,
                "epochs": args.epochs,
                "save_every": args.save_every,
                "lr": args.lr,
                "lambda_med": float(lambda_med),
                "lambda_readout": float(lambda_readout),
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "seed": int(seed),
                "data_seed": int(args.data_seed),
                "optimization_seed": int(seed),
            },
            "mediator": {
                "variable": "answer_token",
                "layer": at_layer,
                "subspace_dim": int(Q_at.shape[1]),
                "checkpoint": args.at_checkpoint,
                "readout_checkpoint": args.at_readout_checkpoint,
            },
        }

        trained = train_one_lambda(
            model=model,
            fit_bank=fit_bank,
            cal_bank=cal_bank,
            ft_source_ap=ft_source_ap,
            cal_source_ap=cal_source_ap,
            ap_layer=ap_layer,
            at_layer=at_layer,
            Q_at=Q_at,
            W_at=W_at,
            b_at=b_at,
            labels=labels,
            token_position=token_position,
            subspace_dim=args.subspace_dim,
            epochs=args.epochs,
            lr=args.lr,
            lambda_med=lambda_med,
            lambda_readout=lambda_readout,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            seed=seed,
            epoch_dir=config_dir,
            save_every=args.save_every,
            checkpoint_common=checkpoint_common,
        )

        def make_result(snapshot_name, snapshot):
            basis = snapshot["basis"]
            return {
                **checkpoint_common,
                "snapshot": snapshot_name,
                "basis": basis,
                "ap_basis": basis,
                "cal": {
                    "selected_epoch": int(snapshot["epoch"]),
                    # Kept for compatibility with older code.
                    "best_epoch": int(snapshot["epoch"]),
                    "selection_method": snapshot["selection_method"],
                    "direct_iia": float(snapshot["direct_cal"]),
                    "mediator_iia": float(snapshot["mediator_cal"]),
                    "readout_accuracy": float(snapshot["readout_cal"]),
                    "score": float(snapshot["score"]),
                },
                "test": None,
            }

        best_result = make_result("best_cal", trained["best"])
        final_result = make_result("final_epoch", trained["final"])

        torch.save(best_result, best_checkpoint_path)
        torch.save(final_result, final_checkpoint_path)

        meta_path = write_config_meta_csv(
            config_dir,
            trained["epoch_history"],
        )
        print(f"[saved meta]  {meta_path}", flush=True)

        summary_rows.append({
            "lambda_med": float(lambda_med),
            "lambda_readout": float(lambda_readout),
            "seed": int(seed),

            "best_epoch": int(trained["best"]["epoch"]),
            "best_direct_cal": float(trained["best"]["direct_cal"]),
            "best_mediator_cal": float(trained["best"]["mediator_cal"]),
            "best_readout_cal": float(trained["best"]["readout_cal"]),
            "best_score": float(trained["best"]["score"]),

            "final_epoch": int(trained["final"]["epoch"]),
            "final_direct_cal": float(trained["final"]["direct_cal"]),
            "final_mediator_cal": float(trained["final"]["mediator_cal"]),
            "final_readout_cal": float(trained["final"]["readout_cal"]),
            "final_score": float(trained["final"]["score"]),

            "best_checkpoint": str(best_checkpoint_path),
            "final_checkpoint": str(final_checkpoint_path),
        })

        summary_fields = [
            "lambda_med",
            "lambda_readout",
            "seed",
            "best_epoch",
            "best_direct_cal",
            "best_mediator_cal",
            "best_readout_cal",
            "best_score",
            "final_epoch",
            "final_direct_cal",
            "final_mediator_cal",
            "final_readout_cal",
            "final_score",
            "best_checkpoint",
            "final_checkpoint",
        ]

        write_csv(
            summary_csv,
            summary_rows,
            fieldnames=summary_fields,
        )

        print(
            f"[saved best]  {best_checkpoint_path} "
            f"(epoch={trained['best']['epoch']}, score={trained['best']['score']:.4f})",
            flush=True,
        )
        print(
            f"[saved final] {final_checkpoint_path} "
            f"(epoch={trained['final']['epoch']}, score={trained['final']['score']:.4f})",
            flush=True,
        )

        torch.cuda.empty_cache()

    # Ensure summary exists even if every run was resumed.
    summary_fields = [
        "lambda_med",
        "lambda_readout",
        "seed",
        "best_epoch",
        "best_direct_cal",
        "best_mediator_cal",
        "best_readout_cal",
        "best_score",
        "final_epoch",
        "final_direct_cal",
        "final_mediator_cal",
        "final_readout_cal",
        "final_score",
        "best_checkpoint",
        "final_checkpoint",
    ]
    write_csv(summary_csv, summary_rows, fieldnames=summary_fields)

    # ------------------------------------------------------------------
    # Select lambda pair from CAL only, averaging over seeds if >1 seed.
    # ------------------------------------------------------------------
    aggregated = aggregate_lambda_rows(summary_rows)
    aggregate_csv = run_output_dir / "gradual_das_lambda_tuning_aggregate.csv"

    write_csv(
        aggregate_csv,
        aggregated,
        fieldnames=[
            "lambda_med",
            "lambda_readout",
            "num_seeds",
            "mean_score",
            "mean_direct_cal",
            "mean_mediator_cal",
            "mean_readout_cal",
            "mean_final_score",
            "mean_final_direct_cal",
            "mean_final_mediator_cal",
            "mean_final_readout_cal",
        ],
    )

    best = aggregated[0]
    best_lambda_med = float(best["lambda_med"])
    best_lambda_readout = float(best["lambda_readout"])

    print("\n===== BEST LAMBDA ON CAL =====", flush=True)
    print(
        f"lambda_med={best_lambda_med:g} "
        f"lambda_readout={best_lambda_readout:g} "
        f"mean_score={best['mean_score']:.4f} "
        f"mean_direct={best['mean_direct_cal']:.4f} "
        f"mean_mediator={best['mean_mediator_cal']:.4f} "
        f"mean_readout={best['mean_readout_cal']:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # TEST only the selected lambda pair.
    # ------------------------------------------------------------------
    te_source_ap = collect_layer_states(
        model,
        te_bank["source_input_ids"],
        te_bank["source_attention_mask"],
        te_bank["source_position_by_id"],
        ap_layer,
        token_position,
        args.eval_batch_size,
    )

    selected_test_rows = []

    for row in summary_rows:
        if (
            float(row["lambda_med"]) != best_lambda_med
            or float(row["lambda_readout"]) != best_lambda_readout
        ):
            continue

        checkpoint_path = Path(row["best_checkpoint"])
        result = torch.load(checkpoint_path, map_location="cpu")
        Q_ap = result["basis"].float().to(device)

        direct_test, mediator_test, readout_test = evaluate_ap(
            model,
            te_bank,
            te_source_ap,
            Q_ap,
            ap_layer,
            at_layer,
            Q_at,
            W_at,
            b_at,
            labels,
            token_position,
            args.eval_batch_size,
        )

        result["test"] = {
            "direct_iia": direct_test,
            "mediator_iia": mediator_test,
            "readout_accuracy": readout_test,
        }
        torch.save(result, checkpoint_path)

        selected_test_rows.append({
            "seed": int(row["seed"]),
            "direct_test": float(direct_test),
            "mediator_test": float(mediator_test),
            "readout_test": float(readout_test),
            "checkpoint": str(checkpoint_path),
        })

        print(
            f"[BEST TEST seed={row['seed']}] "
            f"direct={direct_test:.4f} "
            f"mediator={mediator_test:.4f} "
            f"readout={readout_test:.4f}",
            flush=True,
        )

    best_info = {
        "lambda_med": best_lambda_med,
        "lambda_readout": best_lambda_readout,
        "cal": best,
        "test_by_seed": selected_test_rows,
    }

    with open(run_output_dir / "gradual_das_best_lambda.json", "w") as f:
        json.dump(best_info, f, indent=2)

    # Convenient canonical checkpoint when tuning with exactly one seed.
    if len(seeds) == 1 and len(selected_test_rows) == 1:
        source = Path(selected_test_rows[0]["checkpoint"])
        destination = run_output_dir / "gradual_das_ap_tuned.pt"
        shutil.copy2(source, destination)
        print(f"[best checkpoint] {destination}", flush=True)

    print(f"[summary] {summary_csv}", flush=True)
    print(f"[aggregate] {aggregate_csv}", flush=True)
    print(f"[best info] {run_output_dir / 'gradual_das_best_lambda.json'}", flush=True)


if __name__ == "__main__":
    main()