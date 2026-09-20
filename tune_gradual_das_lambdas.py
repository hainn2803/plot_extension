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
- Lambda selection uses CAL only.
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
):
    """
    Train one AP alignment for one (lambda_med, lambda_readout, seed).

    The best epoch is selected only on CAL using a fixed score:
        min(direct_cal, mediator_cal, readout_cal)
    """
    set_seed(seed)
    device = next(model.parameters()).device

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    target_ap_ft = fit_bank["counterfactual_label_ids"]["answer_pointer"]
    target_at_ft = fit_bank["counterfactual_label_ids"]["answer_token"]

    best_score = -1.0
    best_basis = None
    best_epoch = None
    best_direct_cal = None
    best_mediator_cal = None
    best_readout_cal = None

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

        if score > best_score:
            best_score = score
            best_basis = Q_eval.detach().cpu().clone()
            best_epoch = epoch
            best_direct_cal = direct_cal
            best_mediator_cal = mediator_cal
            best_readout_cal = readout_cal

    return {
        "basis": best_basis,
        "best_epoch": int(best_epoch),
        "direct_cal": float(best_direct_cal),
        "mediator_cal": float(best_mediator_cal),
        "readout_cal": float(best_readout_cal),
        "score": float(best_score),
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
            "mean_score": mean("score"),
            "mean_direct_cal": mean("direct_cal"),
            "mean_mediator_cal": mean("mediator_cal"),
            "mean_readout_cal": mean("readout_cal"),
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

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--token-position", default=None)

    parser.add_argument(
        "--output-dir",
        default="results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run lambda combinations even if their checkpoint exists.",
    )

    args = parser.parse_args()

    lambda_med_grid = parse_float_grid(args.lambda_med_grid)
    lambda_readout_grid = parse_float_grid(args.lambda_readout_grid)
    seeds = parse_int_grid(args.seeds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== LAMBDA TUNING CONFIG =====", flush=True)
    print(f"lambda_med_grid={lambda_med_grid}", flush=True)
    print(f"lambda_readout_grid={lambda_readout_grid}", flush=True)
    print(f"seeds={seeds}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
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
    summary_csv = output_dir / "gradual_das_lambda_tuning_summary.csv"

    for lambda_med, lambda_readout, seed in itertools.product(
        lambda_med_grid,
        lambda_readout_grid,
        seeds,
    ):
        tag = (
            f"gradual_das_ap_lambda_med_{float_tag(lambda_med)}_"
            f"lambda_readout_{float_tag(lambda_readout)}_"
            f"seed_{seed}"
        )
        checkpoint_path = output_dir / f"{tag}.pt"

        if checkpoint_path.exists() and not args.overwrite:
            print(f"[resume] skip existing {checkpoint_path}", flush=True)
            result = torch.load(checkpoint_path, map_location="cpu")
            cal = result["cal"]

            summary_rows.append({
                "lambda_med": float(lambda_med),
                "lambda_readout": float(lambda_readout),
                "seed": int(seed),
                "best_epoch": int(cal["best_epoch"]),
                "direct_cal": float(cal["direct_iia"]),
                "mediator_cal": float(cal["mediator_iia"]),
                "readout_cal": float(cal["readout_accuracy"]),
                "score": float(cal["score"]),
                "checkpoint": str(checkpoint_path),
            })
            continue

        print(
            "\n"
            "============================================================\n"
            f"TRAIN lambda_med={lambda_med:g} "
            f"lambda_readout={lambda_readout:g} seed={seed}\n"
            "============================================================",
            flush=True,
        )

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
        )

        result = {
            "variable": "answer_pointer",
            "method": "Gradual DAS lambda tuning",
            "layer": ap_layer,
            "subspace_dim": args.subspace_dim,
            "basis": trained["basis"],
            "token_position": token_position,
            "layer_selection": {
                "method": "full_layer_interchange_cal_iia",
                "selected_full_layer_cal_iia": float(best_layer_row["cal_iia"]),
                "all_layers": layer_results,
            },
            "cal": {
                "best_epoch": trained["best_epoch"],
                "direct_iia": trained["direct_cal"],
                "mediator_iia": trained["mediator_cal"],
                "readout_accuracy": trained["readout_cal"],
                "score": trained["score"],
            },
            "test": None,
            "config": {
                "ft_size": ft_size,
                "cal_size": cal_size,
                "te_size": te_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "lambda_med": float(lambda_med),
                "lambda_readout": float(lambda_readout),
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "seed": int(seed),
                "data_seed": int(args.data_seed),
            },
            "mediator": {
                "variable": "answer_token",
                "layer": at_layer,
                "subspace_dim": int(Q_at.shape[1]),
                "checkpoint": args.at_checkpoint,
                "readout_checkpoint": args.at_readout_checkpoint,
            },
        }

        torch.save(result, checkpoint_path)

        summary_rows.append({
            "lambda_med": float(lambda_med),
            "lambda_readout": float(lambda_readout),
            "seed": int(seed),
            "best_epoch": trained["best_epoch"],
            "direct_cal": trained["direct_cal"],
            "mediator_cal": trained["mediator_cal"],
            "readout_cal": trained["readout_cal"],
            "score": trained["score"],
            "checkpoint": str(checkpoint_path),
        })

        write_csv(
            summary_csv,
            summary_rows,
            fieldnames=[
                "lambda_med",
                "lambda_readout",
                "seed",
                "best_epoch",
                "direct_cal",
                "mediator_cal",
                "readout_cal",
                "score",
                "checkpoint",
            ],
        )

        print(f"[saved] {checkpoint_path}", flush=True)
        torch.cuda.empty_cache()

    # Ensure summary exists even if every run was resumed.
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "lambda_med",
            "lambda_readout",
            "seed",
            "best_epoch",
            "direct_cal",
            "mediator_cal",
            "readout_cal",
            "score",
            "checkpoint",
        ],
    )

    # ------------------------------------------------------------------
    # Select lambda pair from CAL only, averaging over seeds if >1 seed.
    # ------------------------------------------------------------------
    aggregated = aggregate_lambda_rows(summary_rows)
    aggregate_csv = output_dir / "gradual_das_lambda_tuning_aggregate.csv"

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

        checkpoint_path = Path(row["checkpoint"])
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

    with open(output_dir / "gradual_das_best_lambda.json", "w") as f:
        json.dump(best_info, f, indent=2)

    # Convenient canonical checkpoint when tuning with exactly one seed.
    if len(seeds) == 1 and len(selected_test_rows) == 1:
        source = Path(selected_test_rows[0]["checkpoint"])
        destination = output_dir / "gradual_das_ap_tuned.pt"
        shutil.copy2(source, destination)
        print(f"[best checkpoint] {destination}", flush=True)

    print(f"[summary] {summary_csv}", flush=True)
    print(f"[aggregate] {aggregate_csv}", flush=True)
    print(f"[best info] {output_dir / 'gradual_das_best_lambda.json'}", flush=True)


if __name__ == "__main__":
    main()
