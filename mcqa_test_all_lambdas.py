#!/usr/bin/env python3
#SBATCH --job-name=das-test-grid
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=das-test-grid-%j.out
#SBATCH --error=das-test-grid-%j.err

"""
Evaluate Gradual-DAS configs saved as folders.

Expected layout
---------------
RESULTS_DIR/
  gradual_das_ap_lambda_med_0_lambda_readout_0_seed_1/
    meta.csv
    best.pt
    final.pt                  # optional
    epoch_001.pt
    epoch_002.pt
    ...

Checkpoint selection is controlled by --selection:
  max_readout   -> meta.csv row criterion=max_readout_cal
  max_direct    -> meta.csv row criterion=max_direct_cal
  max_mediator  -> meta.csv row criterion=max_mediator_cal
  max_min_3     -> meta.csv row criterion=max_min_3
  best          -> best.pt
  final         -> final.pt if present, otherwise meta.csv final_epoch

Example
-------
python3 mcqa_test_all_lambdas.py \
    --results-dir results_save_every_dims_128_seeds_1 \
    --selection max_readout \
    --overwrite
"""

import argparse
import csv
import os
import re
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression

import mcqa_gradual_tests as causal_tests
from mcqa_data_load_all import build_mcqa_banks
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import collect_layer_states


CONFIG_DIR_RE = re.compile(
    r"^gradual_das_ap_lambda_med_(?P<lambda_med>.+?)_"
    r"lambda_readout_(?P<lambda_readout>.+?)_"
    r"seed_(?P<seed>\d+)$"
)

SELECTION_TO_CRITERION = {
    "max_readout": "max_readout_cal",
    "max_direct": "max_direct_cal",
    "max_mediator": "max_mediator_cal",
    "max_min_3": "max_min_3",
}


def decode_float_tag(text):
    return float(text.replace("m", "-").replace("p", "."))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp, path)


def read_meta_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_meta_row(rows, criterion):
    for row in rows:
        if row.get("criterion") == criterion:
            return row
    raise KeyError(f"criterion={criterion!r} not found in meta.csv")


def epoch_checkpoint_path(config_dir, epoch):
    path = config_dir / f"epoch_{int(epoch):03d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def choose_checkpoint(config_dir, selection):
    meta_path = config_dir / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.csv: {meta_path}")

    rows = read_meta_csv(meta_path)

    if selection == "best":
        path = config_dir / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        return path, None

    if selection == "final":
        final_path = config_dir / "final.pt"
        if final_path.exists():
            return final_path, get_meta_row(rows, "final_epoch")

        row = get_meta_row(rows, "final_epoch")
        epoch = int(float(row["epoch"]))
        return epoch_checkpoint_path(config_dir, epoch), row

    criterion = SELECTION_TO_CRITERION[selection]
    row = get_meta_row(rows, criterion)
    epoch = int(float(row["epoch"]))
    return epoch_checkpoint_path(config_dir, epoch), row


def discover_configs(results_dir, selection):
    configs = []

    for config_dir in results_dir.iterdir():
        if not config_dir.is_dir():
            continue

        match = CONFIG_DIR_RE.fullmatch(config_dir.name)
        if match is None:
            continue

        meta_path = config_dir / "meta.csv"
        if not meta_path.exists():
            print(f"[skip] incomplete config, no meta.csv: {config_dir.name}", flush=True)
            continue


        checkpoint_path, selected_meta = choose_checkpoint(config_dir, selection)
        result = torch.load(checkpoint_path, map_location="cpu")
        selected_epoch = int(result.get("epoch", -1))

        if selected_epoch < 0 and selected_meta is not None:
            selected_epoch = int(float(selected_meta["epoch"]))

        configs.append({
            "config_dir": config_dir,
            "checkpoint_path": checkpoint_path,
            "result": result,
            "selected_meta": selected_meta,
            "selected_epoch": selected_epoch,
            "lambda_med": decode_float_tag(match.group("lambda_med")),
            "lambda_readout": decode_float_tag(match.group("lambda_readout")),
            "optimization_seed": int(match.group("seed")),
            "token_position": result.get(
                "token_position",
                result.get("config", {}).get("token_position", "last_token"),
            ),
        })

    configs.sort(
        key=lambda row: (
            row["lambda_med"],
            row["lambda_readout"],
            row["optimization_seed"],
        )
    )
    return configs


def get_ap_basis(result):
    if "ap_basis" in result:
        return result["ap_basis"].float()
    if "basis" in result:
        return result["basis"].float()
    raise KeyError(f"Checkpoint has neither 'ap_basis' nor 'basis'. Keys={list(result.keys())}")


def get_ap_layer(result, fallback=None):
    if "ap_layer" in result:
        return int(result["ap_layer"])
    if "layer" in result:
        return int(result["layer"])

    cfg = result.get("config", {})
    if "ap_layer" in cfg:
        return int(cfg["ap_layer"])
    if "layer" in cfg:
        return int(cfg["layer"])

    if fallback is not None:
        return int(fallback)

    raise KeyError(
        "Could not find AP layer in checkpoint. "
        "Pass --ap-layer explicitly if epoch checkpoints do not store it."
    )


def cache_metadata_matches(cache, ft_size, cal_size, te_size, data_seed, invariance_size):
    meta = cache.get("metadata", {})
    expected = {
        "ft_size": int(ft_size),
        "cal_size": int(cal_size),
        "te_size": int(te_size),
        "data_seed": int(data_seed),
        "invariance_size": int(invariance_size),
    }
    actual = {
        "ft_size": int(meta.get("ft_size", -1)),
        "cal_size": int(meta.get("cal_size", -1)),
        "te_size": int(meta.get("te_size", -1)),
        "data_seed": int(meta.get("data_seed", -1)),
        "invariance_size": int(meta.get("invariance_size", -1)),
    }
    return actual == expected, expected, actual


def build_or_load_shared_data(
    model,
    tokenizer,
    cache_path,
    ft_size,
    cal_size,
    te_size,
    data_seed,
    invariance_size,
    batch_size,
    rebuild_cache=False,
):
    cache_path = Path(cache_path)

    if cache_path.exists() and not rebuild_cache:
        print(f"[data cache] loading {cache_path}", flush=True)
        cache = torch.load(cache_path, map_location="cpu")
        ok, expected, actual = cache_metadata_matches(
            cache, ft_size, cal_size, te_size, data_seed, invariance_size
        )
        if not ok:
            raise ValueError(
                "Existing data cache does not match this experiment.\n"
                f"expected={expected}\nactual={actual}\n"
                "Use --rebuild-data-cache to regenerate it."
            )
        return cache["fit_bank"], cache["direct_banks"], cache["invariance_banks"]

    print("[data cache] generating shared data once", flush=True)
    device = next(model.parameters()).device

    fit_bank, _, _ = build_mcqa_banks(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=ft_size,
        cal_size=cal_size,
        te_size=te_size,
        dataset_size=None,
        split="train",
        device=device,
        batch_size=batch_size,
        seed=data_seed,
    )

    direct_banks, invariance_banks = causal_tests.build_eval_banks(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=ft_size,
        cal_size=cal_size,
        te_size=te_size,
        invariance_size=invariance_size,
        dataset_size=None,
        split="train",
        batch_size=batch_size,
        seed=data_seed,
    )

    payload = {
        "metadata": {
            "ft_size": int(ft_size),
            "cal_size": int(cal_size),
            "te_size": int(te_size),
            "data_seed": int(data_seed),
            "invariance_size": int(invariance_size),
        },
        "fit_bank": fit_bank,
        "direct_banks": direct_banks,
        "invariance_banks": invariance_banks,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"[data cache] saved to {cache_path}", flush=True)
    return fit_bank, direct_banks, invariance_banks


def normalize_checkpoint(
    result,
    at_result,
    ap_layer,
    at_checkpoint,
    at_readout_checkpoint,
    ap_readout_checkpoint,
    data_seed,
    optimization_seed,
    ft_size,
    cal_size,
    te_size,
):
    normalized = dict(result)
    Q_ap = get_ap_basis(result)
    at_layer = int(at_result["layer"])

    normalized["basis"] = Q_ap
    normalized["ap_basis"] = Q_ap
    normalized["layer"] = int(ap_layer)
    normalized["ap_layer"] = int(ap_layer)
    normalized["at_layer"] = at_layer
    normalized["subspace_dim"] = int(Q_ap.shape[1])
    normalized["ap_subspace_dim"] = int(Q_ap.shape[1])
    normalized["at_subspace_dim"] = int(at_result["basis"].shape[1])
    normalized["at_checkpoint"] = str(at_checkpoint)
    normalized["at_readout_checkpoint"] = str(at_readout_checkpoint)
    normalized["ap_readout_checkpoint"] = str(ap_readout_checkpoint)

    cfg = dict(normalized.get("config", {}))
    cfg["seed"] = int(data_seed)
    cfg["data_seed"] = int(data_seed)
    cfg["optimization_seed"] = int(optimization_seed)
    cfg["ft_size"] = int(ft_size)
    cfg["cal_size"] = int(cal_size)
    cfg["te_size"] = int(te_size)
    normalized["config"] = cfg
    return normalized


def fit_ap_readout(Q_ap, base_states, source_states, fit_bank, seed):
    Q_ap = Q_ap.detach().cpu().float()
    X_base = base_states.float() @ Q_ap
    X_source = source_states.float() @ Q_ap
    X = torch.cat([X_base, X_source], dim=0).numpy()

    y_base = fit_bank["base_answer_pointer_ids"].long()
    y_source = fit_bank["source_answer_pointer_ids"].long()
    y = torch.cat([y_base, y_source], dim=0).numpy()

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=int(seed),
    )
    clf.fit(X, y)

    classes = [int(x) for x in clf.classes_.tolist()]
    if classes != [0, 1, 2, 3]:
        raise ValueError(f"Expected AP classes [0,1,2,3], got {classes}")

    return {
        "W_AP": torch.tensor(clf.coef_, dtype=torch.float32),
        "b_AP": torch.tensor(clf.intercept_, dtype=torch.float32),
        "classes": torch.tensor(classes, dtype=torch.long),
        "train_accuracy": float(clf.score(X, y)),
    }


def selected_training_metrics(meta, result):
    def pick(key):
        if meta is not None and meta.get(key, "") != "":
            return float(meta[key])
        value = result.get(key)
        return float(value) if value is not None else float("nan")

    return {
        "selected_direct_cal": pick("direct_cal"),
        "selected_mediator_cal": pick("mediator_cal"),
        "selected_readout_cal": pick("readout_cal"),
        "selected_min_3_score": pick("min_3_score"),
    }


def flatten_result(meta, test_result, all_tests_path, selection):
    recovery = test_result["ap_to_at_recovery"]
    restoration = test_result["ap_to_at_restoration"]
    ap_inv = test_result["ap_invariance"]
    at_inv = test_result["at_invariance"]
    conflict = test_result["conflict"]
    random_ap = test_result["random_controls"]["ap"]
    random_at = test_result["random_controls"]["at"]

    train_metrics = selected_training_metrics(meta["selected_meta"], meta["result"])

    return {
        "lambda_med": meta["lambda_med"],
        "lambda_readout": meta["lambda_readout"],
        "seed": meta["optimization_seed"],
        "selection": selection,
        "selected_epoch": meta["selected_epoch"],
        **train_metrics,
        "ap_direct_iia": float(test_result["ap_direct"]["counterfactual_iia"]),
        "ap_readout_clean": float(test_result["ap_readout"]["clean_base_accuracy"]),
        "ap_readout_source": float(test_result["ap_readout"]["source_accuracy"]),
        "at_direct_iia": float(test_result["at_direct"]["counterfactual_iia"]),
        "at_readout_after_ap": float(
            test_result["ap_to_at_readout"]["after_ap_intervention_readout_accuracy"]
        ),
        "recovery_iia": float(recovery["counterfactual_iia"]),
        "recovered_fraction": float(recovery["mean_output_effect_recovered_fraction"]),
        "recovery_cosine": float(recovery["shift_cosine"]),
        "restoration_base_acc": float(restoration["restored_base_accuracy"]),
        "restoration_removed_fraction": float(restoration["mean_output_effect_removed_fraction"]),
        "ap_invariance_acc": float(ap_inv["same_ap_different_at_accuracy"]),
        "ap_invariance_l1_drift": float(ap_inv["mean_l1_probability_drift"]),
        "at_invariance_acc": float(at_inv["same_at_different_ap_accuracy"]),
        "at_invariance_l1_drift": float(at_inv["mean_l1_probability_drift"]),
        "conflict_follows_at": float(conflict["follows_conflicting_at"]),
        "random_ap_direct": float(random_ap["mean_direct_iia"]),
        "random_ap_recovery": float(random_ap["mean_recovery_iia"]),
        "random_ap_readout": float(random_ap["mean_readout_accuracy"]),
        "random_at_direct": float(random_at["mean_direct_iia"]),
        "source_checkpoint": str(meta["checkpoint_path"]),
        "all_tests_checkpoint": str(all_tests_path),
    }


SUMMARY_FIELDS = [
    "lambda_med",
    "lambda_readout",
    "seed",
    "selection",
    "selected_epoch",
    "selected_direct_cal",
    "selected_mediator_cal",
    "selected_readout_cal",
    "selected_min_3_score",
    "ap_direct_iia",
    "ap_readout_clean",
    "ap_readout_source",
    "at_direct_iia",
    "at_readout_after_ap",
    "recovery_iia",
    "recovered_fraction",
    "recovery_cosine",
    "restoration_base_acc",
    "restoration_removed_fraction",
    "ap_invariance_acc",
    "ap_invariance_l1_drift",
    "at_invariance_acc",
    "at_invariance_l1_drift",
    "conflict_follows_at",
    "random_ap_direct",
    "random_ap_recovery",
    "random_ap_readout",
    "random_at_direct",
    "source_checkpoint",
    "all_tests_checkpoint",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--selection",
        choices=["max_readout", "max_direct", "max_mediator", "max_min_3", "best", "final"],
        default="max_readout",
    )
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--at-checkpoint", default="results/das_at.pt")
    parser.add_argument("--at-readout-checkpoint", default="results/at_readout_LogisticRegression.pt")
    parser.add_argument("--ap-layer", type=int, default=None)

    parser.add_argument("--ft-size", type=int, default=400)
    parser.add_argument("--cal-size", type=int, default=200)
    parser.add_argument("--te-size", type=int, default=200)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--invariance-size", type=int, default=200)
    parser.add_argument("--num-random-controls", type=int, default=5)
    parser.add_argument("--data-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=32)

    parser.add_argument("--data-cache", default=None)
    parser.add_argument("--rebuild-data-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / f"eval_{args.selection}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cache = Path(args.data_cache) if args.data_cache else results_dir / "mcqa_lambda_eval_data_cache.pt"
    summary_csv = output_dir / "summary.csv"

    configs = discover_configs(results_dir, args.selection)
    if not configs:
        raise RuntimeError(f"No config folders found under {results_dir}")

    print(f"[discover] found {len(configs)} configs", flush=True)
    print(f"[selection] {args.selection}", flush=True)
    for row in configs:
        print(
            f"  lambda_med={row['lambda_med']:g} "
            f"lambda_readout={row['lambda_readout']:g} "
            f"seed={row['optimization_seed']} "
            f"epoch={row['selected_epoch']} -> {row['checkpoint_path'].name}",
            flush=True,
        )

    at_result = torch.load(args.at_checkpoint, map_location="cpu")

    model, tokenizer = load_gemma_model()
    model.eval()
    tokenizer.padding_side = "left"

    fit_bank, shared_direct_banks, shared_invariance_banks = build_or_load_shared_data(
        model=model,
        tokenizer=tokenizer,
        cache_path=data_cache,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        data_seed=args.data_seed,
        invariance_size=args.invariance_size,
        batch_size=args.data_batch_size,
        rebuild_cache=args.rebuild_data_cache,
    )

    def cached_build_eval_banks(**_kwargs):
        return shared_direct_banks, shared_invariance_banks

    causal_tests.build_eval_banks = cached_build_eval_banks
    print("[shared test banks] every config uses identical tensors", flush=True)

    readout_state_cache = {}
    summary_rows = []

    for index, meta in enumerate(configs, start=1):
        result = meta["result"]
        Q_ap = get_ap_basis(result)
        ap_layer = get_ap_layer(result, args.ap_layer)
        token_position = meta["token_position"]

        tag = meta["config_dir"].name.replace("gradual_das_ap_", "", 1)

        readout_path = output_dir / f"ap_readout_{tag}.pt"
        ready_path = output_dir / f"ap_ready_{tag}.pt"
        all_tests_path = output_dir / f"all_tests_{tag}.pt"

        print("\n" + "=" * 78, flush=True)
        print(
            f"[CONFIG {index}/{len(configs)}] "
            f"lambda_med={meta['lambda_med']:g} "
            f"lambda_readout={meta['lambda_readout']:g} "
            f"seed={meta['optimization_seed']} "
            f"epoch={meta['selected_epoch']}",
            flush=True,
        )
        print("=" * 78, flush=True)

        if all_tests_path.exists() and not args.overwrite:
            print("[resume] all-tests already complete -> skip", flush=True)
        else:
            state_key = (ap_layer, token_position)
            if state_key not in readout_state_cache:
                print(f"[shared readout states] collect L{ap_layer} token={token_position}", flush=True)
                base_states = collect_layer_states(
                    model,
                    fit_bank["base_input_ids"],
                    fit_bank["base_attention_mask"],
                    fit_bank["base_position_by_id"],
                    ap_layer,
                    token_position,
                    args.data_batch_size,
                )
                source_states = collect_layer_states(
                    model,
                    fit_bank["source_input_ids"],
                    fit_bank["source_attention_mask"],
                    fit_bank["source_position_by_id"],
                    ap_layer,
                    token_position,
                    args.data_batch_size,
                )
                readout_state_cache[state_key] = (base_states.cpu(), source_states.cpu())

            base_states, source_states = readout_state_cache[state_key]
            readout = fit_ap_readout(
                Q_ap=Q_ap,
                base_states=base_states,
                source_states=source_states,
                fit_bank=fit_bank,
                seed=args.data_seed,
            )

            torch.save({
                "variable": "answer_pointer",
                "method": "LogisticRegression on tuned DAS AP coordinates",
                **readout,
                "ap_checkpoint": str(meta["checkpoint_path"]),
                "ap_layer": ap_layer,
                "subspace_dim": int(Q_ap.shape[1]),
                "token_position": token_position,
                "data_seed": args.data_seed,
            }, readout_path)
            print(f"[1/2 readout] fitted train_acc={readout['train_accuracy']:.4f}", flush=True)

            normalized = normalize_checkpoint(
                result=result,
                at_result=at_result,
                ap_layer=ap_layer,
                at_checkpoint=args.at_checkpoint,
                at_readout_checkpoint=args.at_readout_checkpoint,
                ap_readout_checkpoint=readout_path,
                data_seed=args.data_seed,
                optimization_seed=meta["optimization_seed"],
                ft_size=args.ft_size,
                cal_size=args.cal_size,
                te_size=args.te_size,
            )
            torch.save(normalized, ready_path)

            print("[2/2 all-tests] start", flush=True)
            causal_tests.evaluate_all_tests(
                model=model,
                tokenizer=tokenizer,
                at_checkpoint=args.at_checkpoint,
                ap_checkpoint=str(ready_path),
                at_readout_checkpoint=args.at_readout_checkpoint,
                ap_readout_checkpoint=str(readout_path),
                batch_size=args.test_batch_size,
                invariance_size=args.invariance_size,
                num_random_controls=args.num_random_controls,
                save_path=str(all_tests_path),
            )
            print("[2/2 all-tests] complete", flush=True)

        if all_tests_path.exists():
            test_result = torch.load(all_tests_path, map_location="cpu")
            summary_rows.append(flatten_result(meta, test_result, all_tests_path, args.selection))
            write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)

    print("\n===== DONE =====", flush=True)
    print(f"completed={len(summary_rows)}/{len(configs)}", flush=True)
    print(f"selection={args.selection}", flush=True)
    print(f"data_cache={data_cache}", flush=True)
    print(f"summary={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
