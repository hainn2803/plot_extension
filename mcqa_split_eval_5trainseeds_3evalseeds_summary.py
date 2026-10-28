import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path


METHODS = ["gradual", "standard"]
TRAIN_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEEDS = [0, 1, 2, 3, 4]


def run_eval(eval_script, result_dir, eval_seed, test_size, batch_size, output_space, change_ratio, output_path):
    command = [
        sys.executable,
        eval_script,
        "--result-dir", str(result_dir),
        "--seed", str(eval_seed),
        "--test-size", str(test_size),
        "--batch-size", str(batch_size),
        "--output-space", output_space,
        "--change-ratio", str(change_ratio),
    ]

    print("\n" + "=" * 100)
    print(f"EVAL result_dir={result_dir} eval_seed={eval_seed}")
    print("=" * 100)
    print(" ".join(command))

    process = subprocess.run(command)

    if process.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {' '.join(command)}")

    default_eval_path = result_dir / "eval_results.txt"

    if not default_eval_path.exists():
        raise FileNotFoundError(f"Expected evaluation output not found: {default_eval_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_eval_path.replace(output_path)


def clean_key(text):
    text = text.strip()
    text = text.replace(" ", "_")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("->", "to")
    text = text.replace("|", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def parse_eval_results(path):
    text = Path(path).read_text()
    metrics = {}
    section = None
    subgroup = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line == "":
            continue

        if line.startswith("===") and line.endswith("==="):
            section = clean_key(line.strip("=").strip())
            subgroup = None
            continue

        if line.startswith("changed:"):
            subgroup = "changed"
            content = line[len("changed:"):].strip()
            matches = re.findall(r"([A-Za-z0-9_]+)\s*=\s*([0-9]*\.?[0-9]+)", content)

            for name, value in matches:
                key = f"{section}__changed__{clean_key(name)}"
                metrics[key] = float(value)

            subgroup = None
            continue

        if line.startswith("fixed:"):
            subgroup = "fixed"
            content = line[len("fixed:"):].strip()
            matches = re.findall(r"([A-Za-z0-9_]+)\s*=\s*([0-9]*\.?[0-9]+)", content)

            for name, value in matches:
                key = f"{section}__fixed__{clean_key(name)}"
                metrics[key] = float(value)

            subgroup = None
            continue

        if line.startswith("changed replay:"):
            content = line[len("changed replay:"):].strip()
            matches = re.findall(r"([A-Za-z0-9_]+)\s*=\s*([0-9]*\.?[0-9]+)", content)

            for name, value in matches:
                key = f"{section}__changed_replay__{clean_key(name)}"
                metrics[key] = float(value)

            continue

        if line.startswith("fixed replay:"):
            content = line[len("fixed replay:"):].strip()
            matches = re.findall(r"([A-Za-z0-9_]+)\s*=\s*([0-9]*\.?[0-9]+)", content)

            for name, value in matches:
                key = f"{section}__fixed_replay__{clean_key(name)}"
                metrics[key] = float(value)

            continue

        # Parse ordinary "name: value" metrics.
        colon_matches = re.findall(r"([^|:]+):\s*([0-9]*\.?[0-9]+)", line)

        for name, value in colon_matches:
            key = f"{section}__{clean_key(name)}"
            metrics[key] = float(value)

        # Parse clean readout lines such as:
        # AP overall base/source: 0.9000 / 0.9100
        slash_match = re.match(r"(.+?):\s*([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)$", line)

        if slash_match is not None:
            name = clean_key(slash_match.group(1))
            metrics[f"{section}__{name}_base"] = float(slash_match.group(2))
            metrics[f"{section}__{name}_source"] = float(slash_match.group(3))

    return metrics


def save_csv(rows, path):
    if len(rows) == 0:
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def metric_names(rows):
    names = []

    for row in rows:
        for key in row:
            if key in ("method", "train_seed", "eval_seed"):
                continue

            if key not in names:
                names.append(key)

    return names


def aggregate_eval_seeds(rows):
    metrics = metric_names(rows)
    output = []

    for method in METHODS:
        for train_seed in TRAIN_SEEDS:
            config_rows = []

            for row in rows:
                if row["method"] == method and int(row["train_seed"]) == train_seed:
                    config_rows.append(row)

            if len(config_rows) == 0:
                continue

            for metric in metrics:
                values = []

                for row in config_rows:
                    if metric in row and row[metric] != "":
                        values.append(float(row[metric]))

                if len(values) == 0:
                    continue

                output.append({
                    "method": method,
                    "train_seed": train_seed,
                    "metric": metric,
                    "mean_over_eval_seeds": statistics.mean(values),
                    "std_over_eval_seeds": statistics.pstdev(values),
                    "min": min(values),
                    "max": max(values),
                    "n_eval_seeds": len(values),
                })

    return output


def aggregate_training_seeds(config_rows):
    metrics = []

    for row in config_rows:
        if row["metric"] not in metrics:
            metrics.append(row["metric"])

    output = []

    for method in METHODS:
        for metric in metrics:
            values = []

            for row in config_rows:
                if row["method"] == method and row["metric"] == metric:
                    values.append(float(row["mean_over_eval_seeds"]))

            if len(values) == 0:
                continue

            output.append({
                "method": method,
                "metric": metric,
                "mean_over_train_seeds": statistics.mean(values),
                "std_over_train_seeds": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
                "n_train_seeds": len(values),
            })

    return output


def save_fixed_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def print_summary(rows):
    print("\n" + "=" * 100)
    print("FINAL SPLIT-EVAL SUMMARY: 5 TRAIN SEEDS x 3 EVAL SEEDS")
    print("=" * 100)

    for method in METHODS:
        print(f"\n=== {method.upper()} ===")

        for row in rows:
            if row["method"] == method:
                print(f'{row["metric"]}: {row["mean_over_train_seeds"]:.4f} ± {row["std_over_train_seeds"]:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results_standard_vs_gradual_5seeds_no_dims")
    parser.add_argument("--eval-script", default="mcqa_eval_readout_mediator_split.py")
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--change-ratio", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    all_rows = []

    for method in METHODS:
        for train_seed in TRAIN_SEEDS:
            result_dir = root / method / f"seed_{train_seed}"
            best_layers_path = result_dir / "best_layers.txt"

            if not best_layers_path.exists():
                print(f"[SKIP] Missing completed training run: {result_dir}")
                continue

            eval_dir = result_dir / "split_eval_seeds"
            eval_dir.mkdir(parents=True, exist_ok=True)

            for eval_seed in EVAL_SEEDS:
                eval_path = eval_dir / f"eval_seed_{eval_seed}.txt"

                if args.resume and eval_path.exists():
                    print(f"[RESUME] Using existing split evaluation: {eval_path}")
                else:
                    run_eval(args.eval_script, result_dir, eval_seed, args.test_size, args.batch_size, args.output_space, args.change_ratio, eval_path)

                metrics = parse_eval_results(eval_path)

                row = {
                    "method": method,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                }

                for key, value in metrics.items():
                    row[key] = value

                all_rows.append(row)
                save_csv(all_rows, root / "split_eval_all_runs.csv")

    per_config = aggregate_eval_seeds(all_rows)

    save_fixed_csv(
        per_config,
        root / "split_eval_per_config_summary.csv",
        ["method", "train_seed", "metric", "mean_over_eval_seeds", "std_over_eval_seeds", "min", "max", "n_eval_seeds"],
    )

    final_summary = aggregate_training_seeds(per_config)

    save_fixed_csv(
        final_summary,
        root / "split_eval_final_summary.csv",
        ["method", "metric", "mean_over_train_seeds", "std_over_train_seeds", "min", "max", "n_train_seeds"],
    )

    print_summary(final_summary)

    print(f"\nall eval runs      = {root / 'split_eval_all_runs.csv'}")
    print(f"per-config summary = {root / 'split_eval_per_config_summary.csv'}")
    print(f"final summary      = {root / 'split_eval_final_summary.csv'}")


if __name__ == "__main__":
    main()
