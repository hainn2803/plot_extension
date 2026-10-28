import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = [0, 1, 2, 3, 4]

METHODS = {
    "standard": {
        "ap_lambda_med": 0.0,
        "ap_lambda_readout": 0.0,
        "c_lambda_ap_med": 0.0,
        "c_lambda_ap_readout": 0.0,
        "c_lambda_at_med": 0.0,
        "c_lambda_at_readout": 0.0,
    },
    "gradual": {
        "ap_lambda_med": 1.0,
        "ap_lambda_readout": 1.0,
        "c_lambda_ap_med": 0.0,
        "c_lambda_ap_readout": 0.0,
        "c_lambda_at_med": 2.0,
        "c_lambda_at_readout": 2.0,
    },
}


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed with return code {return_code}: {' '.join(command)}")


def read_csv(path):
    rows = []

    with open(path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


def read_best_layers(path):
    best = {}

    for line in Path(path).read_text().splitlines():
        line = line.strip()

        if line == "":
            continue

        left, score_text = line.split("|")
        variable_text, layer_text = left.split(":")
        variable = variable_text.strip().replace(" layer", "")
        layer = int(layer_text.strip())
        score = float(score_text.split(":")[1].strip())

        best[variable] = {
            "layer": layer,
            "score": score,
        }

    return best


def find_layer_row(rows, layer):
    for row in rows:
        if int(row["layer"]) == int(layer):
            return row

    raise ValueError(f"Cannot find layer {layer}")


def collect_run_summary(method, seed, result_dir):
    best = read_best_layers(result_dir / "best_layers.txt")

    at_rows = read_csv(result_dir / "at_layer_summary.csv")
    ap_rows = read_csv(result_dir / "ap_layer_summary.csv")
    c_rows = read_csv(result_dir / "c_layer_summary.csv")

    at_row = find_layer_row(at_rows, best["AT"]["layer"])
    ap_row = find_layer_row(ap_rows, best["AP"]["layer"])
    c_row = find_layer_row(c_rows, best["C"]["layer"])

    row = {
        "method": method,
        "seed": seed,
        "at_layer": best["AT"]["layer"],
        "at_score": best["AT"]["score"],
        "at_test_iia": float(at_row["test_iia"]),
        "ap_layer": best["AP"]["layer"],
        "ap_score": best["AP"]["score"],
        "ap_test_direct_iia": float(ap_row["test_direct_iia"]),
        "ap_test_mediator_iia": float(ap_row["test_mediator_iia"]),
        "ap_test_readout_acc": float(ap_row["test_readout_acc"]),
        "c_layer": best["C"]["layer"],
        "c_score": best["C"]["score"],
        "c_test_direct_iia": float(c_row["test_direct_iia"]),
        "c_test_ap_mediator_iia": float(c_row["test_ap_mediator_iia"]),
        "c_test_ap_readout_acc": float(c_row["test_ap_readout_acc"]),
        "c_test_at_mediator_iia": float(c_row["test_at_mediator_iia"]),
        "c_test_at_readout_acc": float(c_row["test_at_readout_acc"]),
    }

    return row


def save_csv(rows, path):
    if len(rows) == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows):
    metrics = [
        "at_test_iia",
        "ap_test_direct_iia",
        "ap_test_mediator_iia",
        "ap_test_readout_acc",
        "c_test_direct_iia",
        "c_test_ap_mediator_iia",
        "c_test_ap_readout_acc",
        "c_test_at_mediator_iia",
        "c_test_at_readout_acc",
    ]

    aggregate = []

    for method in METHODS:
        method_rows = []

        for row in rows:
            if row["method"] == method:
                method_rows.append(row)

        for metric in metrics:
            values = []

            for row in method_rows:
                values.append(float(row[metric]))

            aggregate.append({
                "method": method,
                "metric": metric,
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            })

    return aggregate


def print_aggregate(rows):
    print("\n================ 5-SEED SUMMARY ================")

    for method in METHODS:
        print(f"\n=== {method.upper()} ===")

        for row in rows:
            if row["method"] == method:
                print(f'{row["metric"]}: {float(row["mean"]):.4f} ± {float(row["std"]):.4f}')


def check_extra_args(extra_args):
    blocked = [
        "--seed",
        "--result-dir",
        "--ap-lambda-med",
        "--ap-lambda-readout",
        "--c-lambda-ap-med",
        "--c-lambda-ap-readout",
        "--c-lambda-at-med",
        "--c-lambda-at-readout",
    ]

    for arg in extra_args:
        for name in blocked:
            if arg == name or arg.startswith(name + "="):
                raise ValueError(f"{name} is controlled by this multi-seed wrapper and should not be passed manually.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-script", default="mcqa_joint_das_ap_at_c_layer_sweep.py")
    parser.add_argument("--output-root", default="results_standard_vs_gradual_5seeds_no_dims")
    parser.add_argument("--resume", action="store_true")
    args, extra_args = parser.parse_known_args()

    check_extra_args(extra_args)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for method in METHODS:
        weights = METHODS[method]

        for seed in SEEDS:
            result_dir = output_root / method / f"seed_{seed}"
            log_path = result_dir / "run.log"

            print("\n" + "=" * 80)
            print(f"METHOD = {method} | SEED = {seed}")
            print("=" * 80)

            completed = result_dir / "best_layers.txt"

            if args.resume and completed.exists():
                print(f"Skipping completed run: {result_dir}")
            else:
                command = [
                    sys.executable,
                    args.sweep_script,
                    "--result-dir", str(result_dir),
                    "--seed", str(seed),
                    "--ap-lambda-med", str(weights["ap_lambda_med"]),
                    "--ap-lambda-readout", str(weights["ap_lambda_readout"]),
                    "--c-lambda-ap-med", str(weights["c_lambda_ap_med"]),
                    "--c-lambda-ap-readout", str(weights["c_lambda_ap_readout"]),
                    "--c-lambda-at-med", str(weights["c_lambda_at_med"]),
                    "--c-lambda-at-readout", str(weights["c_lambda_at_readout"]),
                ]

                command.extend(extra_args)

                print(" ".join(command))
                run_command(command, log_path)

            row = collect_run_summary(method, seed, result_dir)
            all_rows.append(row)
            save_csv(all_rows, output_root / "seed_summary.csv")

    aggregate = aggregate_rows(all_rows)
    save_csv(aggregate, output_root / "aggregate_summary.csv")
    print_aggregate(aggregate)

    print(f"\nseed summary      = {output_root / 'seed_summary.csv'}")
    print(f"aggregate summary = {output_root / 'aggregate_summary.csv'}")


if __name__ == "__main__":
    main()
