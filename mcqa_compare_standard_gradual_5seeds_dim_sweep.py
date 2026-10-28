import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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


def parse_int_list(value):
    values = []

    for item in value.split(","):
        item = item.strip()

        if item != "":
            values.append(int(item))

    if len(values) == 0:
        raise ValueError("Dimension list cannot be empty.")

    return values


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


def collect_run_summary(method, seed, at_dim, c_dim, result_dir):
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
        "at_dim": at_dim,
        "c_dim": c_dim,
        "at_layer": best["AT"]["layer"],
        "at_cal_iia": float(at_row["cal_iia"]),
        "at_test_iia": float(at_row["test_iia"]),
        "ap_layer": best["AP"]["layer"],
        "ap_cal_direct_iia": float(ap_row["cal_direct_iia"]),
        "ap_cal_mediator_iia": float(ap_row["cal_mediator_iia"]),
        "ap_cal_readout_acc": float(ap_row["cal_readout_acc"]),
        "ap_test_direct_iia": float(ap_row["test_direct_iia"]),
        "ap_test_mediator_iia": float(ap_row["test_mediator_iia"]),
        "ap_test_readout_acc": float(ap_row["test_readout_acc"]),
        "c_layer": best["C"]["layer"],
        "c_cal_direct_iia": float(c_row["cal_direct_iia"]),
        "c_cal_ap_mediator_iia": float(c_row["cal_ap_mediator_iia"]),
        "c_cal_ap_readout_acc": float(c_row["cal_ap_readout_acc"]),
        "c_cal_at_mediator_iia": float(c_row["cal_at_mediator_iia"]),
        "c_cal_at_readout_acc": float(c_row["cal_at_readout_acc"]),
        "c_test_direct_iia": float(c_row["test_direct_iia"]),
        "c_test_ap_mediator_iia": float(c_row["test_ap_mediator_iia"]),
        "c_test_ap_readout_acc": float(c_row["test_ap_readout_acc"]),
        "c_test_at_mediator_iia": float(c_row["test_at_mediator_iia"]),
        "c_test_at_readout_acc": float(c_row["test_at_readout_acc"]),
        "result_dir": str(result_dir),
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


def check_extra_args(extra_args):
    blocked = [
        "--seed",
        "--result-dir",
        "--at-dim",
        "--c-dim",
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
                raise ValueError(f"{name} is controlled by this wrapper and should not be passed manually.")


def build_command(args, extra_args, method, seed, at_dim, c_dim, result_dir):
    weights = METHODS[method]

    command = [
        sys.executable,
        args.sweep_script,
        "--result-dir", str(result_dir),
        "--seed", str(seed),
        "--at-dim", str(at_dim),
        "--c-dim", str(c_dim),
        "--ap-lambda-med", str(weights["ap_lambda_med"]),
        "--ap-lambda-readout", str(weights["ap_lambda_readout"]),
        "--c-lambda-ap-med", str(weights["c_lambda_ap_med"]),
        "--c-lambda-ap-readout", str(weights["c_lambda_ap_readout"]),
        "--c-lambda-at-med", str(weights["c_lambda_at_med"]),
        "--c-lambda-at-readout", str(weights["c_lambda_at_readout"]),
    ]

    command.extend(extra_args)
    return command


def choose_best_at(rows):
    best = None

    for row in rows:
        if best is None or row["at_cal_iia"] > best["at_cal_iia"]:
            best = row

    return best


def choose_best_c(rows):
    best = None

    for row in rows:
        if best is None or row["c_cal_direct_iia"] > best["c_cal_direct_iia"]:
            best = row

    return best


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


def print_selected(rows):
    print("\n================ SELECTED CONFIGS ================")

    for row in rows:
        print(
            f'{row["method"]} seed={row["seed"]} | '
            f'AT dim={row["at_dim"]} layer={row["at_layer"]} cal={row["at_cal_iia"]:.4f} test={row["at_test_iia"]:.4f} | '
            f'C dim={row["c_dim"]} layer={row["c_layer"]} cal={row["c_cal_direct_iia"]:.4f} test={row["c_test_direct_iia"]:.4f}'
        )


def print_aggregate(rows):
    print("\n================ 5-SEED SUMMARY ================")

    for method in METHODS:
        print(f"\n=== {method.upper()} ===")

        for row in rows:
            if row["method"] == method:
                print(f'{row["metric"]}: {float(row["mean"]):.4f} ± {float(row["std"]):.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-script", default="mcqa_joint_das_ap_at_c_layer_sweep.py")
    parser.add_argument("--output-root", default="results_standard_vs_gradual_5seeds_dim_sweep")
    parser.add_argument("--at-dims", default="32,64,128,256")
    parser.add_argument("--c-dims", default="32,64,128,256")
    parser.add_argument("--resume", action="store_true")
    args, extra_args = parser.parse_known_args()

    check_extra_args(extra_args)

    at_dims = parse_int_list(args.at_dims)
    c_dims = parse_int_list(args.c_dims)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_candidate_rows = []
    selected_rows = []

    for method in METHODS:
        for seed in SEEDS:
            print("\n" + "#" * 100)
            print(f"METHOD = {method} | SEED = {seed}")
            print("#" * 100)

            at_candidate_rows = []

            for at_dim in at_dims:
                result_dir = output_root / method / f"seed_{seed}" / "at_sweep" / f"at_dim_{at_dim}"

                print("\n" + "=" * 80)
                print(f"AT DIM SWEEP | method={method} | seed={seed} | at_dim={at_dim}")
                print("=" * 80)

                completed = result_dir / "best_layers.txt"

                if args.resume and completed.exists():
                    print(f"Skipping completed run: {result_dir}")
                else:
                    command = build_command(args, extra_args, method, seed, at_dim, c_dims[0], result_dir)
                    print(" ".join(command))
                    run_command(command, result_dir / "run.log")

                row = collect_run_summary(method, seed, at_dim, c_dims[0], result_dir)
                row["stage"] = "at_sweep"
                at_candidate_rows.append(row)
                all_candidate_rows.append(row)
                save_csv(all_candidate_rows, output_root / "all_candidate_runs.csv")

            best_at = choose_best_at(at_candidate_rows)
            selected_at_dim = int(best_at["at_dim"])

            print(
                f"\n[SELECTED AT] method={method} seed={seed} "
                f"dim={selected_at_dim} layer={best_at['at_layer']} "
                f"cal_iia={best_at['at_cal_iia']:.4f}"
            )

            c_candidate_rows = []

            for c_dim in c_dims:
                result_dir = output_root / method / f"seed_{seed}" / "c_sweep" / f"at_dim_{selected_at_dim}_c_dim_{c_dim}"

                print("\n" + "=" * 80)
                print(f"C DIM SWEEP | method={method} | seed={seed} | at_dim={selected_at_dim} | c_dim={c_dim}")
                print("=" * 80)

                completed = result_dir / "best_layers.txt"

                if args.resume and completed.exists():
                    print(f"Skipping completed run: {result_dir}")
                else:
                    command = build_command(args, extra_args, method, seed, selected_at_dim, c_dim, result_dir)
                    print(" ".join(command))
                    run_command(command, result_dir / "run.log")

                row = collect_run_summary(method, seed, selected_at_dim, c_dim, result_dir)
                row["stage"] = "c_sweep"
                c_candidate_rows.append(row)
                all_candidate_rows.append(row)
                save_csv(all_candidate_rows, output_root / "all_candidate_runs.csv")

            best_c = choose_best_c(c_candidate_rows)

            selected = dict(best_c)
            selected["selected_at_dim"] = selected_at_dim
            selected["selected_at_cal_iia"] = float(best_at["at_cal_iia"])
            selected["selected_at_test_iia"] = float(best_at["at_test_iia"])
            selected["selected_at_layer"] = int(best_at["at_layer"])
            selected_rows.append(selected)

            save_csv(selected_rows, output_root / "selected_runs.csv")

    aggregate = aggregate_rows(selected_rows)
    save_csv(aggregate, output_root / "aggregate_summary.csv")

    print_selected(selected_rows)
    print_aggregate(aggregate)

    print(f"\nall candidates     = {output_root / 'all_candidate_runs.csv'}")
    print(f"selected runs      = {output_root / 'selected_runs.csv'}")
    print(f"aggregate summary  = {output_root / 'aggregate_summary.csv'}")


if __name__ == "__main__":
    main()
