import argparse
import csv
import shutil
import statistics
from pathlib import Path

from mcqa_gradual_das import set_seed
from mcqa_gradual_das_ap import train_ap
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_c import train_c
from mcqa_joint_das_ap_at_c_balanced import fit_readout, parse_layers
from mcqa_load_randomized_ap_at_c_balanced_eval import build_all_banks
from mcqa_load_randomized_ap_at_c import build_ap_readout_datasets, build_at_readout_datasets, build_c_readout_datasets, build_shared_readout_banks
from mcqa_neural_net import load_gemma_model


SEEDS = [0, 1, 2, 3, 4]
METHOD_ORDER = ["gradual2.02.04.0", "standard", "gradual2.02.02.0"]

METHODS = {
    "gradual2.02.04.0": {
        "ap_lambda_med": 2.0,
        "ap_lambda_readout": 2.0,
        "c_lambda_ap_med": 2.0,
        "c_lambda_ap_readout": 2.0,
        "c_lambda_at_med": 4.0,
        "c_lambda_at_readout": 4.0,
    },
    "standard": {
        "ap_lambda_med": 0.0,
        "ap_lambda_readout": 0.0,
        "c_lambda_ap_med": 0.0,
        "c_lambda_ap_readout": 0.0,
        "c_lambda_at_med": 0.0,
        "c_lambda_at_readout": 0.0,
    },
    "gradual2.02.02.0": {
        "ap_lambda_med": 2.0,
        "ap_lambda_readout": 2.0,
        "c_lambda_ap_med": 2.0,
        "c_lambda_ap_readout": 2.0,
        "c_lambda_at_med": 2.0,
        "c_lambda_at_readout": 2.0,
    }
}


def copy_file(source, destination):
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def save_csv(rows, path):
    if len(rows) == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def required_files_exist(paths):
    for path in paths:
        if not Path(path).exists():
            return False

    return True


def read_csv(path):
    rows = []

    with open(path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


def find_layer_row(rows, layer):
    for row in rows:
        if int(row["layer"]) == int(layer):
            return row

    raise ValueError(f"Cannot find layer {layer}")


def ap_score(metrics, lambda_med, lambda_readout):
    direct_iia, mediator_iia, readout_acc = metrics
    active = [direct_iia]

    if lambda_med > 0:
        active.append(mediator_iia)

    if lambda_readout > 0:
        active.append(readout_acc)

    return min(active)


def c_score(metrics, lambda_ap_med, lambda_ap_readout, lambda_at_med, lambda_at_readout):
    direct_iia, ap_mediator_iia, ap_readout_acc, at_mediator_iia, at_readout_acc = metrics
    active = [direct_iia]

    if lambda_ap_med > 0:
        active.append(ap_mediator_iia)

    if lambda_ap_readout > 0:
        active.append(ap_readout_acc)

    if lambda_at_med > 0:
        active.append(at_mediator_iia)

    if lambda_at_readout > 0:
        active.append(at_readout_acc)

    return min(active)


def train_shared_at(model, tokenizer, das_banks, shared_readout, seed_dir, args, num_layers, seed):
    shared_dir = seed_dir / "shared_at"
    final_at_path = shared_dir / "das_at.pt"
    final_at_readout_path = shared_dir / "at_readout_LogisticRegression.pt"
    summary_path = shared_dir / "at_layer_summary.csv"
    best_path = shared_dir / "best_at.txt"

    required = [final_at_path, final_at_readout_path, summary_path, best_path]

    if required_files_exist(required):
        print(f"[RESUME] Shared AT complete for seed={seed}. Reusing {final_at_path}")
        best_tokens = best_path.read_text().strip().split()
        best_layer = int(best_tokens[1])
        best_score = float(best_tokens[3])
        return final_at_path, final_at_readout_path, best_layer, best_score

    print(f"[TRAIN] Shared AT incomplete for seed={seed}. Retraining AT from scratch.")

    if shared_dir.exists():
        shutil.rmtree(shared_dir)

    shared_dir.mkdir(parents=True, exist_ok=True)

    requested_at_layers = parse_layers(args.at_layers)

    if requested_at_layers is None:
        at_layers = list(range(num_layers))
    else:
        at_layers = requested_at_layers

    at_rows = []
    best_at_score = -1.0
    best_at_layer = None
    best_at_path = None
    best_at_readout_path = None

    for layer in at_layers:
        set_seed(seed)

        layer_dir = shared_dir / "layers" / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        at_path = layer_dir / "das_at.pt"
        at_readout_path = layer_dir / "at_readout_LogisticRegression.pt"

        print(f"\n=== SHARED AT | seed={seed} | layer={layer} ===")

        at_result = train_at(model, tokenizer, [layer], args.at_dim, args.ft_size, args.cal_size, args.te_size, args.at_epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.at_token_position, seed, str(at_path), das_banks["AT"], args.output_space)

        set_seed(seed)
        fit_readout(model, at_path, at_readout_path, build_at_readout_datasets(shared_readout), "AT", args.at_token_position, args.eval_batch_size, seed, 26)

        score = float(at_result["cal_iia"])

        at_rows.append({
            "layer": layer,
            "score": score,
            "cal_iia": float(at_result["cal_iia"]),
            "test_iia": float(at_result["test_iia"]),
            "best_epoch": int(at_result["best_epoch"]),
        })

        if score > best_at_score:
            best_at_score = score
            best_at_layer = layer
            best_at_path = at_path
            best_at_readout_path = at_readout_path

    save_csv(at_rows, summary_path)
    copy_file(best_at_path, final_at_path)
    copy_file(best_at_readout_path, final_at_readout_path)
    best_path.write_text(f"layer {best_at_layer} score {best_at_score:.8f}\n")

    print(f"\n[SHARED AT BEST] seed={seed} layer={best_at_layer} score={best_at_score:.4f}")
    return final_at_path, final_at_readout_path, best_at_layer, best_at_score


def train_method(model, tokenizer, das_banks, shared_readout, seed_dir, shared_at_path, shared_at_readout_path, best_at_layer, method, args, seed):
    method_dir = seed_dir / method

    final_ap_path = method_dir / "das_ap.pt"
    final_ap_readout_path = method_dir / "ap_readout_LogisticRegression.pt"
    final_c_path = method_dir / "das_c.pt"
    final_c_readout_path = method_dir / "c_readout_LogisticRegression.pt"
    ap_summary_path = method_dir / "ap_layer_summary.csv"
    c_summary_path = method_dir / "c_layer_summary.csv"
    best_layers_path = method_dir / "best_layers.txt"

    required = [final_ap_path, final_ap_readout_path, final_c_path, final_c_readout_path, ap_summary_path, c_summary_path, best_layers_path]

    if required_files_exist(required):
        print(f"[RESUME] {method} complete for seed={seed}. Skipping.")
        return

    print(f"[TRAIN] {method} incomplete for seed={seed}. Retraining {method} AP/C from scratch.")

    if method_dir.exists():
        shutil.rmtree(method_dir)

    method_dir.mkdir(parents=True, exist_ok=True)
    weights = METHODS[method]

    requested_ap_layers = parse_layers(args.ap_layers)

    if requested_ap_layers is None:
        ap_layers = list(range(best_at_layer))
    else:
        ap_layers = []

        for layer in requested_ap_layers:
            if layer < best_at_layer:
                ap_layers.append(layer)

    if len(ap_layers) == 0:
        raise ValueError("No valid AP layers before selected AT layer.")

    ap_rows = []
    best_ap_score = -1.0
    best_ap_layer = None
    best_ap_path = None
    best_ap_readout_path = None

    for layer in ap_layers:
        set_seed(seed)

        layer_dir = method_dir / "ap_layers" / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        ap_path = layer_dir / "das_ap.pt"
        ap_readout_path = layer_dir / "ap_readout_LogisticRegression.pt"

        print(f"\n=== {method.upper()} AP | seed={seed} | layer={layer} ===")

        ap_result = train_ap(model, tokenizer, str(shared_at_path), str(shared_at_readout_path), [layer], args.ap_dim, args.ft_size, args.cal_size, args.te_size, args.ap_epochs, args.lr, weights["ap_lambda_med"], weights["ap_lambda_readout"], args.train_batch_size, args.eval_batch_size, args.ap_token_position, seed, str(ap_path), das_banks["AP"], args.output_space)

        set_seed(seed)
        fit_readout(model, ap_path, ap_readout_path, build_ap_readout_datasets(shared_readout), "AP", args.ap_token_position, args.eval_batch_size, seed, 4)

        direct_iia, mediator_iia, readout_acc = ap_result["cal"]
        test_direct_iia, test_mediator_iia, test_readout_acc = ap_result["test"]
        score = ap_score(ap_result["cal"], weights["ap_lambda_med"], weights["ap_lambda_readout"])

        ap_rows.append({
            "layer": layer,
            "score": score,
            "cal_direct_iia": float(direct_iia),
            "cal_mediator_iia": float(mediator_iia),
            "cal_readout_acc": float(readout_acc),
            "test_direct_iia": float(test_direct_iia),
            "test_mediator_iia": float(test_mediator_iia),
            "test_readout_acc": float(test_readout_acc),
            "best_epoch": int(ap_result["best_epoch"]),
        })

        if score > best_ap_score:
            best_ap_score = score
            best_ap_layer = layer
            best_ap_path = ap_path
            best_ap_readout_path = ap_readout_path

    save_csv(ap_rows, ap_summary_path)
    copy_file(best_ap_path, final_ap_path)
    copy_file(best_ap_readout_path, final_ap_readout_path)

    requested_c_layers = parse_layers(args.c_layers)
    max_c_layer = min(best_ap_layer, best_at_layer)

    if requested_c_layers is None:
        c_layers = list(range(max_c_layer))
    else:
        c_layers = []

        for layer in requested_c_layers:
            if layer < max_c_layer:
                c_layers.append(layer)

    if len(c_layers) == 0:
        raise ValueError("No valid C layers before selected AP and AT layers.")

    c_rows = []
    best_c_score = -1.0
    best_c_layer = None
    best_c_path = None
    best_c_readout_path = None

    for layer in c_layers:
        set_seed(seed)

        layer_dir = method_dir / "c_layers" / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        c_path = layer_dir / "das_c.pt"
        c_readout_path = layer_dir / "c_readout_LogisticRegression.pt"

        print(f"\n=== {method.upper()} C | seed={seed} | layer={layer} ===")

        c_result = train_c(model, tokenizer, str(final_ap_path), str(final_ap_readout_path), str(shared_at_path), str(shared_at_readout_path), [layer], args.c_dim, args.ft_size, args.cal_size, args.te_size, args.c_epochs, args.lr, weights["c_lambda_ap_med"], weights["c_lambda_ap_readout"], weights["c_lambda_at_med"], weights["c_lambda_at_readout"], args.train_batch_size, args.eval_batch_size, args.c_token_position, seed, str(c_path), das_banks["C"], args.output_space)

        set_seed(seed)

        colors = shared_readout[0]["color_to_class"]
        extra = {"color_to_class": colors}
        fit_readout(model, c_path, c_readout_path, build_c_readout_datasets(shared_readout), "C", args.c_token_position, args.eval_batch_size, seed, len(colors), extra)

        cal_direct_iia, cal_ap_mediator_iia, cal_ap_readout_acc, cal_at_mediator_iia, cal_at_readout_acc = c_result["cal"]
        test_direct_iia, test_ap_mediator_iia, test_ap_readout_acc, test_at_mediator_iia, test_at_readout_acc = c_result["test"]

        score = c_score(c_result["cal"], weights["c_lambda_ap_med"], weights["c_lambda_ap_readout"], weights["c_lambda_at_med"], weights["c_lambda_at_readout"])

        c_rows.append({
            "layer": layer,
            "score": score,
            "cal_direct_iia": float(cal_direct_iia),
            "cal_ap_mediator_iia": float(cal_ap_mediator_iia),
            "cal_ap_readout_acc": float(cal_ap_readout_acc),
            "cal_at_mediator_iia": float(cal_at_mediator_iia),
            "cal_at_readout_acc": float(cal_at_readout_acc),
            "test_direct_iia": float(test_direct_iia),
            "test_ap_mediator_iia": float(test_ap_mediator_iia),
            "test_ap_readout_acc": float(test_ap_readout_acc),
            "test_at_mediator_iia": float(test_at_mediator_iia),
            "test_at_readout_acc": float(test_at_readout_acc),
            "best_epoch": int(c_result["best_epoch"]),
        })

        if score > best_c_score:
            best_c_score = score
            best_c_layer = layer
            best_c_path = c_path
            best_c_readout_path = c_readout_path

    save_csv(c_rows, c_summary_path)
    copy_file(best_c_path, final_c_path)
    copy_file(best_c_readout_path, final_c_readout_path)

    best_text = []
    best_text.append(f"AT layer: {best_at_layer}")
    best_text.append(f"AP layer: {best_ap_layer} | score: {best_ap_score:.6f}")
    best_text.append(f"C layer: {best_c_layer} | score: {best_c_score:.6f}")
    best_layers_path.write_text("\n".join(best_text) + "\n")

    print(f"\n[{method.upper()} COMPLETE] seed={seed}")


def collect_method_summary(seed_dir, method, seed):
    method_dir = seed_dir / method
    shared_dir = seed_dir / "shared_at"

    at_rows = read_csv(shared_dir / "at_layer_summary.csv")
    ap_rows = read_csv(method_dir / "ap_layer_summary.csv")
    c_rows = read_csv(method_dir / "c_layer_summary.csv")

    best_at_tokens = (shared_dir / "best_at.txt").read_text().strip().split()
    at_layer = int(best_at_tokens[1])
    at_score = float(best_at_tokens[3])

    best_lines = (method_dir / "best_layers.txt").read_text().splitlines()
    ap_layer = int(best_lines[1].split("|")[0].split(":")[1].strip())
    c_layer = int(best_lines[2].split("|")[0].split(":")[1].strip())

    at_row = find_layer_row(at_rows, at_layer)
    ap_row = find_layer_row(ap_rows, ap_layer)
    c_row = find_layer_row(c_rows, c_layer)

    return {
        "method": method,
        "seed": seed,
        "at_layer": at_layer,
        "at_score": at_score,
        "at_test_iia": float(at_row["test_iia"]),
        "ap_layer": ap_layer,
        "ap_test_direct_iia": float(ap_row["test_direct_iia"]),
        "ap_test_mediator_iia": float(ap_row["test_mediator_iia"]),
        "ap_test_readout_acc": float(ap_row["test_readout_acc"]),
        "c_layer": c_layer,
        "c_test_direct_iia": float(c_row["test_direct_iia"]),
        "c_test_ap_mediator_iia": float(c_row["test_ap_mediator_iia"]),
        "c_test_ap_readout_acc": float(c_row["test_ap_readout_acc"]),
        "c_test_at_mediator_iia": float(c_row["test_at_mediator_iia"]),
        "c_test_at_readout_acc": float(c_row["test_at_readout_acc"]),
    }


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

    output = []

    for method in METHOD_ORDER:
        method_rows = []

        for row in rows:
            if row["method"] == method:
                method_rows.append(row)

        for metric in metrics:
            values = []

            for row in method_rows:
                values.append(float(row[metric]))

            output.append({
                "method": method,
                "metric": metric,
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            })

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--output-root", default="results_standard_vs_gradual_5seeds_shared_at")
    parser.add_argument("--ft-size", type=int, default=800)
    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--te-size", type=int, default=400)
    parser.add_argument("--at-dim", type=int, default=128)
    parser.add_argument("--ap-dim", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=128)
    parser.add_argument("--at-layers", default="23,24,25")
    parser.add_argument("--ap-layers", default="17")
    parser.add_argument("--c-layers", default="0,1")
    parser.add_argument("--at-epochs", type=int, default=10)
    parser.add_argument("--ap-epochs", type=int, default=10)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--readout-train-size", type=int, default=1300)
    parser.add_argument("--readout-val-size", type=int, default=520)
    parser.add_argument("--at-token-position", default="last_token")
    parser.add_argument("--ap-token-position", default="color_0,color_1,color_2,color_3")
    parser.add_argument("--c-token-position", default="color")
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--change-ratio", type=float, default=0.5)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    model, tokenizer = load_gemma_model()
    num_layers = model.config.num_hidden_layers

    rows = []

    for seed in SEEDS:
        print("\n" + "#" * 100)
        print(f"SEED = {seed}")
        print("#" * 100)

        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)
        das_banks = build_all_banks(model, tokenizer, args.data_path, args.ft_size, args.cal_size, args.te_size, args.eval_batch_size, seed, args.change_ratio)

        set_seed(seed)
        shared_readout = build_shared_readout_banks(model, tokenizer, args.data_path, args.readout_train_size, args.readout_val_size, args.eval_batch_size, seed)

        shared_at_path, shared_at_readout_path, best_at_layer, _ = train_shared_at(model, tokenizer, das_banks, shared_readout, seed_dir, args, num_layers, seed)

        for method in METHOD_ORDER:
            train_method(model, tokenizer, das_banks, shared_readout, seed_dir, shared_at_path, shared_at_readout_path, best_at_layer, method, args, seed)

            row = collect_method_summary(seed_dir, method, seed)
            rows.append(row)
            save_csv(rows, output_root / "seed_summary.csv")

    aggregate = aggregate_rows(rows)
    save_csv(aggregate, output_root / "aggregate_summary.csv")

    print("\n================ FINAL SUMMARY ================")

    for row in aggregate:
        print(f"{row['method']} | {row['metric']} = {row['mean']:.4f} ± {row['std']:.4f}")


if __name__ == "__main__":
    main()
