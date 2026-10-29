import argparse
import csv
import shutil
import statistics
import torch
from pathlib import Path

from mcqa_gradual_das import fit_readout, set_seed
from mcqa_gradual_das_ap_cross_tokens import train_ap
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_c import train_c
from mcqa_joint_das_ap_at_c_balanced import parse_layers
from mcqa_load_randomized_ap_at_c_balanced_eval import build_all_banks
from mcqa_load_randomized_ap_at_c import build_ap_readout_datasets, build_at_readout_datasets, build_c_readout_datasets, build_shared_readout_banks
from mcqa_neural_net import load_gemma_model


DEFAULT_SEEDS = [0,1,2,3,4]

METHOD_ORDER = [
    "gradual2.00.02.0",
]

METHODS = {
    "gradual2.00.02.0": {
        "ap_lambda_med": 2.0,
        "ap_lambda_readout": 2.0,
        "c_lambda_ap_med": 0.0,
        "c_lambda_ap_readout": 0.0,
        "c_lambda_at_med": 2.0,
        "c_lambda_at_readout": 0.0,
    },
    # "standard": {
    #     "ap_lambda_med": 0.0,
    #     "ap_lambda_readout": 0.0,
    #     "c_lambda_ap_med": 0.0,
    #     "c_lambda_ap_readout": 0.0,
    #     "c_lambda_at_med": 0.0,
    #     "c_lambda_at_readout": 0.0,
    # },
    # "gradual2.02.02.0": {
    #     "ap_lambda_med": 2.0,
    #     "ap_lambda_readout": 2.0,
    #     "c_lambda_ap_med": 2.0,
    #     "c_lambda_ap_readout": 2.0,
    #     "c_lambda_at_med": 2.0,
    #     "c_lambda_at_readout": 2.0,
    # }
}


def parse_int_list(value):
    values = []
    for item in value.split(","):
        item = item.strip()
        if item != "":
            values.append(int(item))
    return values


def parse_token_positions(value):
    positions = []

    for item in value.split(","):
        item = item.strip()

        if item != "":
            positions.append(item)

    return positions


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


def read_csv(path):
    rows = []
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def all_files_exist(paths):
    for path in paths:
        if not Path(path).exists():
            return False
    return True


def find_layer_row(rows, layer):
    for row in rows:
        if int(row["layer"]) == int(layer):
            return row
    raise ValueError(f"Cannot find layer {layer}")


def ap_score(metrics, weights):
    direct_iia, mediator_iia, readout_acc = metrics
    active = [direct_iia]

    if weights["ap_lambda_med"] > 0:
        active.append(mediator_iia)

    if weights["ap_lambda_readout"] > 0:
        active.append(readout_acc)

    return min(active)


def c_score(metrics, weights):
    direct_iia, ap_mediator_iia, ap_readout_acc, at_mediator_iia, at_readout_acc = metrics
    active = [direct_iia]

    if weights["c_lambda_ap_med"] > 0:
        active.append(ap_mediator_iia)

    if weights["c_lambda_ap_readout"] > 0:
        active.append(ap_readout_acc)

    if weights["c_lambda_at_med"] > 0:
        active.append(at_mediator_iia)

    if weights["c_lambda_at_readout"] > 0:
        active.append(at_readout_acc)

    return min(active)


def parse_best_layer_from_best_layers(path, variable):
    for line in Path(path).read_text().splitlines():
        line = line.strip()

        if line.startswith(f"{variable} layer:"):
            left = line.split("|")[0]
            return int(left.split(":")[1].strip())

    raise ValueError(f"Cannot find {variable} layer in {path}")


def try_reuse_existing_at(seed_dir, shared_dir):
    for method in METHOD_ORDER:
        method_dir = seed_dir / method
        at_path = method_dir / "das_at.pt"
        at_readout_path = method_dir / "at_readout_LogisticRegression.pt"
        at_summary_path = method_dir / "at_layer_summary.csv"
        best_layers_path = method_dir / "best_layers.txt"

        required = [at_path, at_readout_path, at_summary_path, best_layers_path]

        if not all_files_exist(required):
            continue

        best_layer = parse_best_layer_from_best_layers(best_layers_path, "AT")
        at_rows = read_csv(at_summary_path)
        at_row = find_layer_row(at_rows, best_layer)

        final_at_path = shared_dir / "das_at.pt"
        final_at_readout_path = shared_dir / "at_readout_LogisticRegression.pt"
        final_summary_path = shared_dir / "at_layer_summary.csv"
        final_best_path = shared_dir / "best_at.txt"

        shared_dir.mkdir(parents=True, exist_ok=True)

        copy_file(at_path, final_at_path)
        copy_file(at_readout_path, final_at_readout_path)
        copy_file(at_summary_path, final_summary_path)

        best_score = float(at_row["cal_iia"])
        final_best_path.write_text(f"layer {best_layer} score {best_score:.8f}\n")

        print(f"[REUSE] Copied existing AT from {method_dir} into {shared_dir}")
        return final_at_path, final_at_readout_path, best_layer, best_score

    return None


def train_shared_at(model, tokenizer, das_banks, shared_readout, seed_dir, args, num_layers, seed):
    shared_dir = seed_dir / "shared_at"

    final_at_path = shared_dir / "das_at.pt"
    final_at_readout_path = shared_dir / "at_readout_LogisticRegression.pt"
    summary_path = shared_dir / "at_layer_summary.csv"
    best_path = shared_dir / "best_at.txt"

    required = [final_at_path, final_at_readout_path, summary_path, best_path]

    if all_files_exist(required):
        best_tokens = best_path.read_text().strip().split()
        best_layer = int(best_tokens[1])
        best_score = float(best_tokens[3])

        print(f"[RESUME] Shared AT complete for seed={seed}: layer={best_layer}, score={best_score:.4f}")
        return final_at_path, final_at_readout_path, best_layer, best_score

    reused = try_reuse_existing_at(seed_dir, shared_dir)

    if reused is not None:
        return reused

    shared_dir.mkdir(parents=True, exist_ok=True)

    requested_layers = parse_layers(args.at_layers)

    if requested_layers is None:
        at_layers = list(range(num_layers))
    else:
        at_layers = requested_layers

    at_rows = []
    best_score = -1.0
    best_layer = None
    best_at_path = None
    best_at_readout_path = None

    for layer in at_layers:
        set_seed(seed)

        layer_dir = shared_dir / "layers" / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        at_path = layer_dir / "das_at.pt"
        at_readout_path = layer_dir / "at_readout_LogisticRegression.pt"
        at_result_path = layer_dir / "result.pt"

        print("\n" + "=" * 100)
        print(f"SHARED AT | seed={seed} | layer={layer}")
        print("=" * 100)

        if all_files_exist([at_path, at_readout_path, at_result_path]):
            print(f"[RESUME] AT seed={seed} layer={layer} complete. Skipping training.")
            at_result = torch.load(at_result_path, map_location="cpu", weights_only=False)
        else:
            set_seed(seed)
            at_result = train_at(model, tokenizer, [layer], args.at_dim, args.ft_size, args.cal_size, args.te_size, args.at_epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.at_token_position, seed, str(at_path), das_banks["AT"], args.output_space)

            set_seed(seed)
            fit_readout(model=model, checkpoint=at_path, save_path=at_readout_path, bank_pair=build_at_readout_datasets(shared_readout), target="AT", list_token_positions=[args.at_token_position], batch_size=args.eval_batch_size, seed=seed, num_classes=26)

            torch.save(at_result, at_result_path)

        score = float(at_result["cal_iia"])

        row = {
            "layer": layer,
            "score": score,
            "cal_iia": float(at_result["cal_iia"]),
            "test_iia": float(at_result["test_iia"]),
            "best_epoch": int(at_result["best_epoch"]),
        }

        found = False
        for i in range(len(at_rows)):
            if int(at_rows[i]["layer"]) == int(layer):
                at_rows[i] = row
                found = True
                break

        if not found:
            at_rows.append(row)

        save_csv(at_rows, summary_path)

        if score >= best_score:
            best_score = score
            best_layer = layer
            best_at_path = at_path
            best_at_readout_path = at_readout_path

    copy_file(best_at_path, final_at_path)
    copy_file(best_at_readout_path, final_at_readout_path)
    best_path.write_text(f"layer {best_layer} score {best_score:.8f}\n")

    print(f"\n[SHARED AT COMPLETE] seed={seed} layer={best_layer} score={best_score:.4f}")
    return final_at_path, final_at_readout_path, best_layer, best_score


def method_complete(method_dir):
    required = [
        method_dir / "das_ap.pt",
        method_dir / "ap_readout_LogisticRegression.pt",
        method_dir / "das_c.pt",
        method_dir / "c_readout_LogisticRegression.pt",
        method_dir / "ap_layer_summary.csv",
        method_dir / "c_layer_summary.csv",
        method_dir / "best_layers.txt",
    ]

    return all_files_exist(required)


def train_method(model, tokenizer, das_banks, shared_readout, seed_dir, shared_at_path, shared_at_readout_path, best_at_layer, method, args, seed):
    method_dir = seed_dir / method

    if method_complete(method_dir):
        print(f"[RESUME] {method} seed={seed} complete. Skipping AP/C training.")
        return

    method_dir.mkdir(parents=True, exist_ok=True)

    weights = METHODS[method]

    final_ap_path = method_dir / "das_ap.pt"
    final_ap_readout_path = method_dir / "ap_readout_LogisticRegression.pt"
    final_c_path = method_dir / "das_c.pt"
    final_c_readout_path = method_dir / "c_readout_LogisticRegression.pt"

    ap_summary_path = method_dir / "ap_layer_summary.csv"
    c_summary_path = method_dir / "c_layer_summary.csv"
    best_layers_path = method_dir / "best_layers.txt"

    ap_token_positions = parse_token_positions(args.ap_token_position)

    if len(ap_token_positions) == 0:
        raise ValueError("--ap-token-position must contain at least one token position")

    set_seed(seed)

    ap_path = method_dir / "das_ap.pt"
    ap_readout_path = method_dir / "ap_readout_LogisticRegression.pt"
    ap_result_path = method_dir / "ap_result.pt"

    print("\n" + "=" * 100)
    print(f"{method.upper()} AP | seed={seed} | cross-token positions={ap_token_positions}")
    print("=" * 100)

    if all_files_exist([ap_path, ap_readout_path, ap_result_path, ap_summary_path]):
        print(f"[RESUME] {method} AP seed={seed} complete. Skipping AP training.")
        ap_result = torch.load(ap_result_path, map_location="cpu", weights_only=False)
    else:
        ap_result = train_ap(model=model, tokenizer=tokenizer, at_checkpoint=str(shared_at_path), at_readout_checkpoint=str(shared_at_readout_path), layers=None, subspace_dim=args.ap_dim, ft_size=args.ft_size, cal_size=args.cal_size, te_size=args.te_size, epochs=args.ap_epochs, lr=args.lr, lambda_med=weights["ap_lambda_med"], lambda_readout=weights["ap_lambda_readout"], train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, list_ap_token_positions=ap_token_positions, at_token_position=args.at_token_position, seed=seed, save_path=str(ap_path), banks=das_banks["AP"], output_space=args.output_space)

        set_seed(seed)
        fit_readout(model=model, checkpoint=ap_path, save_path=ap_readout_path, bank_pair=build_ap_readout_datasets(shared_readout), target="AP", list_token_positions=ap_token_positions, batch_size=args.eval_batch_size, seed=seed, num_classes=4)

        torch.save(ap_result, ap_result_path)

    cal_direct_iia, cal_mediator_iia, cal_readout_acc = ap_result["cal"]
    test_direct_iia, test_mediator_iia, test_readout_acc = ap_result["test"]

    best_ap_score = ap_score(ap_result["cal"], weights)
    best_ap_layer = int(ap_result["layer"])

    ap_rows = [{
        "layer": best_ap_layer,
        "score": best_ap_score,
        "cal_direct_iia": float(cal_direct_iia),
        "cal_mediator_iia": float(cal_mediator_iia),
        "cal_readout_acc": float(cal_readout_acc),
        "test_direct_iia": float(test_direct_iia),
        "test_mediator_iia": float(test_mediator_iia),
        "test_readout_acc": float(test_readout_acc),
        "best_epoch": int(ap_result["best_epoch"]),
    }]

    save_csv(ap_rows, ap_summary_path)

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
        raise ValueError(f"No valid C layers before AP layer {best_ap_layer} and AT layer {best_at_layer}")

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
        c_result_path = layer_dir / "result.pt"

        print("\n" + "=" * 100)
        print(f"{method.upper()} C | seed={seed} | layer={layer}")
        print("=" * 100)

        if all_files_exist([c_path, c_readout_path, c_result_path]):
            print(f"[RESUME] {method} C seed={seed} layer={layer} complete. Skipping training.")
            c_result = torch.load(c_result_path, map_location="cpu", weights_only=False)
        else:
            c_result = train_c(model, tokenizer, str(final_ap_path), str(final_ap_readout_path), str(shared_at_path), str(shared_at_readout_path), [layer], args.c_dim, args.ft_size, args.cal_size, args.te_size, args.c_epochs, args.lr, weights["c_lambda_ap_med"], weights["c_lambda_ap_readout"], weights["c_lambda_at_med"], weights["c_lambda_at_readout"], args.train_batch_size, args.eval_batch_size, args.c_token_position, seed, str(c_path), das_banks["C"], args.output_space)

            set_seed(seed)

            colors = shared_readout[0]["color_to_class"]
            extra = {"color_to_class": colors}
            fit_readout(model=model, checkpoint=c_path, save_path=c_readout_path, bank_pair=build_c_readout_datasets(shared_readout), target="C", list_token_positions=[args.c_token_position], batch_size=args.eval_batch_size, seed=seed, num_classes=len(colors), extra=extra)

            torch.save(c_result, c_result_path)

        cal_direct_iia, cal_ap_mediator_iia, cal_ap_readout_acc, cal_at_mediator_iia, cal_at_readout_acc = c_result["cal"]
        test_direct_iia, test_ap_mediator_iia, test_ap_readout_acc, test_at_mediator_iia, test_at_readout_acc = c_result["test"]

        score = c_score(c_result["cal"], weights)

        row = {
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
        }

        found = False
        for i in range(len(c_rows)):
            if int(c_rows[i]["layer"]) == int(layer):
                c_rows[i] = row
                found = True
                break

        if not found:
            c_rows.append(row)

        save_csv(c_rows, c_summary_path)

        if score >= best_c_score:
            best_c_score = score
            best_c_layer = layer
            best_c_path = c_path
            best_c_readout_path = c_readout_path

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

    ap_layer = parse_best_layer_from_best_layers(method_dir / "best_layers.txt", "AP")
    c_layer = parse_best_layer_from_best_layers(method_dir / "best_layers.txt", "C")

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
        "ap_score": float(ap_row["score"]),
        "ap_test_direct_iia": float(ap_row["test_direct_iia"]),
        "ap_test_mediator_iia": float(ap_row["test_mediator_iia"]),
        "ap_test_readout_acc": float(ap_row["test_readout_acc"]),
        "c_layer": c_layer,
        "c_score": float(c_row["score"]),
        "c_test_direct_iia": float(c_row["test_direct_iia"]),
        "c_test_ap_mediator_iia": float(c_row["test_ap_mediator_iia"]),
        "c_test_ap_readout_acc": float(c_row["test_ap_readout_acc"]),
        "c_test_at_mediator_iia": float(c_row["test_at_mediator_iia"]),
        "c_test_at_readout_acc": float(c_row["test_at_readout_acc"]),
    }


def seed_complete(seed_dir):
    shared_dir = seed_dir / "shared_at"

    if not all_files_exist([
        shared_dir / "das_at.pt",
        shared_dir / "at_readout_LogisticRegression.pt",
        shared_dir / "at_layer_summary.csv",
        shared_dir / "best_at.txt",
    ]):
        return False

    for method in METHOD_ORDER:
        if not method_complete(seed_dir / method):
            return False

    return True

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

    for method in METHOD_ORDER:
        method_rows = []

        for row in rows:
            if row["method"] == method:
                method_rows.append(row)

        if len(method_rows) == 0:
            continue

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--output-root", default="results_mcqa_many_seeds_change_all")
    parser.add_argument("--seeds", default="")

    parser.add_argument("--ft-size", type=int, default=800)
    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--te-size", type=int, default=400)

    parser.add_argument("--at-dim", type=int, default=128)
    parser.add_argument("--ap-dim", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=128)

    parser.add_argument("--at-layers", default="23,24,25")
    parser.add_argument("--ap-layers", default="14")  # compatibility only; cross-token AP trainer uses AP_LAYER=14
    parser.add_argument("--c-layers", default="0")

    parser.add_argument("--at-epochs", type=int, default=10)
    parser.add_argument("--ap-epochs", type=int, default=10)
    parser.add_argument("--c-epochs", type=int, default=30)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)

    parser.add_argument("--readout-train-size", type=int, default=1300)
    parser.add_argument("--readout-val-size", type=int, default=520)

    parser.add_argument("--at-token-position", default="last_token")
    parser.add_argument("--ap-token-position", default="color_0,color_1,color_2,color_3")
    parser.add_argument("--c-token-position", default="color")

    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--change-ratio", type=float, default=1.0)

    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)

    if len(seeds) == 0:
        seeds = DEFAULT_SEEDS

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    model, tokenizer = load_gemma_model()
    num_layers = model.config.num_hidden_layers

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("model device:", next(model.parameters()).device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("allocated GB:", torch.cuda.memory_allocated(0) / 1024**3)
        print("reserved GB:", torch.cuda.memory_reserved(0) / 1024**3)

    rows = []

    for seed in seeds:
        print("\n" + "#" * 100)
        print(f"TRAIN SEED = {seed}")
        print("#" * 100)

        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        if seed_complete(seed_dir):
            print(f"[RESUME] Seed {seed} already complete. Skipping entire seed.")

            for method in METHOD_ORDER:
                row = collect_method_summary(seed_dir, method, seed)
                rows.append(row)

            save_csv(rows, output_root / "seed_summary.csv")
            continue

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
