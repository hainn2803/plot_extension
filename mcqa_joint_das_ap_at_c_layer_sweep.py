import argparse
import csv
import shutil
from pathlib import Path

from mcqa_gradual_das_ap import train_ap
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_c import train_c
from mcqa_joint_das_ap_at_c_balanced import fit_readout, parse_layers
from mcqa_load_randomized_ap_at_c import build_all_banks, build_ap_readout_datasets, build_at_readout_datasets, build_c_readout_datasets, build_shared_readout_banks
from mcqa_neural_net import load_gemma_model


def copy_file(source, destination):
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


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


def save_summary(rows, path):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_standard_newrun_128")
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
    parser.add_argument("--c-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ap-lambda-med", type=float, default=0.0)
    parser.add_argument("--ap-lambda-readout", type=float, default=0.0)
    parser.add_argument("--c-lambda-ap-med", type=float, default=0.0)
    parser.add_argument("--c-lambda-ap-readout", type=float, default=0.0)
    parser.add_argument("--c-lambda-at-med", type=float, default=0.0)
    parser.add_argument("--c-lambda-at-readout", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--readout-train-size", type=int, default=1300)
    parser.add_argument("--readout-val-size", type=int, default=520)
    parser.add_argument("--at-token-position", default="last_token")
    parser.add_argument("--ap-token-position", default="last_token")
    parser.add_argument("--c-token-position", default="color")
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("ap_lambda_med =", args.ap_lambda_med)
    print("ap_lambda_readout =", args.ap_lambda_readout)
    print("c_lambda_ap_med =", args.c_lambda_ap_med)
    print("c_lambda_ap_readout =", args.c_lambda_ap_readout)
    print("c_lambda_at_med =", args.c_lambda_at_med)
    print("c_lambda_at_readout =", args.c_lambda_at_readout)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gemma_model()
    num_layers = model.config.num_hidden_layers

    das_banks = build_all_banks(model, tokenizer, args.data_path, args.ft_size, args.cal_size, args.te_size, args.eval_batch_size, args.seed)
    shared_readout = build_shared_readout_banks(model, tokenizer, args.data_path, args.readout_train_size, args.readout_val_size, args.eval_batch_size, args.seed)

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

    print("\n================ AT LAYER SWEEP ================")

    for layer in at_layers:
        layer_dir = result_dir / "at_layers" / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        at_path = layer_dir / "das_at.pt"
        at_readout_path = layer_dir / "at_readout_LogisticRegression.pt"

        print(f"\n=== TRAIN AT LAYER {layer} ===")
        at_result = train_at(model, tokenizer, [layer], args.at_dim, args.ft_size, args.cal_size, args.te_size, args.at_epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.at_token_position, args.seed, str(at_path), das_banks["AT"], args.output_space)
        fit_readout(model, at_path, at_readout_path, build_at_readout_datasets(shared_readout), "AT", args.at_token_position, args.eval_batch_size, args.seed, 26)

        score = float(at_result["cal_iia"])
        row = {"layer": layer, "score": score, "cal_iia": float(at_result["cal_iia"]), "test_iia": float(at_result["test_iia"]), "best_epoch": int(at_result["best_epoch"])}
        at_rows.append(row)

        if score >= best_at_score:
            best_at_score = score
            best_at_layer = layer
            best_at_path = at_path
            best_at_readout_path = at_readout_path

    save_summary(at_rows, result_dir / "at_layer_summary.csv")
    copy_file(best_at_path, result_dir / "das_at.pt")
    copy_file(best_at_readout_path, result_dir / "at_readout_LogisticRegression.pt")
    print(f"\n[AT SWEEP BEST] layer={best_at_layer} score={best_at_score:.4f}")

    requested_ap_layers = parse_layers(args.ap_layers)
    if requested_ap_layers is None:
        ap_layers = list(range(best_at_layer))
    else:
        ap_layers = []

        for layer in requested_ap_layers:
            if layer < best_at_layer:
                ap_layers.append(layer)

    if len(ap_layers) == 0:
        raise ValueError("No valid AP layers. AP layer must be before the selected AT layer.")

    ap_rows = []
    best_ap_score = -1.0
    best_ap_layer = None
    best_ap_path = None
    best_ap_readout_path = None

    print("\n================ AP LAYER SWEEP ================")

    for layer in ap_layers:
        layer_dir = result_dir / "ap_layers" / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        ap_path = layer_dir / "das_ap.pt"
        ap_readout_path = layer_dir / "ap_readout_LogisticRegression.pt"

        print(f"\n=== TRAIN AP LAYER {layer} ===")
        ap_result = train_ap(model, tokenizer, str(result_dir / "das_at.pt"), str(result_dir / "at_readout_LogisticRegression.pt"), [layer], args.ap_dim, args.ft_size, args.cal_size, args.te_size, args.ap_epochs, args.lr, args.ap_lambda_med, args.ap_lambda_readout, args.train_batch_size, args.eval_batch_size, args.ap_token_position, args.seed, str(ap_path), das_banks["AP"], args.output_space)
        fit_readout(model, ap_path, ap_readout_path, build_ap_readout_datasets(shared_readout), "AP", args.ap_token_position, args.eval_batch_size, args.seed, 4)

        direct_iia, mediator_iia, readout_acc = ap_result["cal"]
        test_direct_iia, test_mediator_iia, test_readout_acc = ap_result["test"]
        score = ap_score(ap_result["cal"], args.ap_lambda_med, args.ap_lambda_readout)

        row = {"layer": layer, "score": score, "cal_direct_iia": float(direct_iia), "cal_mediator_iia": float(mediator_iia), "cal_readout_acc": float(readout_acc), "test_direct_iia": float(test_direct_iia), "test_mediator_iia": float(test_mediator_iia), "test_readout_acc": float(test_readout_acc), "best_epoch": int(ap_result["best_epoch"])}
        ap_rows.append(row)

        if score >= best_ap_score:
            best_ap_score = score
            best_ap_layer = layer
            best_ap_path = ap_path
            best_ap_readout_path = ap_readout_path

    save_summary(ap_rows, result_dir / "ap_layer_summary.csv")
    copy_file(best_ap_path, result_dir / "das_ap.pt")
    copy_file(best_ap_readout_path, result_dir / "ap_readout_LogisticRegression.pt")
    print(f"\n[AP SWEEP BEST] layer={best_ap_layer} score={best_ap_score:.4f}")

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
        raise ValueError("No valid C layers. C layer must be before the selected AP and AT layers.")

    c_rows = []
    best_c_score = -1.0
    best_c_layer = None
    best_c_path = None
    best_c_readout_path = None

    print("\n================ C LAYER SWEEP ================")

    for layer in c_layers:
        layer_dir = result_dir / "c_layers" / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        c_path = layer_dir / "das_c.pt"
        c_readout_path = layer_dir / "c_readout_LogisticRegression.pt"

        print(f"\n=== TRAIN C LAYER {layer} ===")
        c_result = train_c(model, tokenizer, str(result_dir / "das_ap.pt"), str(result_dir / "ap_readout_LogisticRegression.pt"), str(result_dir / "das_at.pt"), str(result_dir / "at_readout_LogisticRegression.pt"), [layer], args.c_dim, args.ft_size, args.cal_size, args.te_size, args.c_epochs, args.lr, args.c_lambda_ap_med, args.c_lambda_ap_readout, args.c_lambda_at_med, args.c_lambda_at_readout, args.train_batch_size, args.eval_batch_size, args.c_token_position, args.seed, str(c_path), das_banks["C"], args.output_space)

        colors = shared_readout[0]["color_to_class"]
        extra = {"color_to_class": colors}
        fit_readout(model, c_path, c_readout_path, build_c_readout_datasets(shared_readout), "C", args.c_token_position, args.eval_batch_size, args.seed, len(colors), extra)

        cal_direct_iia, cal_ap_mediator_iia, cal_ap_readout_acc, cal_at_mediator_iia, cal_at_readout_acc = c_result["cal"]
        test_direct_iia, test_ap_mediator_iia, test_ap_readout_acc, test_at_mediator_iia, test_at_readout_acc = c_result["test"]
        score = c_score(c_result["cal"], args.c_lambda_ap_med, args.c_lambda_ap_readout, args.c_lambda_at_med, args.c_lambda_at_readout)

        row = {"layer": layer, "score": score, "cal_direct_iia": float(cal_direct_iia), "cal_ap_mediator_iia": float(cal_ap_mediator_iia), "cal_ap_readout_acc": float(cal_ap_readout_acc), "cal_at_mediator_iia": float(cal_at_mediator_iia), "cal_at_readout_acc": float(cal_at_readout_acc), "test_direct_iia": float(test_direct_iia), "test_ap_mediator_iia": float(test_ap_mediator_iia), "test_ap_readout_acc": float(test_ap_readout_acc), "test_at_mediator_iia": float(test_at_mediator_iia), "test_at_readout_acc": float(test_at_readout_acc), "best_epoch": int(c_result["best_epoch"])}
        c_rows.append(row)

        if score >= best_c_score:
            best_c_score = score
            best_c_layer = layer
            best_c_path = c_path
            best_c_readout_path = c_readout_path

    save_summary(c_rows, result_dir / "c_layer_summary.csv")
    copy_file(best_c_path, result_dir / "das_c.pt")
    copy_file(best_c_readout_path, result_dir / "c_readout_LogisticRegression.pt")
    print(f"\n[C SWEEP BEST] layer={best_c_layer} score={best_c_score:.4f}")

    best_text = []
    best_text.append(f"AT layer: {best_at_layer} | score: {best_at_score:.6f}")
    best_text.append(f"AP layer: {best_ap_layer} | score: {best_ap_score:.6f}")
    best_text.append(f"C layer: {best_c_layer} | score: {best_c_score:.6f}")
    (result_dir / "best_layers.txt").write_text("\n".join(best_text) + "\n")

    print("\n================ FINAL BEST LAYERS ================")
    print("\n".join(best_text))


if __name__ == "__main__":
    main()
