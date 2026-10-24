"""Train AT -> AP -> C, then fit one factual readout for each variable."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mcqa_gradual_das import collect_layer_states, set_seed
from mcqa_gradual_das_ap import train_ap
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_c import train_c
from mcqa_load_randomized_ap_at_c import build_all_banks, build_ap_readout_datasets, build_at_readout_datasets, build_c_readout_datasets, build_shared_readout_banks
from mcqa_neural_net import load_gemma_model


def parse_layers(value):
    if value == "all":
        return None

    layers = []
    for layer in value.split(","):
        layers.append(int(layer))
    return layers


def fit_readout(model, checkpoint, save_path, bank_pair, target, token_position, batch_size, seed, num_classes, extra=None):
    """Fit a linear classifier on the frozen DAS coordinates."""
    set_seed(seed)
    device = next(model.parameters()).device
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    Q = checkpoint_data["basis"].float().to(device)

    def features(bank):
        states = collect_layer_states(model, bank["input_ids"], bank["attention_mask"], bank["position_by_id"], checkpoint_data["layer"], token_position, batch_size)
        x = (states.to(device) @ Q).detach()
        y = bank["target_ids"].to(device)
        return x, y

    train_bank, val_bank = bank_pair
    x_train, y_train = features(train_bank)
    x_val, y_val = features(val_bank)

    mean = x_train.mean(0)
    std = x_train.std(0, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    best_acc = -1.0
    best = None

    for penalty in (1e-4, 1e-3, 1e-2):
        readout = torch.nn.Linear(Q.shape[1], num_classes).to(device)
        optimizer = torch.optim.Adam(readout.parameters(), lr=1e-2)

        for _ in range(200):
            optimizer.zero_grad()
            logits = readout(x_train)
            loss = F.cross_entropy(logits, y_train)
            loss = loss + penalty * readout.weight.square().sum()
            loss.backward()
            optimizer.step()

            val_prediction = readout(x_val).argmax(-1)
            acc = float((val_prediction == y_val).float().mean())

            if acc > best_acc:
                weights = readout.weight.detach() / std.unsqueeze(0)
                bias = readout.bias.detach() - weights @ mean
                best_acc = acc
                best = {f"W_{target}": weights.cpu(), f"b_{target}": bias.cpu()}

                if extra is not None:
                    best.update(extra)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, save_path)
    print(f"[{target} readout] val_acc={best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_gradual_all_ap1.0_at6.0_newrun")
    parser.add_argument("--ft-size", type=int, default=800)
    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--te-size", type=int, default=400)
    parser.add_argument("--at-dim", type=int, default=128)
    parser.add_argument("--ap-dim", type=int, default=256)
    parser.add_argument("--c-dim", type=int, default=128)
    parser.add_argument("--at-layers", default="all")
    parser.add_argument("--ap-layers", default="all")
    parser.add_argument("--c-layers", default="all")
    parser.add_argument("--at-epochs", type=int, default=20)
    parser.add_argument("--ap-epochs", type=int, default=20)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ap-lambda-med", type=float, default=2.0)
    parser.add_argument("--ap-lambda-readout", type=float, default=2.0)
    parser.add_argument("--c-lambda-ap-med", type=float, default=1.0)
    parser.add_argument("--c-lambda-ap-readout", type=float, default=1.0)
    parser.add_argument("--c-lambda-at-med", type=float, default=6.0)
    parser.add_argument("--c-lambda-at-readout", type=float, default=6.0)
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

    set_seed(args.seed)

    result = Path(args.result_dir)
    result.mkdir(parents=True, exist_ok=True)

    at_path = result / "das_at.pt"
    ap_path = result / "das_ap.pt"
    c_path = result / "das_c.pt"

    at_readout_path = result / "at_readout_LogisticRegression.pt"
    ap_readout_path = result / "ap_readout_LogisticRegression.pt"
    c_readout_path = result / "c_readout_LogisticRegression.pt"

    model, tokenizer = load_gemma_model()
    das_banks = build_all_banks(model, tokenizer, args.data_path, args.ft_size, args.cal_size, args.te_size, args.eval_batch_size, args.seed)
    shared_readout = build_shared_readout_banks(model, tokenizer, args.data_path, args.readout_train_size, args.readout_val_size, args.eval_batch_size, args.seed)

    print("\n=== TRAIN AT ===")
    train_at(model, tokenizer, parse_layers(args.at_layers), args.at_dim, args.ft_size, args.cal_size, args.te_size, args.at_epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.at_token_position, args.seed, str(at_path), das_banks["AT"], args.output_space)
    fit_readout(model, at_path, at_readout_path, build_at_readout_datasets(shared_readout), "AT", args.at_token_position, args.eval_batch_size, args.seed, 26)

    print("\n=== TRAIN AP ===")
    train_ap(model, tokenizer, str(at_path), str(at_readout_path), parse_layers(args.ap_layers), args.ap_dim, args.ft_size, args.cal_size, args.te_size, args.ap_epochs, args.lr, args.ap_lambda_med, args.ap_lambda_readout, args.train_batch_size, args.eval_batch_size, args.ap_token_position, args.seed, str(ap_path), das_banks["AP"], args.output_space)
    fit_readout(model, ap_path, ap_readout_path, build_ap_readout_datasets(shared_readout), "AP", args.ap_token_position, args.eval_batch_size, args.seed, 4)

    print("\n=== TRAIN C ===")
    train_c(model, tokenizer, str(ap_path), str(ap_readout_path), str(at_path), str(at_readout_path), parse_layers(args.c_layers), args.c_dim, args.ft_size, args.cal_size, args.te_size, args.c_epochs, args.lr, args.c_lambda_ap_med, args.c_lambda_ap_readout, args.c_lambda_at_med, args.c_lambda_at_readout, args.train_batch_size, args.eval_batch_size, args.c_token_position, args.seed, str(c_path), das_banks["C"], args.output_space)

    colors = shared_readout[0]["color_to_class"]
    extra = {"color_to_class": colors}
    fit_readout(model, c_path, c_readout_path, build_c_readout_datasets(shared_readout), "C", args.c_token_position, args.eval_batch_size, args.seed, len(colors), extra)


if __name__ == "__main__":
    main()
