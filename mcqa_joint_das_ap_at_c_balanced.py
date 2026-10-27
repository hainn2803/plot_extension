"""Train AT -> AP -> C, then fit one factual readout for each variable."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mcqa_gradual_das import collect_joint_token_states_in_single_layer, set_seed, fit_readout
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
        layer = layer.strip()

        if layer != "":
            layers.append(int(layer))

    return layers


def parse_token_positions(value):
    token_positions = []

    for token_position in value.split(","):
        token_position = token_position.strip()

        if token_position != "":
            token_positions.append(token_position)

    return token_positions


def fit_readout(model, checkpoint, save_path, bank_pair, target, list_token_positions, batch_size, seed, num_classes, extra=None):
    """Fit a linear classifier on the frozen DAS coordinates."""
    set_seed(seed)

    device = next(model.parameters()).device

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    Q = checkpoint_data["basis"].float().to(device)
    layer = checkpoint_data["layer"]

    def features(bank):
        states = collect_joint_token_states_in_single_layer(model, bank["input_ids"], bank["attention_mask"], bank["position_by_id"], layer, list_token_positions, batch_size)
        states = states.reshape(states.shape[0], -1)

        if states.shape[1] != Q.shape[0]:
            raise ValueError(f"State dimension {states.shape[1]} does not match DAS basis dimension {Q.shape[0]}")

        x = (states.to(device) @ Q).detach()
        y = bank["target_ids"].to(device)

        return x, y

    train_bank, val_bank = bank_pair

    x_train, y_train = features(train_bank)
    x_val, y_val = features(val_bank)

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)

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

            with torch.no_grad():
                val_prediction = readout(x_val).argmax(dim=-1)
                acc = float((val_prediction == y_val).float().mean())

            if acc > best_acc:
                weights = readout.weight.detach() / std.unsqueeze(0)
                bias = readout.bias.detach() - weights @ mean

                best_acc = acc
                best = {
                    f"W_{target}": weights.cpu(),
                    f"b_{target}": bias.cpu(),
                }

                if extra is not None:
                    best.update(extra)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, save_path)

    print(f"[{target} readout] val_acc={best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_gradual_ap2.0_newrun")

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
    parser.add_argument("--ap-token-position", default="color_0,color_1,color_2,color_3")
    parser.add_argument("--c-token-position", default="color")

    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    at_layers = parse_layers(args.at_layers)
    ap_layers = parse_layers(args.ap_layers)
    c_layers = parse_layers(args.c_layers)

    at_token_positions = [args.at_token_position]
    ap_token_positions = parse_token_positions(args.ap_token_position)
    c_token_positions = [args.c_token_position]

    if len(ap_token_positions) == 0:
        raise ValueError("--ap-token-position must contain at least one token position")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    at_path = result_dir / "das_at.pt"
    ap_path = result_dir / "das_ap.pt"
    c_path = result_dir / "das_c.pt"

    at_readout_path = result_dir / "at_readout_LogisticRegression.pt"
    ap_readout_path = result_dir / "ap_readout_LogisticRegression.pt"
    c_readout_path = result_dir / "c_readout_LogisticRegression.pt"

    at_path_string = str(at_path)
    ap_path_string = str(ap_path)
    c_path_string = str(c_path)

    at_readout_path_string = str(at_readout_path)
    ap_readout_path_string = str(ap_readout_path)
    c_readout_path_string = str(c_readout_path)

    print(f"AT token positions = {at_token_positions}")
    print(f"AP token positions = {ap_token_positions}")
    print(f"C token positions  = {c_token_positions}")

    model, tokenizer = load_gemma_model()

    das_banks = build_all_banks(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    shared_readout = build_shared_readout_banks(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        train_size=args.readout_train_size,
        val_size=args.readout_val_size,
        batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    at_readout_banks = build_at_readout_datasets(shared_readout)
    ap_readout_banks = build_ap_readout_datasets(shared_readout)
    c_readout_banks = build_c_readout_datasets(shared_readout)

    print("\n=== TRAIN AT ===")

    train_at(
        model=model,
        tokenizer=tokenizer,
        layers=at_layers,
        subspace_dim=args.at_dim,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        epochs=args.at_epochs,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.at_token_position,
        seed=args.seed,
        save_path=at_path_string,
        banks=das_banks["AT"],
        output_space=args.output_space,
    )

    fit_readout(
        model=model,
        checkpoint=at_path,
        save_path=at_readout_path,
        bank_pair=at_readout_banks,
        target="AT",
        list_token_positions=at_token_positions,
        batch_size=args.eval_batch_size,
        seed=args.seed,
        num_classes=26,
    )

    print("\n=== TRAIN AP ===")

    train_ap(
        model=model,
        tokenizer=tokenizer,
        at_checkpoint=at_path_string,
        at_readout_checkpoint=at_readout_path_string,
        layers=ap_layers,
        subspace_dim=args.ap_dim,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        epochs=args.ap_epochs,
        lr=args.lr,
        lambda_med=args.ap_lambda_med,
        lambda_readout=args.ap_lambda_readout,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        list_ap_token_positions=ap_token_positions,
        at_token_position=args.at_token_position,
        seed=args.seed,
        save_path=ap_path_string,
        banks=das_banks["AP"],
        output_space=args.output_space,
    )

    fit_readout(
        model=model,
        checkpoint=ap_path,
        save_path=ap_readout_path,
        bank_pair=ap_readout_banks,
        target="AP",
        list_token_positions=ap_token_positions,
        batch_size=args.eval_batch_size,
        seed=args.seed,
        num_classes=4,
    )

    print("\n=== TRAIN C ===")

    train_c(
        model=model,
        tokenizer=tokenizer,
        ap_checkpoint=ap_path_string,
        ap_readout_checkpoint=ap_readout_path_string,
        at_checkpoint=at_path_string,
        at_readout_checkpoint=at_readout_path_string,
        layers=c_layers,
        subspace_dim=args.c_dim,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        epochs=args.c_epochs,
        lr=args.lr,
        lambda_ap_med=args.c_lambda_ap_med,
        lambda_ap_readout=args.c_lambda_ap_readout,
        lambda_at_med=args.c_lambda_at_med,
        lambda_at_readout=args.c_lambda_at_readout,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.c_token_position,
        seed=args.seed,
        save_path=c_path_string,
        banks=das_banks["C"],
        output_space=args.output_space,
    )

    colors = shared_readout[0]["color_to_class"]
    c_extra = {"color_to_class": colors}

    fit_readout(
        model=model,
        checkpoint=c_path,
        save_path=c_readout_path,
        bank_pair=c_readout_banks,
        target="C",
        list_token_positions=c_token_positions,
        batch_size=args.eval_batch_size,
        seed=args.seed,
        num_classes=len(colors),
        extra=c_extra,
    )


if __name__ == "__main__":
    main()