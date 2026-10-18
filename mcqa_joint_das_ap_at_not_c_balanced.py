"""Train AT -> AP -> C DAS on the randomized AP/AT/C dataset.

AT/AP use balanced factual readouts from the existing JChang readout builder.
C gets its own balanced factual readout over a fixed color vocabulary inferred
once from the entire saved randomized dataset.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_not_c import build_all_banks, color_position, load_generated_dataset
from mcqa_data_load_jchang_disjoint import build_readout_datasets
from mcqa_gradual_das import collect_layer_states, set_seed
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_ap import train_ap
from mcqa_gradual_das_c import train_c


def layers(value):
    return None if value == "all" else [int(x) for x in value.split(",")]


def fit_readout(model, checkpoint_path, output_path, args, target, readout_banks,
                token_position=None, num_classes=None, extra_state=None):
    """Fit a linear factual readout on the learned DAS coordinates."""
    set_seed(args.seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = next(model.parameters()).device
    Q = checkpoint["basis"].to(device).float()
    fit, cal = readout_banks

    if num_classes is None:
        num_classes = 26 if target == "AT" else 4
    if token_position is None:
        token_position = args.token_position

    def data(bank):
        h = collect_layer_states(
            model, bank["input_ids"], bank["attention_mask"], bank["position_by_id"],
            checkpoint["layer"], token_position, args.eval_batch_size,
        )
        z = h.to(device).float() @ Q
        y = bank["target_ids"].long().to(device)
        if y.numel() and (y.min().item() < 0 or y.max().item() >= num_classes):
            raise ValueError(f"{target} labels must be in [0, {num_classes - 1}]")
        return z.detach(), y

    x, y = data(fit)
    x_cal, y_cal = data(cal)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_norm = (x - mean) / std
    x_cal_norm = (x_cal - mean) / std

    best_acc = -1.0
    best_val_loss = float("inf")
    best = None

    for penalty in (1e-4, 1e-3, 1e-2):
        set_seed(args.seed)
        readout = torch.nn.Linear(Q.shape[1], num_classes).to(device)
        optimizer = torch.optim.Adam(readout.parameters(), lr=1e-2)

        for epoch in range(1, 201):
            optimizer.zero_grad()
            logits = readout(x_norm)
            loss = F.cross_entropy(logits, y) + penalty * readout.weight.square().sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_logits = readout(x_cal_norm)
                val_acc = float((val_logits.argmax(-1) == y_cal).float().mean())
                val_loss = float(F.cross_entropy(val_logits, y_cal))

                if val_acc > best_acc or (val_acc == best_acc and val_loss < best_val_loss):
                    best_acc = val_acc
                    best_val_loss = val_loss
                    train_acc = float((readout(x_norm).argmax(-1) == y).float().mean())
                    best_penalty = penalty
                    best_epoch = epoch
                    W = readout.weight.detach() / std.unsqueeze(0)
                    b = readout.bias.detach() - W @ mean
                    best = {f"W_{target}": W.cpu().clone(), f"b_{target}": b.cpu().clone()}

                    if extra_state is not None:
                        best.update(extra_state)

        print(f"[{target} readout] penalty={penalty} done")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, output_path)
    print(
        f"[{target} readout] penalty={best_penalty} epoch={best_epoch} "
        f"train_acc={train_acc:.4f} balanced_val_acc={best_acc:.4f}; saved to {output_path}"
    )


def get_color_mapping(data_path):
    """Define one fixed color vocabulary for the whole saved dataset."""
    ds = load_generated_dataset(data_path)
    colors = set()

    for split in ("train", "validation", "test"):
        for row in ds[split]:
            colors.add(str(row["color"]).strip().casefold())
            colors.add(str(row["color_counterfactual"]["color"]).strip().casefold())
            colors.add(str(row["color_all_counterfactual"]["color"]).strip().casefold())

    colors = sorted(colors)
    color_to_class = {color: i for i, color in enumerate(colors)}
    print(f"[C classes] {len(colors)} colors: {colors}")
    return color_to_class




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors")

    parser.add_argument("--at-save-path", default="results_joint_randomized/das_at.pt")
    parser.add_argument("--ap-save-path", default="results_joint_randomized/das_ap.pt")
    parser.add_argument("--c-save-path", default="results_joint_randomized/das_c.pt")

    parser.add_argument("--at-layers", default="all")
    parser.add_argument("--ap-layers", default="all")
    parser.add_argument("--c-layers", default="all")

    parser.add_argument("--at-dim", type=int, default=128)
    parser.add_argument("--ap-dim", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=128)

    parser.add_argument("--ft-size", type=int, default=400)
    parser.add_argument("--cal-size", type=int, default=200)
    parser.add_argument("--te-size", type=int, default=200)

    parser.add_argument("--at-epochs", type=int, default=20)
    parser.add_argument("--ap-epochs", type=int, default=20)
    parser.add_argument("--c-epochs", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-med", type=float, default=0.0)
    parser.add_argument("--lambda-readout", type=float, default=0.0)

    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)

    parser.add_argument("--token-position", default="last_token")
    parser.add_argument("--c-token-position", default="color")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="az")

    parser.add_argument("--at-readout-per-class", type=int, default=50)
    parser.add_argument("--ap-readout-per-class", type=int, default=50)
    parser.add_argument("--c-readout-per-class", type=int, default=50)
    parser.add_argument("--c-readout-val-per-class", type=int, default=20)
    parser.add_argument("--readout-max-attempts", type=int, default=500)
    args = parser.parse_args()

    if args.lambda_med < 0 or args.lambda_readout < 0:
        parser.error("--lambda-med and --lambda-readout must be nonnegative")

    model, tokenizer = load_gemma_model()

    all_banks = build_all_banks(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("TRAIN AT")
    print("=" * 80)

    train_at(
        model, tokenizer,
        layers=layers(args.at_layers),
        subspace_dim=args.at_dim,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        epochs=args.at_epochs,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.token_position,
        seed=args.seed,
        save_path=args.at_save_path,
        banks=all_banks["AT"],
        output_space=args.output_space,
    )

    at_readout_path = str(Path(args.at_save_path).with_name("at_readout_LogisticRegression.pt"))
    at_readout_banks = build_readout_datasets(
        model, tokenizer,
        target="AT",
        examples_per_class=args.at_readout_per_class,
        batch_size=args.eval_batch_size,
        max_attempts_per_class=args.readout_max_attempts,
        seed=args.seed,
    )
    fit_readout(
        model, args.at_save_path, at_readout_path, args,
        target="AT", readout_banks=at_readout_banks,
    )

    print("\n" + "=" * 80)
    print("TRAIN AP")
    print("=" * 80)

    train_ap(
        model, tokenizer,
        at_checkpoint=args.at_save_path,
        at_readout_checkpoint=at_readout_path,
        layers=layers(args.ap_layers),
        subspace_dim=args.ap_dim,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        epochs=args.ap_epochs,
        lr=args.lr,
        lambda_med=args.lambda_med,
        lambda_readout=args.lambda_readout,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.token_position,
        seed=args.seed,
        save_path=args.ap_save_path,
        banks=all_banks["AP"],
        output_space=args.output_space,
    )

    ap_readout_path = str(Path(args.ap_save_path).with_name("ap_readout_LogisticRegression.pt"))
    ap_readout_banks = build_readout_datasets(
        model, tokenizer,
        target="AP",
        examples_per_class=args.ap_readout_per_class,
        batch_size=args.eval_batch_size,
        max_attempts_per_class=args.readout_max_attempts,
        seed=args.seed,
    )
    fit_readout(
        model, args.ap_save_path, ap_readout_path, args,
        target="AP", readout_banks=ap_readout_banks,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("AT checkpoint:", args.at_save_path)
    print("AT readout   :", at_readout_path)
    print("AP checkpoint:", args.ap_save_path)
    print("AP readout   :", ap_readout_path)


if __name__ == "__main__":
    main()
