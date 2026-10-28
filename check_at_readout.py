"""Train AT DAS, then fit a balanced A-Z factual readout on its frozen basis.

Place next to mcqa_balanced_readout_data.py, mcqa_gradual_das_at.py,
mcqa_gradual_das.py, mcqa_neural_net.py, and mcqa_data_load_all_new_new_new.py.
The saved readout has W_AT and b_AT for the existing AP training code.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mcqa_balanced_readout_data import build_readout_datasets
from mcqa_data_load_all_new_new import build_all_mcqa_banks
from mcqa_gradual_das import collect_layer_states, set_seed
from mcqa_gradual_das_at import train_at
from mcqa_neural_net import load_gemma_model


def parse_layers(value):
    return None if value == "all" else [int(item) for item in value.split(",")]


def check_at_checkpoint(at, args):
    config = at["config"]
    for key in ("ft_size", "cal_size", "te_size", "seed"):
        if config[key] != getattr(args, key):
            raise ValueError(f"--{key.replace('_', '-')} differs from saved AT checkpoint")
    if at["token_position"] != args.token_position:
        raise ValueError("--token-position differs from saved AT checkpoint")
    if config.get("output_space") != args.output_space:
        raise ValueError("--output-space differs from saved AT checkpoint")
    if at["basis"].shape[1] != args.at_dim:
        raise ValueError("--at-dim differs from saved AT checkpoint")


@torch.no_grad()
def readout_features(model, bank, at, batch_size):
    device = next(model.parameters()).device
    Q = at["basis"].to(device=device, dtype=torch.float32)
    states = collect_layer_states(
        model,
        bank["input_ids"], bank["attention_mask"], bank["position_by_id"],
        at["layer"], at["token_position"], batch_size,
    )
    features = states.to(device=device, dtype=torch.float32) @ Q
    targets = bank["target_ids"].to(device=device, dtype=torch.long)
    if targets.numel() == 0 or targets.min().item() < 0 or targets.max().item() >= 26:
        raise ValueError("AT readout labels must be in A-Z (indices 0-25)")
    return features.detach(), targets


def accuracy(logits, targets):
    return float((logits.argmax(dim=-1) == targets).float().mean().item())


def report(name, logits, targets):
    predicted = logits.argmax(dim=-1)
    print(f"[{name}] accuracy={accuracy(logits, targets):.4f} n={len(targets)}")
    for group, mask in (("A-D", targets < 4), ("E-Z", targets >= 4)):
        if mask.any():
            correct = (predicted[mask] == targets[mask]).float().mean().item()
            print(f"[{name}] {group}={correct:.4f} n={int(mask.sum().item())}")


def fit_at_readout(model, at, train_bank, val_bank, args):
    device = next(model.parameters()).device
    train_x, train_y = readout_features(model, train_bank, at, args.eval_batch_size)
    val_x, val_y = readout_features(model, val_bank, at, args.eval_batch_size)

    # Estimate feature scaling only from training prompts. Fold it into the
    # saved W_AT/b_AT so downstream code still computes (hidden @ Q) @ W_AT.T + b_AT.
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-4)
    x_train = (train_x - mean) / scale
    x_val = (val_x - mean) / scale
    best = None

    for penalty in (1e-4, 1e-3, 1e-2):
        set_seed(args.seed)
        classifier = torch.nn.Linear(at["basis"].shape[1], 26).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.readout_lr)

        for epoch in range(1, args.readout_epochs + 1):
            optimizer.zero_grad()
            logits = classifier(x_train)
            loss = F.cross_entropy(logits, train_y) + penalty * classifier.weight.square().sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_logits = classifier(x_val)
                score = (accuracy(val_logits, val_y), -F.cross_entropy(val_logits, val_y).item())
                if best is None or score > best["score"]:
                    W = classifier.weight.detach() / scale.unsqueeze(0)
                    b = classifier.bias.detach() - W @ mean
                    best = {
                        "score": score,
                        "W_AT": W.cpu().clone(),
                        "b_AT": b.cpu().clone(),
                        "penalty": penalty,
                        "epoch": epoch,
                    }

    W = best["W_AT"].to(device)
    b = best["b_AT"].to(device)
    report("AT readout train", F.linear(train_x, W, b), train_y)
    report("AT readout validation", F.linear(val_x, W, b), val_y)
    print(f"[AT readout] selected penalty={best['penalty']:g} epoch={best['epoch']}")

    output_path = args.readout_save_path or str(
        Path(args.at_save_path).with_name("at_readout_LogisticRegression.pt")
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W_AT": best["W_AT"], "b_AT": best["b_AT"]}, output_path)
    print(f"[AT readout] saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at-save-path", default="results_at_readout/das_at.pt")
    parser.add_argument("--readout-save-path", default=None)
    parser.add_argument("--skip-at", action="store_true", help="Fit only readout from saved AT basis")
    parser.add_argument("--at-layers", default="all")
    parser.add_argument("--at-dim", type=int, default=1152)
    parser.add_argument("--ft-size", type=int, default=5000)
    parser.add_argument("--cal-size", type=int, default=1000)
    parser.add_argument("--te-size", type=int, default=1000)
    parser.add_argument("--at-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--token-position", default="last_token")
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--readout-train-per-class", type=int, default=20)
    parser.add_argument("--readout-val-per-class", type=int, default=5)
    parser.add_argument("--readout-max-attempts", type=int, default=100)
    parser.add_argument("--readout-epochs", type=int, default=200)
    parser.add_argument("--readout-lr", type=float, default=1e-2)
    args = parser.parse_args()
    if args.readout_epochs < 1 or args.readout_lr <= 0:
        parser.error("--readout-epochs and --readout-lr must be positive")

    set_seed(args.seed)
    model, tokenizer = load_gemma_model()
    if args.skip_at:
        at = torch.load(args.at_save_path, map_location="cpu", weights_only=False)
        check_at_checkpoint(at, args)
        print(f"[AT] reusing {args.at_save_path}")
    else:
        at_banks = build_all_mcqa_banks(
            model=model, tokenizer=tokenizer,
            ft_size=args.ft_size, cal_size=args.cal_size, te_size=args.te_size,
            batch_size=args.eval_batch_size, seed=args.seed,
        )["AT"]
        train_at(
            model, tokenizer, layers=parse_layers(args.at_layers),
            subspace_dim=args.at_dim, ft_size=args.ft_size,
            cal_size=args.cal_size, te_size=args.te_size,
            epochs=args.at_epochs, lr=args.lr,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            token_position=args.token_position, seed=args.seed,
            save_path=args.at_save_path, banks=at_banks,
            output_space=args.output_space,
        )
        at = torch.load(args.at_save_path, map_location="cpu", weights_only=False)

    train_bank, val_bank = build_readout_datasets(
        model, tokenizer, target="AT",
        examples_per_class=args.readout_train_per_class,
        val_examples_per_class=args.readout_val_per_class,
        batch_size=args.eval_batch_size,
        max_attempts_per_class=args.readout_max_attempts,
        seed=args.seed,
    )
    fit_at_readout(model, at, train_bank, val_bank, args)


if __name__ == "__main__":
    main()
