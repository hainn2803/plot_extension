import argparse
import csv
import os

import torch

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_not_c import build_all_banks
from mcqa_gradual_das_ap import train_ap


def add_offsets(bank, n_tokens):
    """offset_0 = last real token, offset_1 = one token before, ..."""
    base_len = bank["base_attention_mask"].sum(1).long()
    source_len = bank["source_attention_mask"].sum(1).long()

    for offset in range(n_tokens):
        key = f"offset_{offset}"
        bank["base_position_by_id"][key] = base_len - 1 - offset
        bank["source_position_by_id"][key] = source_len - 1 - offset


def example_token(bank, tokenizer, offset):
    """Token at this offset for the first base example, only for printing."""
    mask = bank["base_attention_mask"][0]
    length = int(mask.sum())
    pad = int((mask == 0).sum())
    pos = pad + length - 1 - offset
    token_id = int(bank["base_input_ids"][0, pos])
    return tokenizer.decode([token_id]).replace("\n", "\\n")


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "offset_from_end", "token_example",
                        "cal_iia", "test_iia", "checkpoint"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors")
    parser.add_argument("--layers", default="17,18,19")
    parser.add_argument("--ft-size", type=int, default=400)
    parser.add_argument("--cal-size", type=int, default=200)
    parser.add_argument("--te-size", type=int, default=200)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    # Reuse the AT model/readout already trained by train_joint_multilayer_simple.py.
    parser.add_argument("--at-checkpoint",
                        default="results_joint_randomized/das_at.pt")
    parser.add_argument("--at-readout-checkpoint",
                        default="results_joint_randomized/at_readout_LogisticRegression.pt")

    parser.add_argument("--out-dir", default="results_ap_token_intervention")
    parser.add_argument("--max-offset", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, tokenizer = load_gemma_model()
    model.eval()

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

    train, cal_dict, test_dict = all_banks["AP"]
    cal = cal_dict["answer_pointer"]
    test = test_dict["answer_pointer"]

    # Only sweep offsets that exist in EVERY base/source example.
    n_tokens = min(
        int(train["base_attention_mask"].sum(1).min()),
        int(train["source_attention_mask"].sum(1).min()),
        int(cal["base_attention_mask"].sum(1).min()),
        int(cal["source_attention_mask"].sum(1).min()),
        int(test["base_attention_mask"].sum(1).min()),
        int(test["source_attention_mask"].sum(1).min()),
    )
    if args.max_offset is not None:
        n_tokens = min(n_tokens, args.max_offset + 1)

    for bank in (train, cal, test):
        add_offsets(bank, n_tokens)

    layers = [int(x) for x in args.layers.split(",")]
    csv_path = os.path.join(args.out_dir, "ap_token_intervention.csv")
    rows = []

    print(f"[sweep] layers={layers}")
    print(f"[sweep] offsets=0..{n_tokens - 1} (0 = last token)")

    for layer in layers:
        for offset in range(n_tokens):
            key = f"offset_{offset}"
            ckpt_path = os.path.join(
                args.out_dir, f"das_ap_L{layer}_offset{offset}.pt"
            )

            print("\n" + "=" * 80)
            print(
                f"[RUN] layer={layer} offset={offset} "
                f"token={example_token(test, tokenizer, offset)!r}"
            )
            print("=" * 80)

            # This is DAS intervention training, not a probe.
            # lambda_med=lambda_readout=0 => optimize/select only direct AP IIA.
            train_ap(
                model,
                tokenizer,
                at_checkpoint=args.at_checkpoint,
                at_readout_checkpoint=args.at_readout_checkpoint,
                layers=[layer],
                subspace_dim=args.dim,
                ft_size=args.ft_size,
                cal_size=args.cal_size,
                te_size=args.te_size,
                epochs=args.epochs,
                lr=args.lr,
                lambda_med=0.0,
                lambda_readout=0.0,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                token_position=key,
                seed=args.seed,
                save_path=ckpt_path,
                banks=all_banks["AP"],
                output_space="full",
            )

            result = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )

            row = {
                "layer": layer,
                "offset_from_end": offset,
                "token_example": example_token(test, tokenizer, offset),
                "cal_iia": float(result["cal"]["direct_iia"]),
                "test_iia": float(result["test"]["direct_iia"]),
                "checkpoint": ckpt_path,
            }
            rows.append(row)
            save_csv(rows, csv_path)

            print(
                f"[RESULT] L{layer} offset={offset} "
                f"cal={row['cal_iia']:.4f} test={row['test_iia']:.4f}"
            )

    best = max(rows, key=lambda x: x["cal_iia"])
    print("\n" + "=" * 80)
    print("[BEST BY CAL]")
    print(best)
    print(f"saved results to: {csv_path}")


if __name__ == "__main__":
    main()
