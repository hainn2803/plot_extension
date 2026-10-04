"""Train AT, fit its readout, then train AP using the existing DAS functions.

Put this file beside mcqa_gradual_das_at.py and mcqa_gradual_das_ap.py.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mcqa_neural_net import load_gemma_model
from mcqa_data_load_all_new_new import build_all_mcqa_banks
from mcqa_gradual_das import answer_label_ids, collect_layer_states, set_seed
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_ap import train_ap


def layers(value):
    return None if value == "all" else [int(x) for x in value.split(",")]


def fit_readout(model, tokenizer, at_path, output_path, args, at_banks):
    """Fit the 26-class AT readout on the frozen AT basis."""
    set_seed(args.seed)
    at = torch.load(at_path, map_location="cpu", weights_only=False)
    device = next(model.parameters()).device
    Q = at["basis"].to(device).float()

    fit, cal_banks, _ = at_banks
    num_classes = len(answer_label_ids(tokenizer))

    def data(bank):
        h = collect_layer_states(
            model, bank["source_input_ids"], bank["source_attention_mask"],
            bank["source_position_by_id"], at["layer"], args.token_position,
            args.eval_batch_size,
        )
        z = h.to(device).float() @ Q
        y = bank["counterfactual_label_ids"]["answer_token"].long()
        if y.numel() and (y.min().item() < 0 or y.max().item() >= num_classes):
            raise ValueError(
                f"AT labels must be in [0, {num_classes - 1}], "
                f"got [{y.min().item()}, {y.max().item()}]"
            )
        return z.detach(), y.to(device)

    x, y = data(fit)
    x_cal, y_cal = data(cal_banks["answer_token"])
    readout = torch.nn.Linear(Q.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(readout.parameters(), lr=1e-2)
    best_acc, best = -1.0, None
    for _ in range(100):
        optimizer.zero_grad()
        F.cross_entropy(readout(x), y).backward()
        optimizer.step()
        with torch.no_grad():
            acc = float((readout(x_cal).argmax(-1) == y_cal).float().mean())
            if acc > best_acc:
                best_acc = acc
                best = {
                    "W_AT": readout.weight.detach().cpu().clone(),
                    "b_AT": readout.bias.detach().cpu().clone(),
                }
    torch.save(best, output_path)
    print(f"[AT readout] cal_source_acc={best_acc:.4f}; saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at-save-path", default="results_joint_5000/das_at.pt")
    parser.add_argument("--ap-save-path", default="results_joint_5000/das_ap.pt")
    parser.add_argument("--at-layers", default="all")
    parser.add_argument("--ap-layers", default="all")
    parser.add_argument("--at-dim", type=int, default=1152)
    parser.add_argument("--ap-dim", type=int, default=128)
    parser.add_argument("--ft-size", type=int, default=5000)
    parser.add_argument("--cal-size", type=int, default=1000)
    parser.add_argument("--te-size", type=int, default=1000)
    parser.add_argument("--at-epochs", type=int, default=20)
    parser.add_argument("--ap-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-med", type=float, default=0.0)
    parser.add_argument("--lambda-readout", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--token-position", default="last_token")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    args = parser.parse_args()
    if args.lambda_med < 0 or args.lambda_readout < 0:
        parser.error("--lambda-med and --lambda-readout must be nonnegative")

    model, tokenizer = load_gemma_model()
    all_banks = build_all_mcqa_banks(
        model=model,
        tokenizer=tokenizer,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    readout_path = str(Path(args.at_save_path).with_name("at_readout_LogisticRegression.pt"))
    # if args.skip_at:
    #     at = torch.load(args.at_save_path, map_location="cpu", weights_only=False)
    #     for key in ("ft_size", "cal_size", "te_size", "seed"):
    #         if getattr(args, key) != at["config"][key]:
    #             parser.error(
    #                 f"--{key.replace('_', '-')} must match the saved AT checkpoint "
    #                 f"({at['config'][key]})"
    #             )
    #     if args.token_position != at["token_position"]:
    #         parser.error(
    #             "--token-position must match the saved AT checkpoint "
    #             f"({at['token_position']})"
    #         )
    #     saved_output_space = at["config"].get("output_space")
    #     if args.output_space != saved_output_space:
    #         parser.error(
    #             "--output-space must match the saved AT checkpoint "
    #             f"({saved_output_space or 'missing: retrain AT with the updated code'})"
    #         )
    #     print(f"[AT] reusing {args.at_save_path}")
    # else:
    train_at(
        model, tokenizer, layers=layers(args.at_layers), subspace_dim=args.at_dim,
        ft_size=args.ft_size, cal_size=args.cal_size, te_size=args.te_size,
        epochs=args.at_epochs, lr=args.lr,
        train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
        token_position=args.token_position, seed=args.seed,
        save_path=args.at_save_path, banks=all_banks["AT"],
        output_space=args.output_space,
    )

    # The AT readout is always a 26-class classifier, regardless of output_space.
    fit_readout(
        model, tokenizer, args.at_save_path, readout_path, args,
        at_banks=all_banks["AT"],
    )
    train_ap(
        model, tokenizer, at_checkpoint=args.at_save_path,
        at_readout_checkpoint=readout_path,
        layers=layers(args.ap_layers), subspace_dim=args.ap_dim,
        ft_size=args.ft_size, cal_size=args.cal_size, te_size=args.te_size,
        epochs=args.ap_epochs, lr=args.lr,
        lambda_med=args.lambda_med, lambda_readout=args.lambda_readout,
        train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
        token_position=args.token_position, seed=args.seed,
        save_path=args.ap_save_path, banks=all_banks["AP"],
        output_space=args.output_space,
    )


if __name__ == "__main__":
    main()
