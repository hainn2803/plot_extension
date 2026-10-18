import argparse
import os

import torch
import torch.nn.functional as F

from mcqa_neural_net import load_gemma_model
from color_var_load_color_pair import build_all_color_banks
from mcqa_gradual_das import (
    set_seed,
    answer_label_ids,
    output_label_ids,
    output_targets,
    output_iia,
    collect_layer_states,
    collect_all_layer_states,
    LearnedSubspace,
    run_full_layer_intervention,
    run_das_intervention,
)


@torch.no_grad()
def evaluate_c_iia(
    model,
    bank,
    source_states,
    Q,
    layer,
    label_ids,
    token_position,
    batch_size,
    tokenizer,
    output_space="full",
):
    logits = run_das_intervention(
        model,
        bank,
        source_states,
        Q,
        layer,
        output_label_ids(label_ids, output_space),
        token_position,
        batch_size,
        require_grad=False,
    )

    target = bank["counterfactual_label_ids"]["color"]

    return output_iia(
        logits,
        target,
        tokenizer,
        output_space,
    )


def train_c(
    model,
    tokenizer,
    layers=None,
    subspace_dim=128,
    ft_size=400,
    cal_size=200,
    te_size=200,
    epochs=50,
    lr=1e-2,
    train_batch_size=32,
    eval_batch_size=32,
    token_position="color",
    seed=0,
    save_path="results/das_c.pt",
    banks=None,
    output_space="full",
):
    set_seed(seed)
    device = next(model.parameters()).device

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    if banks is None:
        raise ValueError("train_c requires COLOR banks via banks=...")

    fit_bank, cal_banks, te_banks = banks
    cal_bank = cal_banks["color"]
    te_bank = te_banks["color"]

    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    # ------------------------------------------------------------
    # Full-layer interchange scan for layer selection.
    # ------------------------------------------------------------
    cal_source_by_layer = collect_all_layer_states(
        model,
        cal_bank["source_input_ids"],
        cal_bank["source_attention_mask"],
        cal_bank["source_position_by_id"],
        layers,
        token_position,
        eval_batch_size,
    )

    layer_results = []

    for layer in layers:
        logits = run_full_layer_intervention(
            model,
            cal_bank,
            cal_source_by_layer[layer],
            layer,
            output_ids,
            token_position,
            eval_batch_size,
        )

        target = cal_bank["counterfactual_label_ids"]["color"]

        iia = output_iia(
            logits,
            target,
            tokenizer,
            output_space,
        )

        layer_results.append(
            {
                "layer": int(layer),
                "cal_iia": iia,
            }
        )

        print(
            f"[C layer] layer={layer} "
            f"iia={iia:.4f}"
        )

    best_layer_row = max(
        layer_results,
        key=lambda row: row["cal_iia"],
    )

    layer = int(best_layer_row["layer"])

    print(
        f"[C layer selected] layer={layer} "
        f"iia={best_layer_row['cal_iia']:.4f}"
    )

    # ------------------------------------------------------------
    # Cache source states at selected layer.
    # ------------------------------------------------------------
    ft_source = collect_layer_states(
        model,
        fit_bank["source_input_ids"],
        fit_bank["source_attention_mask"],
        fit_bank["source_position_by_id"],
        layer,
        token_position,
        eval_batch_size,
    )

    cal_source = cal_source_by_layer[layer]

    te_source = collect_layer_states(
        model,
        te_bank["source_input_ids"],
        te_bank["source_attention_mask"],
        te_bank["source_position_by_id"],
        layer,
        token_position,
        eval_batch_size,
    )

    # ------------------------------------------------------------
    # Learn C subspace.
    # ------------------------------------------------------------
    alignment = LearnedSubspace(
        model.config.hidden_size,
        subspace_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        alignment.parameters(),
        lr=lr,
    )

    target_ft = fit_bank["counterfactual_label_ids"]["color"]

    best_cal = -1.0
    best_basis = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(
            len(fit_bank["base_input_ids"])
        )

        total_loss = 0.0
        total_count = 0

        for start in range(
            0,
            len(perm),
            train_batch_size,
        ):
            idx = perm[
                start:start + train_batch_size
            ]

            mini_bank = {
                "base_input_ids":
                    fit_bank["base_input_ids"][idx],

                "base_attention_mask":
                    fit_bank["base_attention_mask"][idx],

                "base_position_by_id": {
                    k: v[idx]
                    for k, v
                    in fit_bank["base_position_by_id"].items()
                },
            }

            optimizer.zero_grad()

            Q = alignment.basis()

            logits = run_das_intervention(
                model,
                mini_bank,
                ft_source[idx],
                Q,
                layer,
                output_ids,
                token_position,
                len(idx),
                require_grad=True,
            )

            target = output_targets(
                target_ft[idx],
                labels,
                output_space,
                device,
            )

            loss = F.cross_entropy(
                logits,
                target,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * len(idx)
            )
            total_count += len(idx)

        with torch.no_grad():
            Q_eval = alignment.basis()

            cal_iia = evaluate_c_iia(
                model,
                cal_bank,
                cal_source,
                Q_eval,
                layer,
                labels,
                token_position,
                eval_batch_size,
                tokenizer,
                output_space,
            )

        print(
            f"[epoch {epoch:02d}] "
            f"loss={total_loss / total_count:.4f} "
            f"cal_iia={cal_iia:.4f}"
        )

        if cal_iia > best_cal:
            best_cal = cal_iia
            best_basis = (
                Q_eval
                .detach()
                .cpu()
                .clone()
            )
            best_epoch = epoch

    # ------------------------------------------------------------
    # Test best calibration checkpoint.
    # ------------------------------------------------------------
    test_iia = evaluate_c_iia(
        model,
        te_bank,
        te_source,
        best_basis.to(device),
        layer,
        labels,
        token_position,
        eval_batch_size,
        tokenizer,
        output_space,
    )

    result = {
        "variable": "color",
        "method": "DAS learned orthonormal subspace",
        "layer": layer,
        "subspace_dim": subspace_dim,
        "basis": best_basis,
        "token_position": token_position,
        "output_space": output_space,

        "layer_selection": {
            "method": "full_layer_interchange_cal_iia",
            "selected_full_layer_cal_iia":
                float(best_layer_row["cal_iia"]),
            "all_layers": layer_results,
        },

        "cal": {
            "best_epoch": best_epoch,
            "direct_iia": best_cal,
        },

        "test": {
            "direct_iia": test_iia,
        },

        "config": {
            "ft_size": ft_size,
            "cal_size": cal_size,
            "te_size": te_size,
            "epochs": epochs,
            "lr": lr,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "seed": seed,
            "output_space": output_space,
        },
    }

    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True,
    )

    torch.save(
        result,
        save_path,
    )

    print(
        f"[C BEST] layer={layer} "
        f"k={subspace_dim} "
        f"epoch={best_epoch} "
        f"cal_iia={best_cal:.4f} "
        f"test_iia={test_iia:.4f}"
    )

    print(f"saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layers",
        default="all",
    )

    parser.add_argument(
        "--subspace-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--ft-size",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--cal-size",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--te-size",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--token-position",
        default="color",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-path",
        default="results/das_c.pt",
    )

    parser.add_argument(
        "--output-space",
        choices=("full", "az"),
        default="full",
    )

    args = parser.parse_args()

    model, tokenizer = load_gemma_model()

    layers = (
        None
        if args.layers == "all"
        else [
            int(x)
            for x in args.layers.split(",")
            if x.strip()
        ]
    )

    banks = build_all_color_banks(
        model=model,
        tokenizer=tokenizer,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    train_c(
        model,
        tokenizer,
        layers=layers,
        subspace_dim=args.subspace_dim,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        epochs=args.epochs,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.token_position,
        seed=args.seed,
        save_path=args.save_path,
        banks=banks,
        output_space=args.output_space,
    )


if __name__ == "__main__":
    main()
