"""
Evaluate downstream AP, AT, and Y after a COLOR-only DAS intervention.

The COLOR intervention is applied to the BASE prompt:
    h_base[c_site] <- h_base[c_site] + Proj_Qc(h_source[c_site] - h_base[c_site])

Then, in the SAME forward pass, we:
  1) capture the downstream AP state and apply the saved AP readout,
  2) capture the downstream AT state and apply the saved AT readout,
  3) evaluate the final model output Y.

Targets are the SOURCE values from the COLOR pair:
    AP target = source_AP
    AT target = source_AT
    Y  target = source_Y

So this directly tests whether:
    C intervention -> correct downstream AP -> correct downstream AT -> correct Y
"""

import argparse

import torch

from mcqa_neural_net import load_gemma_model
from color_var_load_color_pair import build_all_color_banks
from mcqa_gradual_das import (
    answer_label_ids,
    answer_logits,
    collect_layer_states,
    output_iia,
    output_label_ids,
    set_seed,
)


def load_readout(path, target, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    w_key = f"W_{target}"
    b_key = f"b_{target}"

    if w_key not in obj or b_key not in obj:
        raise KeyError(
            f"{path} must contain {w_key!r} and {b_key!r}; "
            f"found keys={list(obj.keys())}"
        )

    return (
        obj[w_key].to(device).float(),
        obj[b_key].to(device).float(),
    )


def class_accuracy(logits, targets):
    pred = logits.argmax(dim=-1).detach().cpu()
    targets = targets.detach().cpu().long()
    return float((pred == targets).float().mean())


@torch.no_grad()
def run_c_intervention_capture_downstream(
    model,
    bank,
    c_source_states,
    Q_c,
    c_layer,
    c_token_position,
    ap_layer,
    ap_token_position,
    at_layer,
    at_token_position,
    output_ids,
    batch_size,
):
    """
    Apply ONLY the learned C intervention, then capture downstream AP/AT states
    from the resulting base trajectory.

    All stored token positions are unpadded positions; left-padding offsets are
    added here before indexing the actual batched tensor.
    """
    device = next(model.parameters()).device

    all_ap_states = []
    all_at_states = []
    all_y_logits = []

    n = len(bank["base_input_ids"])

    if c_layer > ap_layer or c_layer > at_layer:
        raise ValueError(
            f"C layer L{c_layer} must be no later than downstream capture "
            f"layers AP=L{ap_layer}, AT=L{at_layer}"
        )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)

        source_c = c_source_states[start:end].to(
            device=device,
            dtype=torch.float32,
        )

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)

        c_pos = (
            pad_offset
            + bank["base_position_by_id"][c_token_position][start:end].to(device)
        )

        ap_pos = (
            pad_offset
            + bank["base_position_by_id"][ap_token_position][start:end].to(device)
        )

        at_pos = (
            pad_offset
            + bank["base_position_by_id"][at_token_position][start:end].to(device)
        )

        captured = {}

        def c_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()

            base_c = hidden[rows, c_pos, :].float()
            delta = source_c - base_c

            hidden_new[rows, c_pos, :] = (
                base_c + ((delta @ Q_c) @ Q_c.T)
            ).to(hidden.dtype)

            if c_layer == ap_layer:
                captured["ap"] = hidden_new[rows, ap_pos, :].detach().float().cpu()

            if c_layer == at_layer:
                captured["at"] = hidden_new[rows, at_pos, :].detach().float().cpu()

            return (
                (hidden_new,) + output[1:]
                if isinstance(output, tuple)
                else hidden_new
            )

        def ap_hook(_module, _inputs, output):
            if ap_layer == c_layer:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            captured["ap"] = hidden[rows, ap_pos, :].detach().float().cpu()

        def at_hook(_module, _inputs, output):
            if at_layer == c_layer:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            captured["at"] = hidden[rows, at_pos, :].detach().float().cpu()

        handles = []

        handles.append(
            model.model.layers[c_layer].register_forward_hook(c_hook)
        )

        if ap_layer != c_layer:
            handles.append(
                model.model.layers[ap_layer].register_forward_hook(ap_hook)
            )

        if at_layer != c_layer and at_layer != ap_layer:
            handles.append(
                model.model.layers[at_layer].register_forward_hook(at_hook)
            )
        elif at_layer == ap_layer and at_layer != c_layer:
            # One hook can capture both states when AP/AT are at same layer.
            handles[-1].remove()

            def shared_hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured["ap"] = hidden[rows, ap_pos, :].detach().float().cpu()
                captured["at"] = hidden[rows, at_pos, :].detach().float().cpu()

            handles[-1] = model.model.layers[ap_layer].register_forward_hook(
                shared_hook
            )

        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(
                    mask.long().cumsum(dim=-1) - 1
                ).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )

            y_logits = answer_logits(
                model,
                outputs,
                mask,
                output_ids,
            )

            all_y_logits.append(y_logits.detach().cpu())

        finally:
            for handle in handles:
                handle.remove()

        if "ap" not in captured:
            raise RuntimeError(
                f"Failed to capture AP state at layer {ap_layer}"
            )

        if "at" not in captured:
            raise RuntimeError(
                f"Failed to capture AT state at layer {at_layer}"
            )

        all_ap_states.append(captured["ap"])
        all_at_states.append(captured["at"])

    return (
        torch.cat(all_ap_states, dim=0),
        torch.cat(all_at_states, dim=0),
        torch.cat(all_y_logits, dim=0),
    )


@torch.no_grad()
def clean_readout_metrics(
    model,
    bank,
    Q_ap,
    W_ap,
    b_ap,
    ap_layer,
    ap_token_position,
    Q_at,
    W_at,
    b_at,
    at_layer,
    at_token_position,
    batch_size,
    device,
):
    """Optional sanity checks for the saved AP/AT readouts on clean base/source."""

    base_ap = collect_layer_states(
        model,
        bank["base_input_ids"],
        bank["base_attention_mask"],
        bank["base_position_by_id"],
        ap_layer,
        ap_token_position,
        batch_size,
    )

    source_ap = collect_layer_states(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        ap_layer,
        ap_token_position,
        batch_size,
    )

    base_at = collect_layer_states(
        model,
        bank["base_input_ids"],
        bank["base_attention_mask"],
        bank["base_position_by_id"],
        at_layer,
        at_token_position,
        batch_size,
    )

    source_at = collect_layer_states(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        at_layer,
        at_token_position,
        batch_size,
    )

    base_ap_logits = (base_ap.to(device) @ Q_ap) @ W_ap.T + b_ap
    source_ap_logits = (source_ap.to(device) @ Q_ap) @ W_ap.T + b_ap

    base_at_logits = (base_at.to(device) @ Q_at) @ W_at.T + b_at
    source_at_logits = (source_at.to(device) @ Q_at) @ W_at.T + b_at

    return {
        "AP clean base": class_accuracy(
            base_ap_logits,
            bank["base_answer_pointer_ids"],
        ),
        "AP clean source": class_accuracy(
            source_ap_logits,
            bank["source_answer_pointer_ids"],
        ),
        "AT clean base": class_accuracy(
            base_at_logits,
            bank["base_answer_label_ids"],
        ),
        "AT clean source": class_accuracy(
            source_at_logits,
            bank["counterfactual_label_ids"]["answer_token"],
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--c-checkpoint",
        default="results/das_c.pt",
    )
    parser.add_argument(
        "--ap-checkpoint",
        default="results_joint_standard_L17/das_ap.pt",
    )
    parser.add_argument(
        "--at-checkpoint",
        default="results_joint_standard_L17/das_at.pt",
    )
    parser.add_argument(
        "--ap-readout",
        default="results_joint_standard_L17/ap_readout_LogisticRegression.pt",
    )
    parser.add_argument(
        "--at-readout",
        default="results_joint_standard_L17/at_readout_LogisticRegression.pt",
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
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--output-space",
        choices=("full", "az"),
        default="full",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    # ------------------------------------------------------------
    # Load C / AP / AT DAS checkpoints and readouts.
    # ------------------------------------------------------------
    c_ckpt = torch.load(
        args.c_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    ap_ckpt = torch.load(
        args.ap_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    at_ckpt = torch.load(
        args.at_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    Q_c = c_ckpt["basis"].to(device).float()
    Q_ap = ap_ckpt["basis"].to(device).float()
    Q_at = at_ckpt["basis"].to(device).float()

    c_layer = int(c_ckpt["layer"])
    ap_layer = int(ap_ckpt["layer"])
    at_layer = int(at_ckpt["layer"])

    c_token_position = c_ckpt["token_position"]
    ap_token_position = ap_ckpt["token_position"]
    at_token_position = at_ckpt["token_position"]

    W_ap, b_ap = load_readout(
        args.ap_readout,
        "AP",
        device,
    )
    W_at, b_at = load_readout(
        args.at_readout,
        "AT",
        device,
    )

    print(
        "[alignment] "
        f"C=L{c_layer}/k{Q_c.shape[1]}/token={c_token_position} | "
        f"AP=L{ap_layer}/k{Q_ap.shape[1]}/token={ap_token_position} | "
        f"AT=L{at_layer}/k{Q_at.shape[1]}/token={at_token_position}"
    )

    # ------------------------------------------------------------
    # Build COLOR evaluation bank.
    # ------------------------------------------------------------
    fit_bank, cal_banks, te_banks = build_all_color_banks(
        model=model,
        tokenizer=tokenizer,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.split == "train":
        bank = fit_bank
    elif args.split == "validation":
        bank = cal_banks["color"]
    else:
        bank = te_banks["color"]

    print(
        f"[bank] split={args.split} "
        f"n={len(bank['base_input_ids'])}"
    )

    # ------------------------------------------------------------
    # Clean readout sanity checks.
    # ------------------------------------------------------------
    clean = clean_readout_metrics(
        model=model,
        bank=bank,
        Q_ap=Q_ap,
        W_ap=W_ap,
        b_ap=b_ap,
        ap_layer=ap_layer,
        ap_token_position=ap_token_position,
        Q_at=Q_at,
        W_at=W_at,
        b_at=b_at,
        at_layer=at_layer,
        at_token_position=at_token_position,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n=== CLEAN READOUT SANITY ===")
    for name, value in clean.items():
        print(f"{name:20s}: {value:.4f}")

    # ------------------------------------------------------------
    # Collect source C states, then intervene ONLY on C.
    # ------------------------------------------------------------
    c_source_states = collect_layer_states(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        c_layer,
        c_token_position,
        args.batch_size,
    )

    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(
        labels,
        args.output_space,
    )

    ap_after_c_states, at_after_c_states, y_after_c_logits = (
        run_c_intervention_capture_downstream(
            model=model,
            bank=bank,
            c_source_states=c_source_states,
            Q_c=Q_c,
            c_layer=c_layer,
            c_token_position=c_token_position,
            ap_layer=ap_layer,
            ap_token_position=ap_token_position,
            at_layer=at_layer,
            at_token_position=at_token_position,
            output_ids=output_ids,
            batch_size=args.batch_size,
        )
    )

    # ------------------------------------------------------------
    # AP after C.
    # ------------------------------------------------------------
    ap_after_c_logits = (
        (ap_after_c_states.to(device) @ Q_ap)
        @ W_ap.T
        + b_ap
    )

    ap_target = bank["source_answer_pointer_ids"]

    ap_after_c_acc = class_accuracy(
        ap_after_c_logits,
        ap_target,
    )

    # ------------------------------------------------------------
    # AT after C.
    # ------------------------------------------------------------
    at_after_c_logits = (
        (at_after_c_states.to(device) @ Q_at)
        @ W_at.T
        + b_at
    )

    at_target = bank["counterfactual_label_ids"]["answer_token"]

    at_after_c_acc = class_accuracy(
        at_after_c_logits,
        at_target,
    )

    # ------------------------------------------------------------
    # Y after C.
    # ------------------------------------------------------------
    y_target = bank["counterfactual_label_ids"]["color"]

    y_after_c_iia = output_iia(
        y_after_c_logits,
        y_target,
        tokenizer,
        args.output_space,
    )

    print("\n=== C-ONLY INTERVENTION -> DOWNSTREAM ===")
    print(
        f"AP readout follows source AP : {ap_after_c_acc:.4f}"
    )
    print(
        f"AT readout follows source AT : {at_after_c_acc:.4f}"
    )
    print(
        f"Y follows source Y           : {y_after_c_iia:.4f}"
    )

    print("\nTargets:")
    print("  AP -> source_answer_pointer_ids")
    print("  AT -> source answer letter")
    print("  Y  -> source answer letter")


if __name__ == "__main__":
    main()
