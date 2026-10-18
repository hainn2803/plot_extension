"""
Evaluate the causal chain after intervening ONLY on C:

    C  --readout--> AP
    C  --readout--> AT
    C -----------> Y

and mediator replay tests:

    C -> generated AP -> replay AP into clean base -> AT readout / Y
    C -> generated AT -> replay AT into clean base -> Y

Important positions:
    C  : color token
    AP : last_token
    AT : last_token

No AP or AT intervention is used in the initial C trajectory.
AP/AT interventions happen ONLY in the mediator replay tests.
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


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_basis(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    for key in ("basis", "Q", "best_basis"):
        if key in obj:
            Q = obj[key].to(device).float()
            break
    else:
        raise KeyError(
            f"Could not find DAS basis in {path}. "
            f"Keys={list(obj.keys())}"
        )

    if "layer" not in obj:
        raise KeyError(
            f"Checkpoint {path} does not contain 'layer'. "
            f"Keys={list(obj.keys())}"
        )

    return Q, int(obj["layer"]), obj


def load_readout(path, variable, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if variable == "AP":
        weight_keys = ("W_AP", "W_ap", "W")
        bias_keys = ("b_AP", "b_ap", "b")
    elif variable == "AT":
        weight_keys = ("W_AT", "W_at", "W")
        bias_keys = ("b_AT", "b_at", "b")
    else:
        raise ValueError(variable)

    W = next(
        (obj[k] for k in weight_keys if k in obj),
        None,
    )
    b = next(
        (obj[k] for k in bias_keys if k in obj),
        None,
    )

    if W is None or b is None:
        raise KeyError(
            f"Could not find {variable} readout in {path}. "
            f"Keys={list(obj.keys())}"
        )

    return W.to(device).float(), b.to(device).float()


def class_accuracy(logits, targets):
    pred = logits.argmax(dim=-1).detach().cpu()
    targets = targets.detach().cpu().long()
    return float(
        (pred == targets).float().mean().item()
    )


# ---------------------------------------------------------------------
# Clean forward
# ---------------------------------------------------------------------

@torch.no_grad()
def run_clean(
    model,
    bank,
    output_ids,
    batch_size,
):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(
        0,
        len(bank["base_input_ids"]),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(bank["base_input_ids"]),
        )

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)

        outputs = model.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=(
                mask.long().cumsum(dim=-1) - 1
            ).clamp(min=0),
            use_cache=False,
            return_dict=True,
        )

        all_logits.append(
            answer_logits(
                model,
                outputs,
                mask,
                output_ids,
            ).cpu()
        )

    return torch.cat(all_logits, dim=0)


# ---------------------------------------------------------------------
# 1) C-only intervention trajectory
# ---------------------------------------------------------------------

@torch.no_grad()
def run_c_only_and_capture_ap_at(
    model,
    bank,
    source_c_states,
    Q_c,
    c_layer,
    ap_layer,
    at_layer,
    output_ids,
    batch_size,
):
    """
    One forward trajectory:

      clean base
        -> intervene C at (L_C, color)
        -> capture AP state at (L_AP, last_token)
        -> capture AT state at (L_AT, last_token)
        -> Y

    AP and AT are NOT intervened here.
    """
    device = next(model.parameters()).device

    if not (c_layer < ap_layer <= at_layer):
        raise ValueError(
            "Expected C before AP before/equal AT, but got "
            f"C=L{c_layer}, AP=L{ap_layer}, AT=L{at_layer}"
        )

    all_ap = []
    all_at = []
    all_y = []

    n = len(bank["base_input_ids"])

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)

        source_c = source_c_states[start:end].to(
            device=device,
            dtype=torch.float32,
        )

        rows = torch.arange(
            len(ids),
            device=device,
        )

        pad_offset = (mask == 0).sum(dim=1)

        c_pos = (
            pad_offset
            + bank["base_position_by_id"]["color"][start:end].to(device)
        )

        last_pos = (
            pad_offset
            + bank["base_position_by_id"]["last_token"][start:end].to(device)
        )

        captured = {}

        def c_hook(_module, _inputs, output):
            hidden = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            hidden_new = hidden.clone()

            base_c = hidden[rows, c_pos, :].float()

            hidden_new[rows, c_pos, :] = (
                base_c
                + (
                    ((source_c - base_c) @ Q_c)
                    @ Q_c.T
                )
            ).to(hidden.dtype)

            return (
                (hidden_new,) + output[1:]
                if isinstance(output, tuple)
                else hidden_new
            )

        def ap_hook(_module, _inputs, output):
            hidden = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            captured["ap"] = (
                hidden[rows, last_pos, :]
                .detach()
                .float()
                .cpu()
            )

        def at_hook(_module, _inputs, output):
            hidden = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            captured["at"] = (
                hidden[rows, last_pos, :]
                .detach()
                .float()
                .cpu()
            )

        c_handle = (
            model.model.layers[c_layer]
            .register_forward_hook(c_hook)
        )

        if ap_layer == at_layer:
            def shared_hook(_module, _inputs, output):
                hidden = (
                    output[0]
                    if isinstance(output, tuple)
                    else output
                )

                state = (
                    hidden[rows, last_pos, :]
                    .detach()
                    .float()
                    .cpu()
                )

                captured["ap"] = state
                captured["at"] = state

            ap_handle = (
                model.model.layers[ap_layer]
                .register_forward_hook(shared_hook)
            )
            at_handle = None

        else:
            ap_handle = (
                model.model.layers[ap_layer]
                .register_forward_hook(ap_hook)
            )
            at_handle = (
                model.model.layers[at_layer]
                .register_forward_hook(at_hook)
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

            all_y.append(
                y_logits.detach().cpu()
            )

        finally:
            c_handle.remove()
            ap_handle.remove()

            if at_handle is not None:
                at_handle.remove()

        if "ap" not in captured:
            raise RuntimeError(
                "AP state was not captured."
            )

        if "at" not in captured:
            raise RuntimeError(
                "AT state was not captured."
            )

        all_ap.append(captured["ap"])
        all_at.append(captured["at"])

    return (
        torch.cat(all_ap, dim=0),
        torch.cat(all_at, dim=0),
        torch.cat(all_y, dim=0),
    )


# ---------------------------------------------------------------------
# 2) AP mediator replay
# ---------------------------------------------------------------------

@torch.no_grad()
def replay_ap_and_capture_at(
    model,
    bank,
    generated_ap_states,
    Q_ap,
    ap_layer,
    at_layer,
    output_ids,
    batch_size,
):
    """
    Start again from CLEAN BASE.

    Replay only the AP subspace generated by the C intervention:

        clean base
          -> replay generated AP at (L_AP, last_token)
          -> capture resulting AT at (L_AT, last_token)
          -> Y

    This is the AP mediator test for the C intervention.
    """
    device = next(model.parameters()).device

    if ap_layer > at_layer:
        raise ValueError(
            f"AP layer L{ap_layer} is after AT layer L{at_layer}."
        )

    all_at = []
    all_y = []

    n = len(bank["base_input_ids"])

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)

        donor_ap = generated_ap_states[start:end].to(
            device=device,
            dtype=torch.float32,
        )

        rows = torch.arange(
            len(ids),
            device=device,
        )

        pad_offset = (mask == 0).sum(dim=1)

        last_pos = (
            pad_offset
            + bank["base_position_by_id"]["last_token"][start:end].to(device)
        )

        captured = {}

        def ap_hook(_module, _inputs, output):
            hidden = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            hidden_new = hidden.clone()
            base_ap = hidden[rows, last_pos, :].float()

            hidden_new[rows, last_pos, :] = (
                base_ap
                + (
                    ((donor_ap - base_ap) @ Q_ap)
                    @ Q_ap.T
                )
            ).to(hidden.dtype)

            if ap_layer == at_layer:
                captured["at"] = (
                    hidden_new[rows, last_pos, :]
                    .detach()
                    .float()
                    .cpu()
                )

            return (
                (hidden_new,) + output[1:]
                if isinstance(output, tuple)
                else hidden_new
            )

        def at_hook(_module, _inputs, output):
            hidden = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            captured["at"] = (
                hidden[rows, last_pos, :]
                .detach()
                .float()
                .cpu()
            )

        ap_handle = (
            model.model.layers[ap_layer]
            .register_forward_hook(ap_hook)
        )

        if at_layer == ap_layer:
            at_handle = None
        else:
            at_handle = (
                model.model.layers[at_layer]
                .register_forward_hook(at_hook)
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

            all_y.append(
                y_logits.detach().cpu()
            )

        finally:
            ap_handle.remove()

            if at_handle is not None:
                at_handle.remove()

        if "at" not in captured:
            raise RuntimeError(
                "AT state after AP replay was not captured."
            )

        all_at.append(captured["at"])

    return (
        torch.cat(all_at, dim=0),
        torch.cat(all_y, dim=0),
    )


# ---------------------------------------------------------------------
# 3) AT mediator replay
# ---------------------------------------------------------------------

@torch.no_grad()
def replay_at(
    model,
    bank,
    generated_at_states,
    Q_at,
    at_layer,
    output_ids,
    batch_size,
):
    """
    Start again from CLEAN BASE.

    Replay only the AT subspace generated by the C intervention:

        clean base
          -> replay generated AT at (L_AT, last_token)
          -> Y

    This is the AT mediator test for the C intervention.
    """
    device = next(model.parameters()).device
    all_y = []

    n = len(bank["base_input_ids"])

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)

        donor_at = generated_at_states[start:end].to(
            device=device,
            dtype=torch.float32,
        )

        rows = torch.arange(
            len(ids),
            device=device,
        )

        pad_offset = (mask == 0).sum(dim=1)

        last_pos = (
            pad_offset
            + bank["base_position_by_id"]["last_token"][start:end].to(device)
        )

        def at_hook(_module, _inputs, output):
            hidden = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            hidden_new = hidden.clone()
            base_at = hidden[rows, last_pos, :].float()

            hidden_new[rows, last_pos, :] = (
                base_at
                + (
                    ((donor_at - base_at) @ Q_at)
                    @ Q_at.T
                )
            ).to(hidden.dtype)

            return (
                (hidden_new,) + output[1:]
                if isinstance(output, tuple)
                else hidden_new
            )

        at_handle = (
            model.model.layers[at_layer]
            .register_forward_hook(at_hook)
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

            all_y.append(
                y_logits.detach().cpu()
            )

        finally:
            at_handle.remove()

    return torch.cat(all_y, dim=0)


# ---------------------------------------------------------------------
# Clean readout sanity
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_clean_readouts(
    model,
    bank,
    Q_ap,
    W_ap,
    b_ap,
    ap_layer,
    Q_at,
    W_at,
    b_at,
    at_layer,
    batch_size,
    device,
):
    base_ap = collect_layer_states(
        model,
        bank["base_input_ids"],
        bank["base_attention_mask"],
        bank["base_position_by_id"],
        ap_layer,
        "last_token",
        batch_size,
    )

    source_ap = collect_layer_states(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        ap_layer,
        "last_token",
        batch_size,
    )

    base_at = collect_layer_states(
        model,
        bank["base_input_ids"],
        bank["base_attention_mask"],
        bank["base_position_by_id"],
        at_layer,
        "last_token",
        batch_size,
    )

    source_at = collect_layer_states(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        at_layer,
        "last_token",
        batch_size,
    )

    base_ap_logits = (
        (base_ap.to(device) @ Q_ap)
        @ W_ap.T
        + b_ap
    )

    source_ap_logits = (
        (source_ap.to(device) @ Q_ap)
        @ W_ap.T
        + b_ap
    )

    base_at_logits = (
        (base_at.to(device) @ Q_at)
        @ W_at.T
        + b_at
    )

    source_at_logits = (
        (source_at.to(device) @ Q_at)
        @ W_at.T
        + b_at
    )

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--c-checkpoint",
        default="results/das_c.pt",
    )
    parser.add_argument(
        "--ap-checkpoint",
        default="results_joint_gradual_L17/das_ap.pt",
    )
    parser.add_argument(
        "--at-checkpoint",
        default="results_joint_gradual_L17/das_at.pt",
    )
    parser.add_argument(
        "--ap-readout",
        default=(
            "results_joint_gradual_L17/"
            "ap_readout_LogisticRegression.pt"
        ),
    )
    parser.add_argument(
        "--at-readout",
        default=(
            "results_joint_gradual_L17/"
            "at_readout_LogisticRegression.pt"
        ),
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
        default=0,
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
    # Load saved C / AP / AT alignments.
    # ------------------------------------------------------------
    Q_c, c_layer, _ = load_basis(
        args.c_checkpoint,
        device,
    )

    Q_ap, ap_layer, _ = load_basis(
        args.ap_checkpoint,
        device,
    )

    Q_at, at_layer, _ = load_basis(
        args.at_checkpoint,
        device,
    )

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

    # FIXED by construction for this experiment.
    c_token_position = "color"
    downstream_token_position = "last_token"

    print(
        "[alignment] "
        f"C=L{c_layer}/k{Q_c.shape[1]}/token={c_token_position} | "
        f"AP=L{ap_layer}/k{Q_ap.shape[1]}/token={downstream_token_position} | "
        f"AT=L{at_layer}/k{Q_at.shape[1]}/token={downstream_token_position}"
    )

    # ------------------------------------------------------------
    # COLOR bank.
    # ------------------------------------------------------------
    fit_bank, cal_banks, te_banks = (
        build_all_color_banks(
            model=model,
            tokenizer=tokenizer,
            ft_size=args.ft_size,
            cal_size=args.cal_size,
            te_size=args.te_size,
            batch_size=args.batch_size,
            seed=args.seed,
        )
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
    # Targets.
    # ------------------------------------------------------------
    ap_target = bank["counterfactual_label_ids"]["answer_pointer"]

    at_target = (
        bank["counterfactual_label_ids"]
        ["answer_token"]
    )

    y_target = (
        bank["counterfactual_label_ids"]
        ["color"]
    )

    labels = answer_label_ids(tokenizer)

    output_ids = output_label_ids(
        labels,
        args.output_space,
    )

    # ------------------------------------------------------------
    # Clean readout sanity.
    # ------------------------------------------------------------
    clean = evaluate_clean_readouts(
        model=model,
        bank=bank,
        Q_ap=Q_ap,
        W_ap=W_ap,
        b_ap=b_ap,
        ap_layer=ap_layer,
        Q_at=Q_at,
        W_at=W_at,
        b_at=b_at,
        at_layer=at_layer,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n=== CLEAN READOUT SANITY ===")

    for name, value in clean.items():
        print(
            f"{name:22s}: {value:.4f}"
        )

    # ------------------------------------------------------------
    # Collect SOURCE C state for the C intervention.
    # ------------------------------------------------------------
    source_c_states = collect_layer_states(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        c_layer,
        "color",
        args.batch_size,
    )

    # ------------------------------------------------------------
    # DIRECT C trajectory:
    # C intervention only.
    # ------------------------------------------------------------
    (
        generated_ap_states,
        generated_at_states,
        c_direct_y_logits,
    ) = run_c_only_and_capture_ap_at(
        model=model,
        bank=bank,
        source_c_states=source_c_states,
        Q_c=Q_c,
        c_layer=c_layer,
        ap_layer=ap_layer,
        at_layer=at_layer,
        output_ids=output_ids,
        batch_size=args.batch_size,
    )

    ap_after_c_logits = (
        (generated_ap_states.to(device) @ Q_ap)
        @ W_ap.T
        + b_ap
    )

    at_after_c_logits = (
        (generated_at_states.to(device) @ Q_at)
        @ W_at.T
        + b_at
    )

    c_to_ap_readout = class_accuracy(
        ap_after_c_logits,
        ap_target,
    )

    c_to_at_readout = class_accuracy(
        at_after_c_logits,
        at_target,
    )

    c_to_y_iia = output_iia(
        c_direct_y_logits,
        y_target,
        tokenizer,
        args.output_space,
    )

    # ------------------------------------------------------------
    # AP MEDIATOR:
    #
    # C -> generated AP
    # clean base + replay generated AP
    # -> AT readout
    # -> Y
    # ------------------------------------------------------------
    (
        at_after_ap_replay_states,
        ap_mediator_y_logits,
    ) = replay_ap_and_capture_at(
        model=model,
        bank=bank,
        generated_ap_states=generated_ap_states,
        Q_ap=Q_ap,
        ap_layer=ap_layer,
        at_layer=at_layer,
        output_ids=output_ids,
        batch_size=args.batch_size,
    )

    at_after_ap_replay_logits = (
        (
            at_after_ap_replay_states.to(device)
            @ Q_at
        )
        @ W_at.T
        + b_at
    )

    ap_mediator_at_readout = class_accuracy(
        at_after_ap_replay_logits,
        at_target,
    )

    ap_mediator_y_iia = output_iia(
        ap_mediator_y_logits,
        y_target,
        tokenizer,
        args.output_space,
    )

    # ------------------------------------------------------------
    # AT MEDIATOR:
    #
    # C -> generated AT
    # clean base + replay generated AT
    # -> Y
    # ------------------------------------------------------------
    at_mediator_y_logits = replay_at(
        model=model,
        bank=bank,
        generated_at_states=generated_at_states,
        Q_at=Q_at,
        at_layer=at_layer,
        output_ids=output_ids,
        batch_size=args.batch_size,
    )

    at_mediator_y_iia = output_iia(
        at_mediator_y_logits,
        y_target,
        tokenizer,
        args.output_space,
    )

    # ------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------
    print(
        "\n"
        "=== C-ONLY DIRECT PROPAGATION ==="
    )

    print(
        "C -> AP readout "
        f"(target CF_AP) : {c_to_ap_readout:.4f}"
    )

    print(
        "C -> AT readout "
        f"(target CF_AT) : {c_to_at_readout:.4f}"
    )

    print(
        "C -> Y direct "
        f"(target CF_Y)  : {c_to_y_iia:.4f}"
    )

    print(
        "\n"
        "=== AP MEDIATOR REPLAY ==="
    )

    print(
        "C -> generated AP -> replay AP "
        "-> AT readout : "
        f"{ap_mediator_at_readout:.4f}"
    )

    print(
        "C -> generated AP -> replay AP "
        "-> Y          : "
        f"{ap_mediator_y_iia:.4f}"
    )

    print(
        "\n"
        "=== AT MEDIATOR REPLAY ==="
    )

    print(
        "C -> generated AT -> replay AT "
        "-> Y          : "
        f"{at_mediator_y_iia:.4f}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  DIRECT metrics ask whether a C intervention "
        "naturally produces counterfactual AP / AT / Y."
    )

    print(
        "  AP mediator asks whether the AP state generated "
        "by C is sufficient, when replayed alone into a "
        "clean base trajectory, to recover downstream AT / Y."
    )

    print(
        "  AT mediator asks whether the AT state generated "
        "by C is sufficient, when replayed alone into a "
        "clean base trajectory, to recover Y."
    )


if __name__ == "__main__":
    main()
