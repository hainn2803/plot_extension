import argparse
from collections import Counter

import torch

from mcqa_gradual_das import (
    answer_label_ids,
    collect_joint_token_states_in_single_layer,
    collect_layer_states,
    output_label_ids,
)
from mcqa_gradual_das_ap_cross_tokens import run_ap_intervention_and_capture_at
from mcqa_load_randomized_ap_at_c import build_pair_bank
from mcqa_neural_net import load_gemma_model


def load_das_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    Q = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])
    token_position = checkpoint["token_position"]
    return Q, layer, token_position


def load_at_readout(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    W = checkpoint["W_AT"].float().to(device)
    b = checkpoint["b_AT"].float().to(device)
    return W, b


def projected_swap_state(base_state, source_state, Q):
    original_shape = base_state.shape
    base_flat = base_state.reshape(base_state.shape[0], -1).to(Q.device).float()
    source_flat = source_state.reshape(source_state.shape[0], -1).to(Q.device).float()
    swapped_flat = base_flat + ((source_flat - base_flat) @ Q) @ Q.T
    return swapped_flat.reshape(original_shape)


def readout_coordinates(states, Q):
    flat = states.reshape(states.shape[0], -1).to(Q.device).float()
    return flat @ Q


def readout_logits_from_coordinates(z, W, b):
    return z @ W.T + b


def readout_prediction(states, Q, W, b):
    z = readout_coordinates(states, Q)
    logits = readout_logits_from_coordinates(z, W, b)
    return logits.argmax(dim=-1).cpu()


def target_margin(logits, target):
    target = target.to(logits.device)
    rows = torch.arange(len(target), device=logits.device)
    target_logits = logits[rows, target]

    masked = logits.clone()
    masked[rows, target] = float("-inf")
    best_other = masked.max(dim=-1).values
    return target_logits - best_other


def label_name(label):
    label = int(label)
    if 0 <= label < 26:
        return chr(ord("A") + label)
    return str(label)


def print_summary(direct_ok, via_ap_ok):
    both = direct_ok & via_ap_ok
    direct_only = direct_ok & ~via_ap_ok
    via_ap_only = ~direct_ok & via_ap_ok
    both_wrong = ~direct_ok & ~via_ap_ok

    print("\n" + "=" * 80)
    print("SAME AP_AT BANK: DIRECT AT vs AP -> AT")
    print("=" * 80)
    print(f"n = {len(direct_ok)}")
    print(f"Direct AT -> AT readout accuracy = {direct_ok.float().mean().item():.4f}")
    print(f"AP -> generated AT readout accuracy = {via_ap_ok.float().mean().item():.4f}")
    print(f"Both correct = {both.float().mean().item():.4f} ({both.sum().item()}/{len(both)})")
    print(f"Direct AT correct, AP->AT wrong = {direct_only.float().mean().item():.4f} ({direct_only.sum().item()}/{len(both)})")
    print(f"Direct AT wrong, AP->AT correct = {via_ap_only.float().mean().item():.4f} ({via_ap_only.sum().item()}/{len(both)})")
    print(f"Both wrong = {both_wrong.float().mean().item():.4f} ({both_wrong.sum().item()}/{len(both)})")


def print_three_at_readouts(clean_source_ok, direct_ok, via_ap_ok):
    print("\n" + "=" * 80)
    print("THREE AT READOUT ACCURACIES ON THE SAME AP_AT BANK")
    print("=" * 80)
    print(f"Clean source AT readout accuracy       = {clean_source_ok.float().mean().item():.4f}")
    print(f"Direct AT -> AT readout accuracy       = {direct_ok.float().mean().item():.4f}")
    print(f"AP -> generated AT readout accuracy    = {via_ap_ok.float().mean().item():.4f}")


def print_geometry_diagnostics(Q, clean_source_prediction, direct_at_prediction, source_at_states, generated_at, target_at, W_at, b_at):
    eye = torch.eye(Q.shape[1], device=Q.device, dtype=Q.dtype)
    orthogonality_error = (Q.T @ Q - eye).abs().max().item()

    prediction_agreement = (
        clean_source_prediction == direct_at_prediction
    ).float().mean().item()

    z_source = readout_coordinates(source_at_states, Q)
    z_generated = readout_coordinates(generated_at, Q)

    z_distance = torch.linalg.vector_norm(z_generated - z_source, dim=-1)

    logits_source = readout_logits_from_coordinates(z_source, W_at, b_at)
    logits_generated = readout_logits_from_coordinates(z_generated, W_at, b_at)

    margin_source = target_margin(logits_source, target_at)
    margin_generated = target_margin(logits_generated, target_at)

    print("\n" + "=" * 80)
    print("AT SUBSPACE / READOUT DIAGNOSTICS")
    print("=" * 80)
    print(f"max |Q^T Q - I|                         = {orthogonality_error:.8f}")
    print(f"Clean-source/direct prediction agreement = {prediction_agreement:.4f}")

    print("\nAT coordinate distance: ||z_AP - z_source||_2")
    print(f"mean   = {z_distance.mean().item():.4f}")
    print(f"median = {z_distance.median().item():.4f}")
    print(f"std    = {z_distance.std(unbiased=False).item():.4f}")
    print(f"min    = {z_distance.min().item():.4f}")
    print(f"max    = {z_distance.max().item():.4f}")

    print("\nTarget-class margin: logit(target) - max logit(other)")
    print(f"Clean source mean margin    = {margin_source.mean().item():.4f}")
    print(f"Clean source median margin  = {margin_source.median().item():.4f}")
    print(f"AP-generated mean margin    = {margin_generated.mean().item():.4f}")
    print(f"AP-generated median margin  = {margin_generated.median().item():.4f}")

    print("\nFraction with positive target margin")
    print(f"Clean source                = {(margin_source > 0).float().mean().item():.4f}")
    print(f"AP-generated                = {(margin_generated > 0).float().mean().item():.4f}")


def print_accuracy_by_target(direct_ok, via_ap_ok, target):
    print("\n" + "=" * 80)
    print("ACCURACY BY TARGET AT")
    print("=" * 80)
    print(f"{'target':>8s} {'n':>6s} {'direct_AT':>12s} {'AP_to_AT':>12s}")

    for label in sorted(target.unique().tolist()):
        mask = target == label
        n = int(mask.sum())
        direct_acc = direct_ok[mask].float().mean().item()
        via_ap_acc = via_ap_ok[mask].float().mean().item()
        print(f"{label_name(label):>8s} {n:6d} {direct_acc:12.4f} {via_ap_acc:12.4f}")


def print_accuracy_by_transition(direct_ok, via_ap_ok, base_at, target_at):
    transitions = Counter(zip(base_at.tolist(), target_at.tolist()))

    print("\n" + "=" * 80)
    print("ACCURACY BY BASE AT -> TARGET AT TRANSITION")
    print("=" * 80)
    print(f"{'transition':>12s} {'n':>6s} {'direct_AT':>12s} {'AP_to_AT':>12s}")

    for base_label, target_label in sorted(transitions):
        mask = (base_at == base_label) & (target_at == target_label)
        n = int(mask.sum())
        direct_acc = direct_ok[mask].float().mean().item()
        via_ap_acc = via_ap_ok[mask].float().mean().item()
        transition = f"{label_name(base_label)}->{label_name(target_label)}"
        print(f"{transition:>12s} {n:6d} {direct_acc:12.4f} {via_ap_acc:12.4f}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at_no_distractor")
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--at-checkpoint", default="results_standard_das_layer_grid_seed0/trained_handles/AT_L25/das_at.pt")
    parser.add_argument("--at-readout", default="results_standard_das_layer_grid_seed0/trained_handles/AT_L25/at_readout_LogisticRegression.pt")
    parser.add_argument("--ap-checkpoint", default="results_standard_das_layer_grid_seed0/trained_handles/AP_L14/das_ap.pt")

    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    Q_at, at_layer, at_token_position = load_das_checkpoint(args.at_checkpoint, device)
    Q_ap, ap_layer, ap_token_positions = load_das_checkpoint(args.ap_checkpoint, device)
    W_at, b_at = load_at_readout(args.at_readout, device)

    if not isinstance(ap_token_positions, list):
        raise ValueError(f"AP checkpoint token_position must be a list, got {ap_token_positions}")

    if ap_layer >= at_layer:
        raise ValueError(f"Need AP layer < AT layer, got AP={ap_layer}, AT={at_layer}")

    bank_seed = args.seed + 3003
    print("=== BUILD AP_AT BANK ===")
    print(f"split = {args.split}")
    print(f"eval_size = {args.eval_size}")
    print(f"seed = {bank_seed}")

    ap_at = build_pair_bank(
        model,
        tokenizer,
        args.data_path,
        args.split,
        "AP_AT",
        args.eval_size,
        args.batch_size,
        bank_seed,
    )

    target_after_at = ap_at["intervention_targets"]["AT_after_AT_intervene"].cpu()
    target_after_ap = ap_at["intervention_targets"]["AT_after_AP_intervene"].cpu()

    if not torch.equal(target_after_at, target_after_ap):
        mismatch = target_after_at != target_after_ap
        print(f"Target mismatch count = {mismatch.sum().item()} / {len(mismatch)}")
        first = torch.nonzero(mismatch).flatten()[:10].tolist()

        for i in first:
            print(
                f"i={i}: "
                f"AT_after_AT={label_name(target_after_at[i])}, "
                f"AT_after_AP={label_name(target_after_ap[i])}"
            )

        raise ValueError("AP_AT bank does not have the same target AT for direct AT and AP->AT interventions")

    target_at = target_after_at
    base_at_target = ap_at["base_answer_label_ids"].cpu()

    print("\nTarget check passed:")
    print("AT_after_AT_intervene == AT_after_AP_intervene for every AP_AT pair")

    print("\n=== DIRECT AT INTERVENTION ===")

    base_at_states = collect_layer_states(
        model,
        ap_at["base_input_ids"],
        ap_at["base_attention_mask"],
        ap_at["base_position_by_id"],
        at_layer,
        at_token_position,
        args.batch_size,
    )

    source_at_states = collect_layer_states(
        model,
        ap_at["source_input_ids"],
        ap_at["source_attention_mask"],
        ap_at["source_position_by_id"],
        at_layer,
        at_token_position,
        args.batch_size,
    )

    clean_source_prediction = readout_prediction(source_at_states, Q_at, W_at, b_at)
    clean_source_ok = clean_source_prediction == target_at

    post_direct_at = projected_swap_state(base_at_states, source_at_states, Q_at)
    direct_at_prediction = readout_prediction(post_direct_at, Q_at, W_at, b_at)
    direct_ok = direct_at_prediction == target_at

    print("\n=== AP INTERVENTION -> GENERATED AT ===")

    source_ap_states = collect_joint_token_states_in_single_layer(
        model,
        ap_at["source_input_ids"],
        ap_at["source_attention_mask"],
        ap_at["source_position_by_id"],
        ap_layer,
        ap_token_positions,
        args.batch_size,
    )

    output_ids = output_label_ids(answer_label_ids(tokenizer), args.output_space)

    _, generated_at = run_ap_intervention_and_capture_at(
        model,
        ap_at,
        source_ap_states,
        Q_ap,
        ap_layer,
        at_layer,
        output_ids,
        ap_token_positions,
        at_token_position,
        args.batch_size,
        False,
    )

    via_ap_prediction = readout_prediction(generated_at, Q_at, W_at, b_at)
    via_ap_ok = via_ap_prediction == target_at

    print_summary(direct_ok, via_ap_ok)
    print_three_at_readouts(clean_source_ok, direct_ok, via_ap_ok)
    print_geometry_diagnostics(
        Q_at,
        clean_source_prediction,
        direct_at_prediction,
        source_at_states,
        generated_at,
        target_at,
        W_at,
        b_at,
    )
    print_accuracy_by_target(direct_ok, via_ap_ok, target_at)
    print_accuracy_by_transition(direct_ok, via_ap_ok, base_at_target, target_at)


if __name__ == "__main__":
    main()
