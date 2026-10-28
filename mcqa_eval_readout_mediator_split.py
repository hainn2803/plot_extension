"""Evaluate clean readouts plus AP/C causal propagation and mediator replay."""

import argparse
from pathlib import Path

import torch

from mcqa_gradual_das import answer_label_ids, answer_logits, collect_layer_states, output_iia, output_label_ids, set_seed
from mcqa_gradual_das_ap import run_ap_intervention_and_capture_at, run_frozen_at_mediator
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_c_balanced_eval import build_balanced_eval_bank
from mcqa_load_randomized_ap_at_c import build_pair_bank, build_ap_readout_datasets, build_at_readout_datasets, build_c_readout_datasets, build_shared_readout_banks


def load_ckpt(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    basis = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])
    position = checkpoint.get("token_position", "last_token")
    return basis, layer, position


def load_readout(path, name, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint[f"W_{name}"].float().to(device)
    bias = checkpoint[f"b_{name}"].float().to(device)
    return weights, bias, checkpoint


def accuracy(logits, target):
    prediction = logits.argmax(-1).cpu()
    return float((prediction == target.cpu()).float().mean())


def split_masks(bank, variable):
    change_type = f"{variable}_only"
    preserve_type = f"{variable}_preserve"

    change_mask = []
    preserve_mask = []

    for pair_type in bank["pair_types"]:
        change_mask.append(pair_type == change_type)
        preserve_mask.append(pair_type == preserve_type)

    change_mask = torch.tensor(change_mask, dtype=torch.bool)
    preserve_mask = torch.tensor(preserve_mask, dtype=torch.bool)

    if change_mask.sum().item() == 0:
        raise ValueError(f"No changed examples found for {variable}")

    if preserve_mask.sum().item() == 0:
        raise ValueError(f"No fixed examples found for {variable}")

    return change_mask, preserve_mask


def split_accuracy_from_correct(correct, bank, variable):
    correct = correct.cpu().bool()
    change_mask, preserve_mask = split_masks(bank, variable)

    overall = float(correct.float().mean())
    changed = float(correct[change_mask].float().mean())
    fixed = float(correct[preserve_mask].float().mean())

    return overall, changed, fixed


def output_correct(logits, target, tokenizer, output_space):
    prediction = logits.argmax(-1).cpu()

    if output_space == "az":
        return prediction == target.cpu()

    decoded = []
    for token_id in prediction:
        text = tokenizer.decode([int(token_id)]).strip().upper()
        decoded.append(text)

    expected = []
    for target_id in target:
        expected.append(chr(ord("A") + int(target_id)))

    correct = []
    for prediction_text, expected_text in zip(decoded, expected):
        correct.append(prediction_text == expected_text)

    return torch.tensor(correct)


@torch.no_grad()
def readout_accuracy(model, bank, Q, layer, pos, W, b, target, batch_size, side):
    states = collect_layer_states(model, bank[f"{side}_input_ids"], bank[f"{side}_attention_mask"], bank[f"{side}_position_by_id"], layer, pos, batch_size)
    logits = (states.to(Q.device) @ Q) @ W.T + b
    return accuracy(logits, target)


@torch.no_grad()
def readout_accuracy_split(model, bank, Q, layer, pos, W, b, target, batch_size, side, variable):
    states = collect_layer_states(model, bank[f"{side}_input_ids"], bank[f"{side}_attention_mask"], bank[f"{side}_position_by_id"], layer, pos, batch_size)
    logits = (states.to(Q.device) @ Q) @ W.T + b
    correct = logits.argmax(-1).cpu() == target.cpu()
    return split_accuracy_from_correct(correct, bank, variable)


def color_labels(bank, color_to_class, side):
    labels = []

    for pair in bank["pairs"]:
        color = str(pair[f"{side}_color"]).strip().casefold()
        labels.append(color_to_class[color])

    return torch.tensor(labels)


@torch.no_grad()
def replay_ap_capture_at(model, bank, generated_ap, Q_ap, L_ap, pos_ap, L_at, pos_at, output_ids, batch_size):
    device = next(model.parameters()).device
    all_y = []
    all_at = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = generated_ap[start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        pad = (mask == 0).sum(1)
        ap_idx = pad + bank["base_position_by_id"][pos_ap][start:end].to(device)
        at_idx = pad + bank["base_position_by_id"][pos_at][start:end].to(device)
        captured = {}

        def ap_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base = hidden[rows, ap_idx].float()
            delta = ((donor.float() - base) @ Q_ap) @ Q_ap.T
            new_hidden[rows, ap_idx] = (base + delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        def at_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured["at"] = hidden[rows, at_idx].float()

        ap_handle = model.model.layers[L_ap].register_forward_hook(ap_hook)
        at_handle = model.model.layers[L_at].register_forward_hook(at_hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            y = answer_logits(model, outputs, mask, output_ids)
            all_y.append(y.detach().cpu())
            all_at.append(captured["at"].detach().cpu())
        finally:
            ap_handle.remove()
            at_handle.remove()

    return torch.cat(all_y), torch.cat(all_at)


@torch.no_grad()
def eval_ap(model, tokenizer, bank, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    source_ap = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], L_ap, pos_ap, batch_size)
    direct_y, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, L_ap, L_at, output_ids, pos_ap, batch_size, False)
    replay_y = run_frozen_at_mediator(model, bank, generated_at, Q_at, L_at, output_ids, pos_at, batch_size, False)

    targets = bank["intervention_targets"]
    at_target = targets["AT_after_AP_intervene"]
    y_target = targets["Y_after_AP_intervene"]

    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    at_ok = at_logits.argmax(-1).cpu() == at_target.cpu()
    y_ok = output_correct(direct_y, y_target, tokenizer, output_space)

    at_correct_count = at_ok.sum().item()
    y_correct_count = y_ok.sum().item()

    if at_correct_count > 0:
        y_given_at = float((at_ok & y_ok).sum().item() / at_correct_count)
    else:
        y_given_at = 0.0

    if y_correct_count > 0:
        at_given_y = float((at_ok & y_ok).sum().item() / y_correct_count)
    else:
        at_given_y = 0.0

    replay_ok = output_correct(replay_y, y_target, tokenizer, output_space)

    result = {
        "Y_after_AP": output_iia(direct_y, y_target, tokenizer, output_space),
        "AT_after_AP": accuracy(at_logits, at_target),
        "Joint_AT_Y_after_AP": float((at_ok & y_ok).float().mean()),
        "Y_correct_given_AT_correct_after_AP": y_given_at,
        "AT_correct_given_Y_correct_after_AP": at_given_y,
        "Y_via_AT_replay": output_iia(replay_y, y_target, tokenizer, output_space),
    }

    has_split = False
    for pair_type in bank["pair_types"]:
        if pair_type == "AP_preserve":
            has_split = True

    if has_split:
        _, y_changed, y_fixed = split_accuracy_from_correct(y_ok, bank, "AP")
        _, at_changed, at_fixed = split_accuracy_from_correct(at_ok, bank, "AP")
        _, joint_changed, joint_fixed = split_accuracy_from_correct(at_ok & y_ok, bank, "AP")
        _, replay_changed, replay_fixed = split_accuracy_from_correct(replay_ok, bank, "AP")

        result["Y_after_AP_changed"] = y_changed
        result["Y_after_AP_fixed"] = y_fixed
        result["AT_after_AP_changed"] = at_changed
        result["AT_after_AP_fixed"] = at_fixed
        result["Joint_AT_Y_after_AP_changed"] = joint_changed
        result["Joint_AT_Y_after_AP_fixed"] = joint_fixed
        result["Y_via_AT_replay_changed"] = replay_changed
        result["Y_via_AT_replay_fixed"] = replay_fixed

    return result


@torch.no_grad()
def eval_c(model, tokenizer, bank, Q_c, L_c, pos_c, Q_ap, L_ap, pos_ap, W_ap, b_ap, Q_at, L_at, pos_at, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], L_c, pos_c, batch_size)
    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, L_c, L_ap, L_at, output_ids, pos_c, pos_ap, pos_at, batch_size, False)
    ap_replay_y, at_after_ap = replay_ap_capture_at(model, bank, generated_ap, Q_ap, L_ap, pos_ap, L_at, pos_at, output_ids, batch_size)
    at_replay_y = run_frozen_mediator(model, bank, generated_at, Q_at, L_at, output_ids, pos_at, batch_size, False)

    targets = bank["intervention_targets"]
    ap_target = targets["AP_after_C_intervene"]
    at_target = targets["AT_after_C_intervene"]
    y_target = targets["Y_after_C_intervene"]

    ap_logits = (generated_ap.to(Q_ap.device) @ Q_ap) @ W_ap.T + b_ap
    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    at_after_ap_logits = (at_after_ap.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    ap_ok = ap_logits.argmax(-1).cpu() == ap_target.cpu()
    at_ok = at_logits.argmax(-1).cpu() == at_target.cpu()
    y_ok = output_correct(direct_y, y_target, tokenizer, output_space)

    ap_correct_count = ap_ok.sum().item()
    at_correct_count = at_ok.sum().item()
    y_correct_count = y_ok.sum().item()

    if ap_correct_count > 0:
        y_given_ap = float((ap_ok & y_ok).sum().item() / ap_correct_count)
    else:
        y_given_ap = 0.0

    if at_correct_count > 0:
        y_given_at = float((at_ok & y_ok).sum().item() / at_correct_count)
    else:
        y_given_at = 0.0

    if y_correct_count > 0:
        ap_given_y = float((ap_ok & y_ok).sum().item() / y_correct_count)
        at_given_y = float((at_ok & y_ok).sum().item() / y_correct_count)
    else:
        ap_given_y = 0.0
        at_given_y = 0.0

    at_ap_ok = at_after_ap_logits.argmax(-1).cpu() == at_target.cpu()
    y_ap_ok = output_correct(ap_replay_y, y_target, tokenizer, output_space)

    at_replay_ok = output_correct(at_replay_y, y_target, tokenizer, output_space)

    result = {
        "Y_after_C": output_iia(direct_y, y_target, tokenizer, output_space),
        "AP_after_C": accuracy(ap_logits, ap_target),
        "AT_after_C": accuracy(at_logits, at_target),
        "Joint_AT_Y_after_C": float((at_ok & y_ok).float().mean()),
        "Joint_AP_AT_Y_after_C": float((ap_ok & at_ok & y_ok).float().mean()),
        "Y_correct_given_AP_correct_after_C": y_given_ap,
        "AP_correct_given_Y_correct_after_C": ap_given_y,
        "Y_correct_given_AT_correct_after_C": y_given_at,
        "AT_correct_given_Y_correct_after_C": at_given_y,
        "AT_via_AP_replay": accuracy(at_after_ap_logits, at_target),
        "Y_via_AP_replay": output_iia(ap_replay_y, y_target, tokenizer, output_space),
        "Joint_AT_Y_via_AP_replay": float((at_ap_ok & y_ap_ok).float().mean()),
        "Y_via_AT_replay": output_iia(at_replay_y, y_target, tokenizer, output_space),
    }

    has_split = False
    for pair_type in bank["pair_types"]:
        if pair_type == "C_preserve":
            has_split = True

    if has_split:
        _, y_changed, y_fixed = split_accuracy_from_correct(y_ok, bank, "C")
        _, ap_changed, ap_fixed = split_accuracy_from_correct(ap_ok, bank, "C")
        _, at_changed, at_fixed = split_accuracy_from_correct(at_ok, bank, "C")
        _, joint_at_y_changed, joint_at_y_fixed = split_accuracy_from_correct(at_ok & y_ok, bank, "C")
        _, joint_all_changed, joint_all_fixed = split_accuracy_from_correct(ap_ok & at_ok & y_ok, bank, "C")
        _, at_ap_changed, at_ap_fixed = split_accuracy_from_correct(at_ap_ok, bank, "C")
        _, y_ap_changed, y_ap_fixed = split_accuracy_from_correct(y_ap_ok, bank, "C")
        _, joint_replay_changed, joint_replay_fixed = split_accuracy_from_correct(at_ap_ok & y_ap_ok, bank, "C")
        _, at_replay_changed, at_replay_fixed = split_accuracy_from_correct(at_replay_ok, bank, "C")

        result["Y_after_C_changed"] = y_changed
        result["Y_after_C_fixed"] = y_fixed
        result["AP_after_C_changed"] = ap_changed
        result["AP_after_C_fixed"] = ap_fixed
        result["AT_after_C_changed"] = at_changed
        result["AT_after_C_fixed"] = at_fixed
        result["Joint_AT_Y_after_C_changed"] = joint_at_y_changed
        result["Joint_AT_Y_after_C_fixed"] = joint_at_y_fixed
        result["Joint_AP_AT_Y_after_C_changed"] = joint_all_changed
        result["Joint_AP_AT_Y_after_C_fixed"] = joint_all_fixed
        result["AT_via_AP_replay_changed"] = at_ap_changed
        result["AT_via_AP_replay_fixed"] = at_ap_fixed
        result["Y_via_AP_replay_changed"] = y_ap_changed
        result["Y_via_AP_replay_fixed"] = y_ap_fixed
        result["Joint_AT_Y_via_AP_replay_changed"] = joint_replay_changed
        result["Joint_AT_Y_via_AP_replay_fixed"] = joint_replay_fixed
        result["Y_via_AT_replay_changed"] = at_replay_changed
        result["Y_via_AT_replay_fixed"] = at_replay_fixed

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_gradual2.01.0_newrun_v2")
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--change-ratio", type=float, default=0.5)
    args = parser.parse_args()
    set_seed(args.seed)

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    result = args.result_dir

    Q_at, L_at, pos_at = load_ckpt(f"{result}/das_at.pt", device)
    Q_ap, L_ap, pos_ap = load_ckpt(f"{result}/das_ap.pt", device)
    Q_c, L_c, pos_c = load_ckpt(f"{result}/das_c.pt", device)

    W_at, b_at, _ = load_readout(f"{result}/at_readout_LogisticRegression.pt", "AT", device)
    W_ap, b_ap, _ = load_readout(f"{result}/ap_readout_LogisticRegression.pt", "AP", device)
    W_c, b_c, c_info = load_readout(f"{result}/c_readout_LogisticRegression.pt", "C", device)

    ap_only = build_balanced_eval_bank(model, tokenizer, args.data_path, "test", "AP", args.test_size, args.batch_size, args.seed + 1, args.change_ratio)
    at = build_balanced_eval_bank(model, tokenizer, args.data_path, "test", "AT", args.test_size, args.batch_size, args.seed + 2, args.change_ratio)
    c_only = build_balanced_eval_bank(model, tokenizer, args.data_path, "test", "C", args.test_size, args.batch_size, args.seed + 3, args.change_ratio)
    c_joint = build_pair_bank(model, tokenizer, args.data_path, "test", "C_AP_AT", args.test_size, args.batch_size, args.seed + 4)
    ap_at = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_AT", args.test_size, args.batch_size, args.seed + 5)

    ap_base, ap_base_changed, ap_base_fixed = readout_accuracy_split(model, ap_only, Q_ap, L_ap, pos_ap, W_ap, b_ap, ap_only["base_answer_pointer_ids"], args.batch_size, "base", "AP")
    ap_source, ap_source_changed, ap_source_fixed = readout_accuracy_split(model, ap_only, Q_ap, L_ap, pos_ap, W_ap, b_ap, ap_only["source_answer_pointer_ids"], args.batch_size, "source", "AP")
    at_base, at_base_changed, at_base_fixed = readout_accuracy_split(model, at, Q_at, L_at, pos_at, W_at, b_at, at["base_answer_label_ids"], args.batch_size, "base", "AT")
    at_source, at_source_changed, at_source_fixed = readout_accuracy_split(model, at, Q_at, L_at, pos_at, W_at, b_at, at["source_answer_label_ids"], args.batch_size, "source", "AT")

    colors = c_info["color_to_class"]
    c_base, c_base_changed, c_base_fixed = readout_accuracy_split(model, c_only, Q_c, L_c, pos_c, W_c, b_c, color_labels(c_only, colors, "base"), args.batch_size, "base", "C")
    c_source, c_source_changed, c_source_fixed = readout_accuracy_split(model, c_only, Q_c, L_c, pos_c, W_c, b_c, color_labels(c_only, colors, "source"), args.batch_size, "source", "C")

    ap_results = {}
    for name, bank in (("AP_only", ap_only), ("AP_AT", ap_at)):
        ap_results[name] = eval_ap(model, tokenizer, bank, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, W_at, b_at, args.batch_size, args.output_space)

    c_results = {}
    for name, bank in (("C_only", c_only), ("C_AP_AT", c_joint)):
        c_results[name] = eval_c(model, tokenizer, bank, Q_c, L_c, pos_c, Q_ap, L_ap, pos_ap, W_ap, b_ap, Q_at, L_at, pos_at, W_at, b_at, args.batch_size, args.output_space)

    lines = []
    lines.append("=== CLEAN READOUT ===")
    lines.append(f"AP overall base/source: {ap_base:.4f} / {ap_source:.4f}")
    lines.append(f"AP changed base/source: {ap_base_changed:.4f} / {ap_source_changed:.4f}")
    lines.append(f"AP fixed   base/source: {ap_base_fixed:.4f} / {ap_source_fixed:.4f}")
    lines.append(f"AT overall base/source: {at_base:.4f} / {at_source:.4f}")
    lines.append(f"AT changed base/source: {at_base_changed:.4f} / {at_source_changed:.4f}")
    lines.append(f"AT fixed   base/source: {at_base_fixed:.4f} / {at_source_fixed:.4f}")
    lines.append(f"C overall  base/source: {c_base:.4f} / {c_source:.4f}")
    lines.append(f"C changed  base/source: {c_base_changed:.4f} / {c_source_changed:.4f}")
    lines.append(f"C fixed    base/source: {c_base_fixed:.4f} / {c_source_fixed:.4f}")

    for name in ("AP_only", "AP_AT"):
        metrics = ap_results[name]
        lines.append(f"\n=== AP -> AT -> Y [{name}] ===")
        lines.append(f"Y_after_AP: {metrics['Y_after_AP']:.4f} | AT_after_AP: {metrics['AT_after_AP']:.4f} | Joint_AT_Y: {metrics['Joint_AT_Y_after_AP']:.4f}")
        if name == "AP_only":
            lines.append(f"  changed: Y={metrics['Y_after_AP_changed']:.4f} | AT={metrics['AT_after_AP_changed']:.4f} | Joint={metrics['Joint_AT_Y_after_AP_changed']:.4f} | replay={metrics['Y_via_AT_replay_changed']:.4f}")
            lines.append(f"  fixed:   Y={metrics['Y_after_AP_fixed']:.4f} | AT={metrics['AT_after_AP_fixed']:.4f} | Joint={metrics['Joint_AT_Y_after_AP_fixed']:.4f} | replay={metrics['Y_via_AT_replay_fixed']:.4f}")
        lines.append(f"P(Y correct | AT correct): {metrics['Y_correct_given_AT_correct_after_AP']:.4f} | P(AT correct | Y correct): {metrics['AT_correct_given_Y_correct_after_AP']:.4f}")
        lines.append(f"Y_via_AT_replay: {metrics['Y_via_AT_replay']:.4f}")

    for name in ("C_only", "C_AP_AT"):
        metrics = c_results[name]
        lines.append(f"\n=== C -> AP -> AT -> Y [{name}] ===")
        lines.append(f"Y_after_C: {metrics['Y_after_C']:.4f} | AP_after_C: {metrics['AP_after_C']:.4f} | AT_after_C: {metrics['AT_after_C']:.4f} | Joint_AT_Y: {metrics['Joint_AT_Y_after_C']:.4f} | Joint_AP_AT_Y: {metrics['Joint_AP_AT_Y_after_C']:.4f}")
        if name == "C_only":
            lines.append(f"  changed: Y={metrics['Y_after_C_changed']:.4f} | AP={metrics['AP_after_C_changed']:.4f} | AT={metrics['AT_after_C_changed']:.4f} | Joint_AT_Y={metrics['Joint_AT_Y_after_C_changed']:.4f} | Joint_AP_AT_Y={metrics['Joint_AP_AT_Y_after_C_changed']:.4f}")
            lines.append(f"  fixed:   Y={metrics['Y_after_C_fixed']:.4f} | AP={metrics['AP_after_C_fixed']:.4f} | AT={metrics['AT_after_C_fixed']:.4f} | Joint_AT_Y={metrics['Joint_AT_Y_after_C_fixed']:.4f} | Joint_AP_AT_Y={metrics['Joint_AP_AT_Y_after_C_fixed']:.4f}")
            lines.append(f"  changed replay: AT_via_AP={metrics['AT_via_AP_replay_changed']:.4f} | Y_via_AP={metrics['Y_via_AP_replay_changed']:.4f} | Joint={metrics['Joint_AT_Y_via_AP_replay_changed']:.4f} | Y_via_AT={metrics['Y_via_AT_replay_changed']:.4f}")
            lines.append(f"  fixed replay:   AT_via_AP={metrics['AT_via_AP_replay_fixed']:.4f} | Y_via_AP={metrics['Y_via_AP_replay_fixed']:.4f} | Joint={metrics['Joint_AT_Y_via_AP_replay_fixed']:.4f} | Y_via_AT={metrics['Y_via_AT_replay_fixed']:.4f}")
        lines.append(f"P(Y correct | AP correct): {metrics['Y_correct_given_AP_correct_after_C']:.4f} | P(AP correct | Y correct): {metrics['AP_correct_given_Y_correct_after_C']:.4f}")
        lines.append(f"P(Y correct | AT correct): {metrics['Y_correct_given_AT_correct_after_C']:.4f} | P(AT correct | Y correct): {metrics['AT_correct_given_Y_correct_after_C']:.4f}")
        lines.append(f"AT_via_AP_replay: {metrics['AT_via_AP_replay']:.4f} | Y_via_AP_replay: {metrics['Y_via_AP_replay']:.4f} | Joint_AT_Y_via_AP_replay: {metrics['Joint_AT_Y_via_AP_replay']:.4f} | Y_via_AT_replay: {metrics['Y_via_AT_replay']:.4f}")

    text = "\n".join(lines)
    print("\n" + text)

    output_path = Path(result) / "eval_results.txt"
    output_path.write_text(text + "\n")
    print(f"\nsaved results = {output_path}")


if __name__ == "__main__":
    main()
