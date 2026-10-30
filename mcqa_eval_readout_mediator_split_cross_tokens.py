"""Split evaluation for gradual-das-mcqa with cross-token AP and shared AT checkpoints."""

import argparse
from pathlib import Path
import torch

from mcqa_gradual_das import answer_label_ids, answer_logits, collect_joint_token_states_in_single_layer, collect_layer_states, output_label_ids, set_seed
from mcqa_gradual_das_ap_cross_tokens import run_ap_intervention_and_capture_at, run_frozen_at_mediator
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c_balanced_eval import build_balanced_eval_bank
from mcqa_neural_net import load_gemma_model


def load_ckpt(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    Q = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])
    token_position = checkpoint["token_position"]
    return Q, layer, token_position


def load_readout(path, name, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    W = checkpoint[f"W_{name}"].float().to(device)
    b = checkpoint[f"b_{name}"].float().to(device)
    return W, b, checkpoint


def output_correct(logits, target, tokenizer, output_space):
    prediction = logits.argmax(dim=-1).cpu()

    if output_space == "az":
        return prediction == target.cpu()

    correct = []

    for predicted_id, target_id in zip(prediction, target):
        predicted_text = tokenizer.decode([int(predicted_id)]).strip().upper()
        expected_text = chr(ord("A") + int(target_id))
        correct.append(predicted_text == expected_text)

    return torch.tensor(correct, dtype=torch.bool)


def split_masks(bank, variable):
    changed_type = f"{variable}_only"
    fixed_type = f"{variable}_preserve"

    changed = []
    fixed = []

    for pair_type in bank["pair_types"]:
        changed.append(pair_type == changed_type)
        fixed.append(pair_type == fixed_type)

    return torch.tensor(changed, dtype=torch.bool), torch.tensor(fixed, dtype=torch.bool)


def split_accuracy(correct, bank, variable):
    correct = correct.cpu().bool()
    changed_mask, fixed_mask = split_masks(bank, variable)

    overall = float(correct.float().mean())

    if changed_mask.sum().item() > 0:
        changed = float(correct[changed_mask].float().mean())
    else:
        changed = float("nan")

    if fixed_mask.sum().item() > 0:
        fixed = float(correct[fixed_mask].float().mean())
    else:
        fixed = float("nan")

    return overall, changed, fixed


@torch.no_grad()
def readout_correct(model, bank, Q, layer, token_position, W, b, target, batch_size, side):
    input_ids = bank[f"{side}_input_ids"]
    attention_mask = bank[f"{side}_attention_mask"]
    position_by_id = bank[f"{side}_position_by_id"]

    if isinstance(token_position, list):
        states = collect_joint_token_states_in_single_layer(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)
        states = states.reshape(states.shape[0], -1)
    else:
        states = collect_layer_states(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)

    logits = (states.to(Q.device) @ Q) @ W.T + b
    return logits.argmax(dim=-1).cpu() == target.cpu()


def color_labels(bank, color_to_class, side):
    labels = []

    for pair in bank["pairs"]:
        color = str(pair[f"{side}_color"]).strip().casefold()
        labels.append(color_to_class[color])

    return torch.tensor(labels)


@torch.no_grad()
def replay_ap_capture_at(model, bank, generated_ap, Q_ap, ap_layer, list_ap_token_positions, at_layer, at_token_position, output_ids, batch_size):
    device = next(model.parameters()).device
    all_y = []
    all_at = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor_ap = generated_ap[start:end].to(device)

        num_examples = ids.shape[0]
        num_ap_tokens = len(list_ap_token_positions)
        rows = torch.arange(num_examples, device=device)
        pad_offset = (mask == 0).sum(dim=1)

        ap_positions = []

        for token_position in list_ap_token_positions:
            position = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)
            ap_positions.append(position)

        ap_positions = torch.stack(ap_positions, dim=1)
        at_position = pad_offset + bank["base_position_by_id"][at_token_position][start:end].to(device)
        captured = {}

        def ap_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base_ap = hidden[rows[:, None], ap_positions, :].float()
            base_flat = base_ap.reshape(num_examples, -1)
            donor_flat = donor_ap.reshape(num_examples, -1).float()
            delta_flat = ((donor_flat - base_flat) @ Q_ap) @ Q_ap.T
            new_ap = (base_flat + delta_flat).reshape(num_examples, num_ap_tokens, -1)
            new_hidden[rows[:, None], ap_positions, :] = new_ap.to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        def at_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            captured["at"] = hidden[rows, at_position, :].float()

        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        at_handle = model.model.layers[at_layer].register_forward_hook(at_hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            y = answer_logits(model, outputs, mask, output_ids)
            all_y.append(y.detach().cpu())
            all_at.append(captured["at"].detach().cpu())
        finally:
            ap_handle.remove()
            at_handle.remove()

    return torch.cat(all_y), torch.cat(all_at)


@torch.no_grad()
def eval_ap(model, tokenizer, bank, Q_ap, ap_layer, list_ap_token_positions, Q_at, at_layer, at_token_position, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    source_ap = collect_joint_token_states_in_single_layer(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], ap_layer, list_ap_token_positions, batch_size)

    direct_y, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, ap_layer, at_layer, output_ids, list_ap_token_positions, at_token_position, batch_size, False)
    replay_y = run_frozen_at_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, at_token_position, batch_size, False)

    at_target = bank["intervention_targets"]["AT_after_AP_intervene"]
    y_target = bank["intervention_targets"]["Y_after_AP_intervene"]

    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    at_ok = at_logits.argmax(dim=-1).cpu() == at_target.cpu()
    y_ok = output_correct(direct_y, y_target, tokenizer, output_space)
    replay_ok = output_correct(replay_y, y_target, tokenizer, output_space)

    return {
        "Y_after_AP": split_accuracy(y_ok, bank, "AP"),
        "AT_after_AP": split_accuracy(at_ok, bank, "AP"),
        "Joint_AT_Y_after_AP": split_accuracy(at_ok & y_ok, bank, "AP"),
        "Y_via_AT_replay": split_accuracy(replay_ok, bank, "AP"),
    }


@torch.no_grad()
def eval_c(model, tokenizer, bank, Q_c, c_layer, c_token_position, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], c_layer, c_token_position, batch_size)

    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_token_position, list_ap_token_positions, at_token_position, batch_size, False)

    ap_replay_y, at_after_ap = replay_ap_capture_at(model, bank, generated_ap, Q_ap, ap_layer, list_ap_token_positions, at_layer, at_token_position, output_ids, batch_size)

    at_replay_y = run_frozen_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, [at_token_position], batch_size, False)

    ap_target = bank["intervention_targets"]["AP_after_C_intervene"]
    at_target = bank["intervention_targets"]["AT_after_C_intervene"]
    y_target = bank["intervention_targets"]["Y_after_C_intervene"]

    generated_ap_flat = generated_ap.reshape(generated_ap.shape[0], -1)
    ap_logits = (generated_ap_flat.to(Q_ap.device) @ Q_ap) @ W_ap.T + b_ap
    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    at_after_ap_logits = (at_after_ap.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    ap_ok = ap_logits.argmax(dim=-1).cpu() == ap_target.cpu()
    at_ok = at_logits.argmax(dim=-1).cpu() == at_target.cpu()
    y_ok = output_correct(direct_y, y_target, tokenizer, output_space)

    at_after_ap_ok = at_after_ap_logits.argmax(dim=-1).cpu() == at_target.cpu()
    y_after_ap_ok = output_correct(ap_replay_y, y_target, tokenizer, output_space)
    at_replay_ok = output_correct(at_replay_y, y_target, tokenizer, output_space)

    return {
        "Y_after_C": split_accuracy(y_ok, bank, "C"),
        "AP_after_C": split_accuracy(ap_ok, bank, "C"),
        "AT_after_C": split_accuracy(at_ok, bank, "C"),
        "Joint_AT_Y_after_C": split_accuracy(at_ok & y_ok, bank, "C"),
        "Joint_AP_AT_Y_after_C": split_accuracy(ap_ok & at_ok & y_ok, bank, "C"),
        "AT_via_AP_replay": split_accuracy(at_after_ap_ok, bank, "C"),
        "Y_via_AP_replay": split_accuracy(y_after_ap_ok, bank, "C"),
        "Joint_AT_Y_via_AP_replay": split_accuracy(at_after_ap_ok & y_after_ap_ok, bank, "C"),
        "Y_via_AT_replay": split_accuracy(at_replay_ok, bank, "C"),
    }


def append_split(lines, name, values):
    overall, changed, fixed = values
    lines.append(f"{name}: {overall:.4f}")
    lines.append(f"changed: {name} = {changed:.4f}")
    lines.append(f"fixed: {name} = {fixed:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_gradual_cross_tokens_seed0_reprcac_1.01.01.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--change-ratio", type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    result_dir = Path(args.result_dir)
    seed_dir = result_dir.parent
    # shared_at_dir = seed_dir / "shared_at"
    shared_at_dir = result_dir

    at_checkpoint_path = shared_at_dir / "das_at.pt"
    at_readout_path = shared_at_dir / "at_readout_LogisticRegression.pt"
    ap_checkpoint_path = result_dir / "das_ap.pt"
    c_checkpoint_path = result_dir / "das_c.pt"
    ap_readout_path = result_dir / "ap_readout_LogisticRegression.pt"
    c_readout_path = result_dir / "c_readout_LogisticRegression.pt"

    Q_at, at_layer, at_token_position = load_ckpt(at_checkpoint_path, device)
    Q_ap, ap_layer, list_ap_token_positions = load_ckpt(ap_checkpoint_path, device)
    Q_c, c_layer, c_token_position = load_ckpt(c_checkpoint_path, device)

    if not isinstance(list_ap_token_positions, list):
        raise ValueError(f"AP token_position must be a list, got {type(list_ap_token_positions)}")

    expected_ap_width = len(list_ap_token_positions) * model.config.hidden_size

    if Q_ap.shape[0] != expected_ap_width:
        raise ValueError(f"AP basis width {Q_ap.shape[0]} != expected {expected_ap_width}")

    W_at, b_at, _ = load_readout(at_readout_path, "AT", device)
    W_ap, b_ap, _ = load_readout(ap_readout_path, "AP", device)
    W_c, b_c, c_info = load_readout(c_readout_path, "C", device)

    ap_bank = build_balanced_eval_bank(model, tokenizer, args.data_path, "test", "AP", args.test_size, args.batch_size, args.seed + 1, args.change_ratio)
    at_bank = build_balanced_eval_bank(model, tokenizer, args.data_path, "test", "AT", args.test_size, args.batch_size, args.seed + 2, args.change_ratio)
    c_bank = build_balanced_eval_bank(model, tokenizer, args.data_path, "test", "C", args.test_size, args.batch_size, args.seed + 3, args.change_ratio)

    ap_base_ok = readout_correct(model, ap_bank, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, ap_bank["base_answer_pointer_ids"], args.batch_size, "base")
    ap_source_ok = readout_correct(model, ap_bank, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, ap_bank["source_answer_pointer_ids"], args.batch_size, "source")

    at_base_ok = readout_correct(model, at_bank, Q_at, at_layer, at_token_position, W_at, b_at, at_bank["base_answer_label_ids"], args.batch_size, "base")
    at_source_ok = readout_correct(model, at_bank, Q_at, at_layer, at_token_position, W_at, b_at, at_bank["source_answer_label_ids"], args.batch_size, "source")

    colors = c_info["color_to_class"]
    c_base_target = color_labels(c_bank, colors, "base")
    c_source_target = color_labels(c_bank, colors, "source")

    c_base_ok = readout_correct(model, c_bank, Q_c, c_layer, c_token_position, W_c, b_c, c_base_target, args.batch_size, "base")
    c_source_ok = readout_correct(model, c_bank, Q_c, c_layer, c_token_position, W_c, b_c, c_source_target, args.batch_size, "source")

    ap_metrics = eval_ap(model, tokenizer, ap_bank, Q_ap, ap_layer, list_ap_token_positions, Q_at, at_layer, at_token_position, W_at, b_at, args.batch_size, args.output_space)
    c_metrics = eval_c(model, tokenizer, c_bank, Q_c, c_layer, c_token_position, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, args.batch_size, args.output_space)

    lines = []

    lines.append("=== CLEAN READOUT ===")
    lines.append(f"AP overall base/source: {float(ap_base_ok.float().mean()):.4f} / {float(ap_source_ok.float().mean()):.4f}")
    lines.append(f"AT overall base/source: {float(at_base_ok.float().mean()):.4f} / {float(at_source_ok.float().mean()):.4f}")
    lines.append(f"C overall base/source: {float(c_base_ok.float().mean()):.4f} / {float(c_source_ok.float().mean()):.4f}")

    lines.append("")
    lines.append("=== AP -> AT -> Y [AP] ===")

    for name, values in ap_metrics.items():
        append_split(lines, name, values)

    lines.append("")
    lines.append("=== C -> AP -> AT -> Y [C] ===")

    for name, values in c_metrics.items():
        append_split(lines, name, values)

    text = "\n".join(lines)

    print(text)

    output_path = result_dir / "eval_results.txt"
    output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
