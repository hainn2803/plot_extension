"""Evaluate clean readouts plus AP/C causal propagation and mediator replay."""

import argparse

import torch

from mcqa_gradual_das import answer_label_ids, answer_logits, collect_joint_token_states_in_single_layer, collect_layer_states, output_iia, output_label_ids, set_seed
from mcqa_gradual_das_ap_cross_tokens import run_ap_intervention_and_capture_at, run_frozen_at_mediator
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import build_pair_bank
from mcqa_neural_net import load_gemma_model


def load_ckpt(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    basis = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])
    token_position = checkpoint["token_position"]
    return basis, layer, token_position


def load_readout(path, name, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint[f"W_{name}"].float().to(device)
    bias = checkpoint[f"b_{name}"].float().to(device)
    return weights, bias, checkpoint


def accuracy(logits, target):
    prediction = logits.argmax(dim=-1).cpu()
    return float((prediction == target.cpu()).float().mean())


def output_correct(logits, target, tokenizer, output_space):
    prediction = logits.argmax(dim=-1).cpu()

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
def readout_accuracy(model, bank, Q, layer, token_position, W, b, target, batch_size, side):
    input_ids = bank[f"{side}_input_ids"]
    attention_mask = bank[f"{side}_attention_mask"]
    position_by_id = bank[f"{side}_position_by_id"]

    if isinstance(token_position, list):
        states = collect_joint_token_states_in_single_layer(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)
        states = states.reshape(states.shape[0], -1)
    else:
        states = collect_layer_states(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)

    logits = (states.to(Q.device) @ Q) @ W.T + b

    return accuracy(logits, target)


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

        for ap_token_position in list_ap_token_positions:
            ap_position = pad_offset + bank["base_position_by_id"][ap_token_position][start:end].to(device)
            ap_positions.append(ap_position)

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

            base_ap_flat = base_ap.reshape(num_examples, -1)
            donor_ap_flat = donor_ap.reshape(num_examples, -1).float()

            delta_flat = ((donor_ap_flat - base_ap_flat) @ Q_ap) @ Q_ap.T
            new_ap_flat = base_ap_flat + delta_flat

            new_ap = new_ap_flat.reshape(num_examples, num_ap_tokens, -1)

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

    targets = bank["intervention_targets"]

    at_target = targets["AT_after_AP_intervene"]
    y_target = targets["Y_after_AP_intervene"]

    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    at_ok = at_logits.argmax(dim=-1).cpu() == at_target.cpu()
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

    result = {
        "Y_after_AP": output_iia(direct_y, y_target, tokenizer, output_space),
        "AT_after_AP": accuracy(at_logits, at_target),
        "Joint_AT_Y_after_AP": float((at_ok & y_ok).float().mean()),
        "Y_correct_given_AT_correct_after_AP": y_given_at,
        "AT_correct_given_Y_correct_after_AP": at_given_y,
        "Y_via_AT_replay": output_iia(replay_y, y_target, tokenizer, output_space),
    }

    return result


@torch.no_grad()
def eval_c(model, tokenizer, bank, Q_c, c_layer, c_token_position, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], c_layer, c_token_position, batch_size)

    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_token_position, list_ap_token_positions, at_token_position, batch_size, False)

    ap_replay_y, at_after_ap = replay_ap_capture_at(model, bank, generated_ap, Q_ap, ap_layer, list_ap_token_positions, at_layer, at_token_position, output_ids, batch_size)

    list_at_token_positions = [at_token_position]
    at_replay_y = run_frozen_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, list_at_token_positions, batch_size, False)

    targets = bank["intervention_targets"]

    ap_target = targets["AP_after_C_intervene"]
    at_target = targets["AT_after_C_intervene"]
    y_target = targets["Y_after_C_intervene"]

    generated_ap_flat = generated_ap.reshape(generated_ap.shape[0], -1)

    ap_logits = (generated_ap_flat.to(Q_ap.device) @ Q_ap) @ W_ap.T + b_ap
    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    at_after_ap_logits = (at_after_ap.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    ap_ok = ap_logits.argmax(dim=-1).cpu() == ap_target.cpu()
    at_ok = at_logits.argmax(dim=-1).cpu() == at_target.cpu()
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

    at_after_ap_ok = at_after_ap_logits.argmax(dim=-1).cpu() == at_target.cpu()
    y_after_ap_ok = output_correct(ap_replay_y, y_target, tokenizer, output_space)

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
        "Joint_AT_Y_via_AP_replay": float((at_after_ap_ok & y_after_ap_ok).float().mean()),
        "Y_via_AT_replay": output_iia(at_replay_y, y_target, tokenizer, output_space),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_standard_ap_cross_tokens_128")
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")

    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    result_dir = args.result_dir

    at_checkpoint_path = f"{result_dir}/das_at.pt"
    ap_checkpoint_path = f"{result_dir}/das_ap.pt"
    c_checkpoint_path = f"{result_dir}/das_c.pt"

    at_readout_path = f"{result_dir}/at_readout_LogisticRegression.pt"
    ap_readout_path = f"{result_dir}/ap_readout_LogisticRegression.pt"
    c_readout_path = f"{result_dir}/c_readout_LogisticRegression.pt"

    Q_at, at_layer, at_token_position = load_ckpt(at_checkpoint_path, device)
    Q_ap, ap_layer, list_ap_token_positions = load_ckpt(ap_checkpoint_path, device)
    Q_c, c_layer, c_token_position = load_ckpt(c_checkpoint_path, device)

    if not isinstance(list_ap_token_positions, list):
        raise ValueError(f"AP token_position must be a list, got {type(list_ap_token_positions)}")

    expected_ap_width = len(list_ap_token_positions) * model.config.hidden_size

    if Q_ap.shape[0] != expected_ap_width:
        raise ValueError(f"AP basis width {Q_ap.shape[0]} does not match {len(list_ap_token_positions)} tokens x hidden size {model.config.hidden_size} = {expected_ap_width}")

    if Q_at.shape[0] != model.config.hidden_size:
        raise ValueError(f"AT basis width {Q_at.shape[0]} does not match hidden size {model.config.hidden_size}")

    W_at, b_at, _ = load_readout(at_readout_path, "AT", device)
    W_ap, b_ap, _ = load_readout(ap_readout_path, "AP", device)
    W_c, b_c, c_info = load_readout(c_readout_path, "C", device)

    ap_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_only", args.test_size, args.batch_size, args.seed + 1)
    at_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AT_only", args.test_size, args.batch_size, args.seed + 2)
    c_only = build_pair_bank(model, tokenizer, args.data_path, "test", "C_only", args.test_size, args.batch_size, args.seed + 3)
    c_joint = build_pair_bank(model, tokenizer, args.data_path, "test", "C_AP_AT", args.test_size, args.batch_size, args.seed + 4)
    ap_at = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_AT", args.test_size, args.batch_size, args.seed + 5)

    ap_base = readout_accuracy(model, ap_only, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, ap_only["base_answer_pointer_ids"], args.batch_size, "base")
    ap_source = readout_accuracy(model, ap_only, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, ap_only["source_answer_pointer_ids"], args.batch_size, "source")

    at_base = readout_accuracy(model, at_only, Q_at, at_layer, at_token_position, W_at, b_at, at_only["base_answer_label_ids"], args.batch_size, "base")
    at_source = readout_accuracy(model, at_only, Q_at, at_layer, at_token_position, W_at, b_at, at_only["source_answer_label_ids"], args.batch_size, "source")

    colors = c_info["color_to_class"]

    c_base_target = color_labels(c_only, colors, "base")
    c_source_target = color_labels(c_only, colors, "source")

    c_base = readout_accuracy(model, c_only, Q_c, c_layer, c_token_position, W_c, b_c, c_base_target, args.batch_size, "base")
    c_source = readout_accuracy(model, c_only, Q_c, c_layer, c_token_position, W_c, b_c, c_source_target, args.batch_size, "source")

    ap_results = {}

    for name, bank in (("AP_only", ap_only), ("AP_AT", ap_at)):
        ap_results[name] = eval_ap(model, tokenizer, bank, Q_ap, ap_layer, list_ap_token_positions, Q_at, at_layer, at_token_position, W_at, b_at, args.batch_size, args.output_space)

    c_results = {}

    for name, bank in (("C_only", c_only), ("C_AP_AT", c_joint)):
        c_results[name] = eval_c(model, tokenizer, bank, Q_c, c_layer, c_token_position, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, args.batch_size, args.output_space)

    lines = []

    lines.append("=== CLEAN READOUT ===")
    lines.append(f"AP base/source: {ap_base:.4f} / {ap_source:.4f}")
    lines.append(f"AT base/source: {at_base:.4f} / {at_source:.4f}")
    lines.append(f"C base/source : {c_base:.4f} / {c_source:.4f}")

    for name in ("AP_only", "AP_AT"):
        metrics = ap_results[name]

        lines.append(f"\n=== AP -> AT -> Y [{name}] ===")
        lines.append(f"Y_after_AP: {metrics['Y_after_AP']:.4f} | AT_after_AP: {metrics['AT_after_AP']:.4f} | Joint_AT_Y: {metrics['Joint_AT_Y_after_AP']:.4f}")
        lines.append(f"P(Y correct | AT correct): {metrics['Y_correct_given_AT_correct_after_AP']:.4f}")
        lines.append(f"P(AT correct | Y correct): {metrics['AT_correct_given_Y_correct_after_AP']:.4f}")
        lines.append(f"Y_via_AT_replay: {metrics['Y_via_AT_replay']:.4f}")

    for name in ("C_only", "C_AP_AT"):
        metrics = c_results[name]

        lines.append(f"\n=== C -> AP -> AT -> Y [{name}] ===")
        lines.append(f"Y_after_C: {metrics['Y_after_C']:.4f}")
        lines.append(f"AP_after_C: {metrics['AP_after_C']:.4f}")
        lines.append(f"AT_after_C: {metrics['AT_after_C']:.4f}")
        lines.append(f"Joint_AT_Y_after_C: {metrics['Joint_AT_Y_after_C']:.4f}")
        lines.append(f"Joint_AP_AT_Y_after_C: {metrics['Joint_AP_AT_Y_after_C']:.4f}")
        lines.append(f"P(Y correct | AP correct): {metrics['Y_correct_given_AP_correct_after_C']:.4f}")
        lines.append(f"P(AP correct | Y correct): {metrics['AP_correct_given_Y_correct_after_C']:.4f}")
        lines.append(f"P(Y correct | AT correct): {metrics['Y_correct_given_AT_correct_after_C']:.4f}")
        lines.append(f"P(AT correct | Y correct): {metrics['AT_correct_given_Y_correct_after_C']:.4f}")
        lines.append(f"AT_via_AP_replay: {metrics['AT_via_AP_replay']:.4f}")
        lines.append(f"Y_via_AP_replay: {metrics['Y_via_AP_replay']:.4f}")
        lines.append(f"Joint_AT_Y_via_AP_replay: {metrics['Joint_AT_Y_via_AP_replay']:.4f}")
        lines.append(f"Y_via_AT_replay: {metrics['Y_via_AT_replay']:.4f}")

    text = "\n".join(lines)

    print(text)

    output_path = f"{result_dir}/eval_readout_mediator_new.txt"

    with open(output_path, "w") as file:
        file.write(text)
        file.write("\n")


if __name__ == "__main__":
    main()