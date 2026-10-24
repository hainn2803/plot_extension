"""Evaluate clean readouts plus AP/C causal propagation and mediator replay."""

import argparse
from pathlib import Path

import torch

from mcqa_gradual_das import answer_label_ids, answer_logits, collect_layer_states, output_iia, output_label_ids, set_seed
from mcqa_gradual_das_ap import run_ap_intervention_and_capture_at, run_frozen_at_mediator
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import build_pair_bank
from mcqa_neural_net import load_gemma_model


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

    return {
        "Y_after_AP": output_iia(direct_y, y_target, tokenizer, output_space),
        "AT_after_AP": accuracy(at_logits, at_target),
        "Joint_AT_Y_after_AP": float((at_ok & y_ok).float().mean()),
        "Y_via_AT_replay": output_iia(replay_y, y_target, tokenizer, output_space),
    }


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

    at_ap_ok = at_after_ap_logits.argmax(-1).cpu() == at_target.cpu()
    y_ap_ok = output_correct(ap_replay_y, y_target, tokenizer, output_space)

    return {
        "Y_after_C": output_iia(direct_y, y_target, tokenizer, output_space),
        "AP_after_C": accuracy(ap_logits, ap_target),
        "AT_after_C": accuracy(at_logits, at_target),
        "Joint_AP_AT_Y_after_C": float((ap_ok & at_ok & y_ok).float().mean()),
        "AT_via_AP_replay": accuracy(at_after_ap_logits, at_target),
        "Y_via_AP_replay": output_iia(ap_replay_y, y_target, tokenizer, output_space),
        "Joint_AT_Y_via_AP_replay": float((at_ap_ok & y_ap_ok).float().mean()),
        "Y_via_AT_replay": output_iia(at_replay_y, y_target, tokenizer, output_space),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_gradual_all_ap1.0_at4.0_newrun")
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
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

    ap_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_only", args.test_size, args.batch_size, args.seed + 1)
    at = build_pair_bank(model, tokenizer, args.data_path, "test", "AT_only", args.test_size, args.batch_size, args.seed + 2)
    c_only = build_pair_bank(model, tokenizer, args.data_path, "test", "C_only", args.test_size, args.batch_size, args.seed + 3)
    c_joint = build_pair_bank(model, tokenizer, args.data_path, "test", "C_AP_AT", args.test_size, args.batch_size, args.seed + 4)
    ap_at = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_AT", args.test_size, args.batch_size, args.seed + 5)

    ap_base = readout_accuracy(model, ap_only, Q_ap, L_ap, pos_ap, W_ap, b_ap, ap_only["base_answer_pointer_ids"], args.batch_size, "base")
    ap_source = readout_accuracy(model, ap_only, Q_ap, L_ap, pos_ap, W_ap, b_ap, ap_only["source_answer_pointer_ids"], args.batch_size, "source")
    at_base = readout_accuracy(model, at, Q_at, L_at, pos_at, W_at, b_at, at["base_answer_label_ids"], args.batch_size, "base")
    at_source = readout_accuracy(model, at, Q_at, L_at, pos_at, W_at, b_at, at["source_answer_label_ids"], args.batch_size, "source")

    colors = c_info["color_to_class"]
    c_base = readout_accuracy(model, c_only, Q_c, L_c, pos_c, W_c, b_c, color_labels(c_only, colors, "base"), args.batch_size, "base")
    c_source = readout_accuracy(model, c_only, Q_c, L_c, pos_c, W_c, b_c, color_labels(c_only, colors, "source"), args.batch_size, "source")

    ap_results = {}
    for name, bank in (("AP_only", ap_only), ("AP_AT", ap_at)):
        ap_results[name] = eval_ap(model, tokenizer, bank, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, W_at, b_at, args.batch_size, args.output_space)

    c_results = {}
    for name, bank in (("C_only", c_only), ("C_AP_AT", c_joint)):
        c_results[name] = eval_c(model, tokenizer, bank, Q_c, L_c, pos_c, Q_ap, L_ap, pos_ap, W_ap, b_ap, Q_at, L_at, pos_at, W_at, b_at, args.batch_size, args.output_space)

    lines = []
    lines.append("=== CLEAN READOUT ===")
    lines.append(f"AP base/source: {ap_base:.4f} / {ap_source:.4f}")
    lines.append(f"AT base/source: {at_base:.4f} / {at_source:.4f}")
    lines.append(f"C base/source : {c_base:.4f} / {c_source:.4f}")

    for name in ("AP_only", "AP_AT"):
        metrics = ap_results[name]
        lines.append(f"\n=== AP -> AT -> Y [{name}] ===")
        lines.append(f"Y_after_AP: {metrics['Y_after_AP']:.4f} | AT_after_AP: {metrics['AT_after_AP']:.4f} | Joint_AT_Y: {metrics['Joint_AT_Y_after_AP']:.4f}")
        lines.append(f"Y_via_AT_replay: {metrics['Y_via_AT_replay']:.4f}")

    for name in ("C_only", "C_AP_AT"):
        metrics = c_results[name]
        lines.append(f"\n=== C -> AP -> AT -> Y [{name}] ===")
        lines.append(f"Y_after_C: {metrics['Y_after_C']:.4f} | AP_after_C: {metrics['AP_after_C']:.4f} | AT_after_C: {metrics['AT_after_C']:.4f} | Joint_AP_AT_Y: {metrics['Joint_AP_AT_Y_after_C']:.4f}")
        lines.append(f"AT_via_AP_replay: {metrics['AT_via_AP_replay']:.4f} | Y_via_AP_replay: {metrics['Y_via_AP_replay']:.4f} | Joint_AT_Y_via_AP_replay: {metrics['Joint_AT_Y_via_AP_replay']:.4f} | Y_via_AT_replay: {metrics['Y_via_AT_replay']:.4f}")

    text = "\n".join(lines)
    print("\n" + text)

    output_path = Path(result) / "eval_results.txt"
    output_path.write_text(text + "\n")
    print(f"\nsaved results = {output_path}")


if __name__ == "__main__":
    main()
