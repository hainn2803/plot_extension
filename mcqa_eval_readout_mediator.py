"""Evaluate clean readouts plus AP/C causal propagation and mediator replay."""

import argparse, torch
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import answer_label_ids, collect_layer_states, output_iia, output_label_ids, set_seed
from mcqa_gradual_das_ap import run_ap_intervention_and_capture_at, run_frozen_at_mediator
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import build_pair_bank


def load_ckpt(path, device):
    x = torch.load(path, map_location="cpu", weights_only=False); return x["basis"].float().to(device), int(x["layer"]), x.get("token_position", "last_token")


def load_readout(path, name, device):
    x = torch.load(path, map_location="cpu", weights_only=False); return x[f"W_{name}"].float().to(device), x[f"b_{name}"].float().to(device), x


def accuracy(logits, target): return float((logits.argmax(-1).cpu() == target.cpu()).float().mean())


@torch.no_grad()
def readout_accuracy(model, bank, Q, layer, pos, W, b, target, batch_size, side):
    h = collect_layer_states(model, bank[f"{side}_input_ids"], bank[f"{side}_attention_mask"], bank[f"{side}_position_by_id"], layer, pos, batch_size)
    return accuracy((h.to(Q.device) @ Q) @ W.T + b, target)


def color_labels(bank, color_to_class, side):
    return torch.tensor([color_to_class[str(p[f"{side}_color"]).strip().casefold()] for p in bank["pairs"]])


@torch.no_grad()
def eval_ap(model, tokenizer, bank, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    source_ap = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], L_ap, pos_ap, batch_size)
    direct_y, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, L_ap, L_at, output_ids, pos_ap, batch_size, False)
    replay_y = run_frozen_at_mediator(model, bank, generated_at, Q_at, L_at, output_ids, pos_at, batch_size, False)
    t = bank["intervention_targets"]; at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    return output_iia(direct_y, t["Y_after_AP_intervene"], tokenizer, output_space), accuracy(at_logits, t["AT_after_AP_intervene"]), output_iia(replay_y, t["Y_after_AP_intervene"], tokenizer, output_space)


@torch.no_grad()
def eval_c(model, tokenizer, bank, Q_c, L_c, pos_c, Q_ap, L_ap, pos_ap, W_ap, b_ap, Q_at, L_at, pos_at, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], L_c, pos_c, batch_size)
    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, L_c, L_ap, L_at, output_ids, pos_c, pos_ap, pos_at, batch_size, False)
    ap_replay_y = run_frozen_mediator(model, bank, generated_ap, Q_ap, L_ap, output_ids, pos_ap, batch_size, False)
    at_replay_y = run_frozen_mediator(model, bank, generated_at, Q_at, L_at, output_ids, pos_at, batch_size, False)
    t = bank["intervention_targets"]; ap_logits = (generated_ap.to(Q_ap.device) @ Q_ap) @ W_ap.T + b_ap; at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    return {
        "AP_after_C": accuracy(ap_logits, t["AP_after_C_intervene"]),
        "AT_after_C": accuracy(at_logits, t["AT_after_C_intervene"]),
        "Y_after_C": output_iia(direct_y, t["Y_after_C_intervene"], tokenizer, output_space),
        "Y_via_AP_replay": output_iia(ap_replay_y, t["Y_after_C_intervene"], tokenizer, output_space),
        "Y_via_AT_replay": output_iia(at_replay_y, t["Y_after_C_intervene"], tokenizer, output_space),
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-path", default="randomized_copycolors"); p.add_argument("--result-dir", default="results_joint_randomized_gradual2.0"); p.add_argument("--test-size", type=int, default=400); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--seed", type=int, default=0); p.add_argument("--output-space", choices=("full", "az"), default="full"); a = p.parse_args(); set_seed(a.seed)
    model, tokenizer = load_gemma_model(); device = next(model.parameters()).device; result = a.result_dir
    Q_at, L_at, pos_at = load_ckpt(f"{result}/das_at.pt", device); Q_ap, L_ap, pos_ap = load_ckpt(f"{result}/das_ap.pt", device); Q_c, L_c, pos_c = load_ckpt(f"{result}/das_c.pt", device)
    W_at, b_at, _ = load_readout(f"{result}/at_readout_LogisticRegression.pt", "AT", device); W_ap, b_ap, _ = load_readout(f"{result}/ap_readout_LogisticRegression.pt", "AP", device); W_c, b_c, c_info = load_readout(f"{result}/c_readout_LogisticRegression.pt", "C", device)

    ap = build_pair_bank(model, tokenizer, a.data_path, "test", "AP_only", a.test_size, a.batch_size, a.seed + 1)
    at = build_pair_bank(model, tokenizer, a.data_path, "test", "AT_only", a.test_size, a.batch_size, a.seed + 2)
    c_only = build_pair_bank(model, tokenizer, a.data_path, "test", "C_only", a.test_size, a.batch_size, a.seed + 3)
    c_joint = build_pair_bank(model, tokenizer, a.data_path, "test", "C_AP_AT", a.test_size, a.batch_size, a.seed + 4)

    # DEBUG AT SOURCE STATE
    h = collect_layer_states(
        model,
        at["source_input_ids"],
        at["source_attention_mask"],
        at["source_position_by_id"],
        L_at,
        pos_at,
        a.batch_size,
    )

    logits = (h.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    pred = logits.argmax(-1).cpu()

    base_at = at["base_answer_label_ids"]
    source_at = at["source_answer_label_ids"]

    print("source state -> source AT:",
        (pred == source_at).float().mean().item())

    print("source state -> base AT:",
        (pred == base_at).float().mean().item())

    print("\n=== CLEAN READOUT ===")
    print(f"AP base/source: {readout_accuracy(model, ap, Q_ap, L_ap, pos_ap, W_ap, b_ap, ap['base_answer_pointer_ids'], a.batch_size, 'base'):.4f} / {readout_accuracy(model, ap, Q_ap, L_ap, pos_ap, W_ap, b_ap, ap['source_answer_pointer_ids'], a.batch_size, 'source'):.4f}")
    print(f"AT base/source: {readout_accuracy(model, at, Q_at, L_at, pos_at, W_at, b_at, at['base_answer_label_ids'], a.batch_size, 'base'):.4f} / {readout_accuracy(model, at, Q_at, L_at, pos_at, W_at, b_at, at['source_answer_label_ids'], a.batch_size, 'source'):.4f}")
    colors = c_info["color_to_class"]; print(f"C base/source : {readout_accuracy(model, c_only, Q_c, L_c, pos_c, W_c, b_c, color_labels(c_only, colors, 'base'), a.batch_size, 'base'):.4f} / {readout_accuracy(model, c_only, Q_c, L_c, pos_c, W_c, b_c, color_labels(c_only, colors, 'source'), a.batch_size, 'source'):.4f}")

    direct, at_ro, med = eval_ap(model, tokenizer, ap, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, W_at, b_at, a.batch_size, a.output_space)
    print("\n=== AP -> AT -> Y ==="); print(f"Y after AP: {direct:.4f} | AT readout: {at_ro:.4f} | Y via AT replay: {med:.4f}")

    for name, bank in (("C_only", c_only), ("C_AP_AT", c_joint)):
        m = eval_c(model, tokenizer, bank, Q_c, L_c, pos_c, Q_ap, L_ap, pos_ap, W_ap, b_ap, Q_at, L_at, pos_at, W_at, b_at, a.batch_size, a.output_space)
        print(f"\n=== C -> AP -> AT -> Y [{name}] ==="); print(" | ".join(f"{k}: {v:.4f}" for k, v in m.items()))


if __name__ == "__main__": 
    main()
