"""Evaluate clean readouts and the causal chains AP -> AT -> Y and C -> AP -> AT -> Y.

C-chain semantics:
1. Intervene only on C and let the model naturally produce AP, AT, and Y.
2. Replay the generated AP into a clean run to test AP -> AT / Y mediation.
3. Replay the generated AT into a clean run to test AT -> Y mediation.
"""

import argparse, random, torch

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_c import create_bank, load_generated_dataset, predict_correct, rows_to_pairs
from mcqa_gradual_das import answer_label_ids, answer_logits, collect_layer_states, output_iia, output_label_ids, set_seed
from mcqa_gradual_das_ap import run_ap_intervention_and_capture_at, run_frozen_at_mediator



def load_ckpt(path, device):
    x = torch.load(path, map_location="cpu", weights_only=False)
    return x["basis"].to(device).float(), int(x["layer"]), x.get("token_position", "last_token")


def load_readout(path, name, device):
    x = torch.load(path, map_location="cpu", weights_only=False)
    return x[f"W_{name}"].to(device).float(), x[f"b_{name}"].to(device).float(), x


def acc(logits, target):
    return float((logits.argmax(-1).cpu() == target.cpu()).float().mean())


def hidden_from(output):
    """Transformer layer hooks may receive either a tensor or a tuple whose first item is the hidden state."""
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden_new):
    """Put an edited hidden state back into the same output structure used by the layer."""
    return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new


def patch_subspace(base, donor, Q):
    """Replace only the coordinates inside DAS subspace Q; preserve the orthogonal complement."""
    return base + ((donor - base) @ Q) @ Q.T


def positions(mask, bank, start, end, name):
    """Convert unpadded token positions stored in the bank to positions in the left-padded batch."""
    return (mask == 0).sum(1) + bank["base_position_by_id"][name][start:end].to(mask.device)


def forward_base(model, ids, mask):
    pos_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
    return model.model(input_ids=ids, attention_mask=mask, position_ids=pos_ids, use_cache=False, return_dict=True)


def build_test_bank(model, tokenizer, data_path, pair_type, size, batch_size, seed):
    ds = load_generated_dataset(data_path)
    ids = list(range(len(ds["test"])))
    random.Random(seed).shuffle(ids)
    pairs, cache = [], {}

    for s in range(0, len(ids), 64):
        candidates = rows_to_pairs(ds["test"], ids[s:s + 64], "test")
        candidates = [p for p in candidates if p["pair_type"] == pair_type]
        pairs += predict_correct(model, tokenizer, candidates, batch_size, cache)
        if len(pairs) >= size: break

    return create_bank(pairs, tokenizer, size=size, seed=seed)


@torch.no_grad()
def readout_acc(model, bank, Q, layer, pos, W, b, target, batch_size, side="source"):
    h = collect_layer_states(model, bank[f"{side}_input_ids"], bank[f"{side}_attention_mask"], bank[f"{side}_position_by_id"], layer, pos, batch_size)
    return acc((h.to(Q.device) @ Q) @ W.T + b, target)


def color_target(bank, color_to_class, side):
    return torch.tensor([color_to_class[str(p[f"{side}_color"]).strip().casefold()] for p in bank["pairs"]])


# ----------------------------- C intervention -----------------------------

@torch.no_grad()
def run_c_only_and_capture_ap_at(model, bank, source_c, Q_c, L_c, L_ap, L_at, output_ids, batch_size):
    """Intervene only on C, then capture the naturally generated AP, AT, and output Y."""
    if not L_c < L_ap <= L_at: raise ValueError(f"Expected C < AP <= AT, got C=L{L_c}, AP=L{L_ap}, AT=L{L_at}")

    device = next(model.parameters()).device
    all_ap, all_at, all_y = [], [], []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor_c = source_c[start:end].to(device=device, dtype=torch.float32)
        rows = torch.arange(len(ids), device=device)
        c_pos = positions(mask, bank, start, end, "color")
        last_pos = positions(mask, bank, start, end, "last_token")
        captured = {}

        def c_hook(_m, _i, output):
            hidden = hidden_from(output)
            edited = hidden.clone()
            base_c = hidden[rows, c_pos].float()
            edited[rows, c_pos] = patch_subspace(base_c, donor_c, Q_c).to(hidden.dtype)
            return replace_hidden(output, edited)

        def capture_ap(_m, _i, output):
            captured["ap"] = hidden_from(output)[rows, last_pos].detach().float().cpu()

        def capture_at(_m, _i, output):
            captured["at"] = hidden_from(output)[rows, last_pos].detach().float().cpu()

        def capture_both(_m, _i, output):
            state = hidden_from(output)[rows, last_pos].detach().float().cpu()
            captured["ap"] = captured["at"] = state

        handles = [model.model.layers[L_c].register_forward_hook(c_hook)]
        if L_ap == L_at:
            handles.append(model.model.layers[L_ap].register_forward_hook(capture_both))
        else:
            handles += [model.model.layers[L_ap].register_forward_hook(capture_ap), model.model.layers[L_at].register_forward_hook(capture_at)]

        try:
            outputs = forward_base(model, ids, mask)
            all_y.append(answer_logits(model, outputs, mask, output_ids).detach().cpu())
        finally:
            for h in handles: h.remove()

        if "ap" not in captured or "at" not in captured: raise RuntimeError("Failed to capture AP/AT state after C intervention")
        all_ap.append(captured["ap"]); all_at.append(captured["at"])

    return torch.cat(all_ap), torch.cat(all_at), torch.cat(all_y)


@torch.no_grad()
def replay_ap_and_capture_at(model, bank, generated_ap, Q_ap, L_ap, L_at, output_ids, batch_size):
    """Replay C-generated AP into a clean base run, then measure the resulting AT and Y."""
    if L_ap > L_at: raise ValueError(f"AP layer L{L_ap} is after AT layer L{L_at}")

    device = next(model.parameters()).device
    all_at, all_y = [], []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor_ap = generated_ap[start:end].to(device=device, dtype=torch.float32)
        rows = torch.arange(len(ids), device=device)
        last_pos = positions(mask, bank, start, end, "last_token")
        captured = {}

        def ap_hook(_m, _i, output):
            hidden = hidden_from(output)
            edited = hidden.clone()
            base_ap = hidden[rows, last_pos].float()
            edited[rows, last_pos] = patch_subspace(base_ap, donor_ap, Q_ap).to(hidden.dtype)
            if L_ap == L_at: captured["at"] = edited[rows, last_pos].detach().float().cpu()
            return replace_hidden(output, edited)

        def at_hook(_m, _i, output):
            captured["at"] = hidden_from(output)[rows, last_pos].detach().float().cpu()

        handles = [model.model.layers[L_ap].register_forward_hook(ap_hook)]
        if L_ap != L_at: handles.append(model.model.layers[L_at].register_forward_hook(at_hook))

        try:
            outputs = forward_base(model, ids, mask)
            all_y.append(answer_logits(model, outputs, mask, output_ids).detach().cpu())
        finally:
            for h in handles: h.remove()

        if "at" not in captured: raise RuntimeError("Failed to capture AT state after AP replay")
        all_at.append(captured["at"])

    return torch.cat(all_at), torch.cat(all_y)


@torch.no_grad()
def replay_at(model, bank, generated_at, Q_at, L_at, output_ids, batch_size):
    """Replay C-generated AT into a clean base run and measure Y."""
    device = next(model.parameters()).device
    all_y = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor_at = generated_at[start:end].to(device=device, dtype=torch.float32)
        rows = torch.arange(len(ids), device=device)
        last_pos = positions(mask, bank, start, end, "last_token")

        def at_hook(_m, _i, output):
            hidden = hidden_from(output)
            edited = hidden.clone()
            base_at = hidden[rows, last_pos].float()
            edited[rows, last_pos] = patch_subspace(base_at, donor_at, Q_at).to(hidden.dtype)
            return replace_hidden(output, edited)

        handle = model.model.layers[L_at].register_forward_hook(at_hook)
        try:
            outputs = forward_base(model, ids, mask)
            all_y.append(answer_logits(model, outputs, mask, output_ids).detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_y)


# ----------------------------- Evaluation -----------------------------

@torch.no_grad()
def eval_ap_mediator(model, tokenizer, bank, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, W_at, b_at, batch_size, output_space):
    """Direct AP intervention plus frozen-AT mediator test: AP -> AT -> Y."""
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    source_ap = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], L_ap, pos_ap, batch_size)
    direct_y, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, L_ap, L_at, output_ids, pos_ap, batch_size, require_grad=False)
    target = bank["counterfactual_label_ids"]["answer_token_after_pointer_interchange"].cpu()
    at_readout = acc((generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at, target)
    mediator_y = run_frozen_at_mediator(model, bank, generated_at, Q_at, L_at, output_ids, pos_at, batch_size, require_grad=False)
    return {
        "direct_iia": output_iia(direct_y, target, tokenizer, output_space),
        "at_readout_after_ap": at_readout,
        "mediator_iia": output_iia(mediator_y, target, tokenizer, output_space),
    }


@torch.no_grad()
def eval_c_chain(model, tokenizer, bank, Q_c, L_c, Q_ap, L_ap, W_ap, b_ap, Q_at, L_at, W_at, b_at, batch_size, output_space):
    """Evaluate direct C propagation and whether generated AP/AT are sufficient mediators."""
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], L_c, "color", batch_size)
    gen_ap, gen_at, y_after_c = run_c_only_and_capture_ap_at(model, bank, source_c, Q_c, L_c, L_ap, L_at, output_ids, batch_size)

    ap_target = bank["source_answer_pointer_ids"].cpu()
    at_target = bank["counterfactual_label_ids"]["answer_token"].cpu()
    y_target = bank["counterfactual_label_ids"]["color"].cpu()
    ap_implied_target = bank["counterfactual_label_ids"]["answer_token_after_pointer_interchange"].cpu()

    ap_logits = (gen_ap.to(Q_ap.device) @ Q_ap) @ W_ap.T + b_ap
    at_logits = (gen_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    at_after_ap, y_after_ap = replay_ap_and_capture_at(model, bank, gen_ap, Q_ap, L_ap, L_at, output_ids, batch_size)
    at_after_ap_logits = (at_after_ap.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    y_after_at = replay_at(model, bank, gen_at, Q_at, L_at, output_ids, batch_size)

    return {
        "ap_after_c": acc(ap_logits, ap_target),
        "at_after_c": acc(at_logits, at_target),
        "y_after_c": output_iia(y_after_c, y_target, tokenizer, output_space),
        "ap_replay_to_at_ap_implied": acc(at_after_ap_logits, ap_implied_target),
        "ap_replay_to_y_ap_implied": output_iia(y_after_ap, ap_implied_target, tokenizer, output_space),
        "ap_replay_to_at_source": acc(at_after_ap_logits, at_target),
        "ap_replay_to_y_source": output_iia(y_after_ap, y_target, tokenizer, output_space),
        "at_replay_to_y_source": output_iia(y_after_at, y_target, tokenizer, output_space),
    }


# ----------------------------- Main -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="randomized_copycolors")
    p.add_argument("--result-dir", default="results_joint_randomized")
    p.add_argument("--test-size", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-space", choices=["full", "az"], default="full")
    a = p.parse_args(); set_seed(a.seed)

    model, tokenizer = load_gemma_model(); device = next(model.parameters()).device; model.eval()
    for param in model.parameters(): param.requires_grad_(False)

    Q_at, L_at, pos_at = load_ckpt(f"{a.result_dir}/das_at.pt", device)
    Q_ap, L_ap, pos_ap = load_ckpt(f"{a.result_dir}/das_ap.pt", device)
    Q_c, L_c, pos_c = load_ckpt(f"{a.result_dir}/das_c.pt", device)
    W_at, b_at, _ = load_readout(f"{a.result_dir}/at_readout_LogisticRegression.pt", "AT", device)
    W_ap, b_ap, _ = load_readout(f"{a.result_dir}/ap_readout_LogisticRegression.pt", "AP", device)
    W_c, b_c, c_obj = load_readout(f"{a.result_dir}/c_readout_LogisticRegression.pt", "C", device)
    color_to_class = c_obj["color_to_class"]

    # Separate banks make each clean readout use examples where only that variable changes.
    ap_bank = build_test_bank(model, tokenizer, a.data_path, "AP_only", a.test_size, a.batch_size, a.seed + 1)
    at_bank = build_test_bank(model, tokenizer, a.data_path, "AT_only", a.test_size, a.batch_size, a.seed + 2)
    c_bank = build_test_bank(model, tokenizer, a.data_path, "C_only", a.test_size, a.batch_size, a.seed + 3)
    joint_bank = build_test_bank(model, tokenizer, a.data_path, "C_AP_AT", a.test_size, a.batch_size, a.seed + 4)

    print("\n=== CLEAN READOUT ===")
    print(f"AP base   : {readout_acc(model, ap_bank, Q_ap, L_ap, pos_ap, W_ap, b_ap, ap_bank['base_answer_pointer_ids'], a.batch_size, 'base'):.4f}")
    print(f"AP source : {readout_acc(model, ap_bank, Q_ap, L_ap, pos_ap, W_ap, b_ap, ap_bank['source_answer_pointer_ids'], a.batch_size):.4f}")
    print(f"AT base   : {readout_acc(model, at_bank, Q_at, L_at, pos_at, W_at, b_at, at_bank['base_answer_label_ids'], a.batch_size, 'base'):.4f}")
    print(f"AT source : {readout_acc(model, at_bank, Q_at, L_at, pos_at, W_at, b_at, at_bank['counterfactual_label_ids']['answer_token'], a.batch_size):.4f}")
    print(f"C base    : {readout_acc(model, c_bank, Q_c, L_c, pos_c, W_c, b_c, color_target(c_bank, color_to_class, 'base'), a.batch_size, 'base'):.4f}")
    print(f"C source  : {readout_acc(model, c_bank, Q_c, L_c, pos_c, W_c, b_c, color_target(c_bank, color_to_class, 'source'), a.batch_size):.4f}")

    apm = eval_ap_mediator(model, tokenizer, ap_bank, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, W_at, b_at, a.batch_size, a.output_space)
    print("\n=== AP -> AT -> Y ===")
    print(f"AP direct IIA       : {apm['direct_iia']:.4f}")
    print(f"AT readout after AP : {apm['at_readout_after_ap']:.4f}")
    print(f"Mediator IIA        : {apm['mediator_iia']:.4f}")

    cm = eval_c_chain(model, tokenizer, joint_bank, Q_c, L_c, Q_ap, L_ap, W_ap, b_ap, Q_at, L_at, W_at, b_at, a.batch_size, a.output_space)
    print("\n=== C -> AP -> AT -> Y ===")
    print(f"AP readout after C          : {cm['ap_after_c']:.4f}")
    print(f"AT readout after C          : {cm['at_after_c']:.4f}")
    print(f"Y after C                   : {cm['y_after_c']:.4f}")
    print(f"AP replay -> AT (AP target) : {cm['ap_replay_to_at_ap_implied']:.4f}")
    print(f"AP replay -> Y  (AP target) : {cm['ap_replay_to_y_ap_implied']:.4f}")
    print(f"AP replay -> AT (source)    : {cm['ap_replay_to_at_source']:.4f}")
    print(f"AP replay -> Y  (source)    : {cm['ap_replay_to_y_source']:.4f}")
    print(f"AT replay -> Y  (source)    : {cm['at_replay_to_y_source']:.4f}")


if __name__ == "__main__": main()
