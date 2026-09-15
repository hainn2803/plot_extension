import argparse
import os
import random

import torch

from mcqa_data_load_all import build_mcqa_banks
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das_edge_eval import (
    set_seed,
    answer_label_ids,
    collect_layer_states,
    run_ap_patch_capture_at,
    answer_logits,
    ap_iia,
    full_answer_accuracy,
)


@torch.no_grad()
def run_conflict(
    model, bank, source_ap_states, donor_at_states, Q_ap, Q_at,
    ap_layer, at_layer, label_ids, token_position="last_token", batch_size=16,
):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source_ap = source_ap_states[start:end].to(device, dtype=torch.float32)
        donor_at = donor_at_states[start:end].to(device, dtype=torch.float32)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        raw_pos = bank["base_position_by_id"][token_position][start:end].to(device)
        pos = pad_offset + raw_pos

        def ap_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h = hidden.clone()
            base = hidden[rows, pos, :].float()
            h[rows, pos, :] = (base + ((source_ap - base) @ Q_ap) @ Q_ap.T).to(hidden.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        def at_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h = hidden.clone()
            current = hidden[rows, pos, :].float()
            # Force a conflicting AT value downstream of the AP intervention.
            h[rows, pos, :] = (current + ((donor_at - current) @ Q_at) @ Q_at.T).to(hidden.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        h1 = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        h2 = model.model.layers[at_layer].register_forward_hook(at_hook)
        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
            # Use the same left-padding-safe answer readout as the main evaluator.
            all_logits.append(answer_logits(model, outputs, mask, label_ids).cpu())
        finally:
            h2.remove()
            h1.remove()

    return torch.cat(all_logits, dim=0)


def make_conflict_donors(ap_target, base_target, at_target, seed=0):
    """Choose a donor whose AT differs from both AP-implied and base answers."""
    rng = random.Random(seed + 12345)
    donors = []
    n = len(at_target)

    for i in range(n):
        candidates = [
            j for j in range(n)
            if j != i
            and int(at_target[j]) != int(ap_target[i])
            and int(at_target[j]) != int(base_target[i])
        ]
        donors.append(rng.choice(candidates))

    return torch.tensor(donors, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--at-checkpoint", required=True)
    p.add_argument("--ap-checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save-path", default="results/ap_at_conflict_test_fixed.pt")
    args = p.parse_args()

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    at_ckpt = torch.load(args.at_checkpoint, map_location="cpu")
    ap_ckpt = torch.load(args.ap_checkpoint, map_location="cpu")

    L_at = int(at_ckpt["layer"])
    L_ap = int(ap_ckpt["ap_layer"])
    Q_at = at_ckpt["basis"].float().to(device)
    Q_ap = ap_ckpt["ap_basis"].float().to(device)

    cfg = ap_ckpt.get("config", {})
    ft = int(cfg.get("ft_size", 400))
    cal = int(cfg.get("cal_size", 200))
    te = int(cfg.get("te_size", 200))
    seed = int(cfg.get("seed", 0))
    token = ap_ckpt.get("token_position", "last_token")

    set_seed(seed)
    labels = answer_label_ids(tokenizer)

    _, _, te_banks = build_mcqa_banks(
        model=model, tokenizer=tokenizer,
        train_pool_size=ft, cal_size=cal, te_size=te,
        dataset_size=None, split="train", device=device,
        batch_size=args.batch_size, seed=seed,
    )
    bank = te_banks["answer_pointer"]

    source_ap = collect_layer_states(
        model, bank["source_input_ids"], bank["source_attention_mask"],
        bank["source_position_by_id"], L_ap, token, args.batch_size,
    )
    source_at = collect_layer_states(
        model, bank["source_input_ids"], bank["source_attention_mask"],
        bank["source_position_by_id"], L_at, token, args.batch_size,
    )

    ap_target = bank["counterfactual_label_ids"]["answer_pointer"].cpu()
    at_target = bank["counterfactual_label_ids"]["answer_token"].cpu()
    base_target = bank["base_answer_label_ids"].cpu()

    donor_idx = make_conflict_donors(ap_target, base_target, at_target, seed)
    donor_states = source_at[donor_idx]
    conflict_target = at_target[donor_idx]

    direct_logits, _ = run_ap_patch_capture_at(
        model, bank, source_ap, Q_ap, L_ap, L_at, labels, token, args.batch_size,
    )
    conflict_logits = run_conflict(
        model, bank, source_ap, donor_states, Q_ap, Q_at,
        L_ap, L_at, labels, token, args.batch_size,
    )

    direct_ap_acc = ap_iia(direct_logits, ap_target)
    follows_conflict = full_answer_accuracy(conflict_logits, conflict_target)
    still_follows_ap = ap_iia(conflict_logits, ap_target)
    follows_base = full_answer_accuracy(conflict_logits, base_target)

    print("\n===== AP / AT CONFLICT TEST =====")
    print(f"L_AP={L_ap} L_AT={L_at} n={len(bank['base_input_ids'])}")
    print(f"[DIRECT AP]       follows_AP={direct_ap_acc:.4f}")
    print(f"[CONFLICT AT]     follows_conflict_AT={follows_conflict:.4f}")
    print(f"[CONFLICT AT]     still_follows_AP={still_follows_ap:.4f}")
    print(f"[CONFLICT AT]     follows_base={follows_base:.4f}")

    result = {
        "L_AP": L_ap,
        "L_AT": L_at,
        "direct_follows_AP": direct_ap_acc,
        "conflict_follows_AT": follows_conflict,
        "conflict_still_follows_AP": still_follows_ap,
        "conflict_follows_base": follows_base,
        "donor_indices": donor_idx,
        "conflict_targets": conflict_target,
    }
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(result, args.save_path)
    print("saved to:", args.save_path)


if __name__ == "__main__":
    main()