"""Matched clean-AT donor control, simple version."""

import argparse, random, torch

from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import answer_label_ids, collect_layer_states, set_seed
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import build_pair_bank, factual_rows_correct, load_generated_dataset


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    Q = ckpt["basis"].float().to(device)
    layer = int(ckpt["layer"])
    pos = ckpt.get("token_position", "last_token")
    return Q, layer, pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="randomized_copycolors")
    p.add_argument("--result-dir", required=True)
    p.add_argument("--test-size", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    set_seed(a.seed)
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    Q_c, L_c, pos_c = load_ckpt(f"{a.result_dir}/das_c.pt", device)
    Q_ap, L_ap, pos_ap = load_ckpt(f"{a.result_dir}/das_ap.pt", device)
    Q_at, L_at, pos_at = load_ckpt(f"{a.result_dir}/das_at.pt", device)

    if pos_at != "last_token":
        raise ValueError("This simple script assumes AT is at last_token.")

    # ------------------------------------------------------------
    # 1. Build C_only evaluation pairs.
    # Each pair already contains:
    #   base
    #   C-source
    #   target AT after C intervention
    # ------------------------------------------------------------
    bank = build_pair_bank(
        model, tokenizer, a.data_path, "test", "C_only",
        a.test_size, a.batch_size, a.seed + 3
    )

    # ------------------------------------------------------------
    # 2. Build clean donor pool.
    # Keep only clean test prompts that the model answers correctly.
    # ------------------------------------------------------------
    ds = load_generated_dataset(a.data_path)
    rows = [ds["test"][i] for i in range(len(ds["test"]))]
    rows = factual_rows_correct(model, tokenizer, rows, a.batch_size)

    by_at = {}
    for row in rows:
        at = str(row["answer"])
        if at not in by_at:
            by_at[at] = []
        by_at[at].append(row)

    # ------------------------------------------------------------
    # 3. For each C pair, choose one different clean question
    # whose natural AT equals the desired AT_after_C.
    # ------------------------------------------------------------
    rng = random.Random(a.seed + 100)
    donors = []

    for pair in bank["pairs"]:
        target_at = pair["AT_after_C_intervene"]

        candidates = []
        for row in by_at[target_at]:
            if row["question_key"] != pair["question"]:
                candidates.append(row)

        donors.append(rng.choice(candidates))

    # Tokenize clean donors.
    tokenizer.padding_side = "left"
    donor_x = tokenizer(
        [row["prompt"] for row in donors],
        padding=True,
        return_tensors="pt",
    )

    donor_pos = {
        "last_token": donor_x["attention_mask"].sum(1).long() - 1
    }

    # ------------------------------------------------------------
    # 4. Run C intervention and capture the induced AT state.
    # ------------------------------------------------------------
    source_c = collect_layer_states(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        L_c,
        pos_c,
        a.batch_size,
    )

    direct_y, _, induced_at = run_c_intervention_and_capture(
        model, bank, source_c, Q_c, L_c, L_ap, L_at,
        None, pos_c, pos_ap, pos_at, a.batch_size, False
    )

    # ------------------------------------------------------------
    # 5. Replay induced AT into the same clean base.
    # ------------------------------------------------------------
    induced_y = run_frozen_mediator(
        model, bank, induced_at, Q_at, L_at,
        None, pos_at, a.batch_size, False
    )

    # ------------------------------------------------------------
    # 6. Get clean AT states from matched donors,
    # then replay them into the same clean base.
    # ------------------------------------------------------------
    clean_at = collect_layer_states(
        model,
        donor_x["input_ids"],
        donor_x["attention_mask"],
        donor_pos,
        L_at,
        pos_at,
        a.batch_size,
    )

    clean_y = run_frozen_mediator(
        model, bank, clean_at, Q_at, L_at,
        None, pos_at, a.batch_size, False
    )

    # ------------------------------------------------------------
    # 7. Compare predictions with target AT token IDs.
    # ------------------------------------------------------------
    label_ids = torch.tensor(answer_label_ids(tokenizer))
    target_class = bank["intervention_targets"]["AT_after_C_intervene"]
    target_token = label_ids[target_class]

    direct_pred = direct_y.argmax(-1).cpu()
    induced_pred = induced_y.argmax(-1).cpu()
    clean_pred = clean_y.argmax(-1).cpu()

    direct_ok = direct_pred == target_token
    induced_ok = induced_pred == target_token
    clean_ok = clean_pred == target_token

    print("\n=== MATCHED CLEAN-AT DONOR CONTROL ===")
    print("Direct C:", direct_ok.float().mean().item())
    print("Induced AT replay:", induced_ok.float().mean().item())
    print("Clean AT replay:", clean_ok.float().mean().item())
    print("Direct AND induced:", (direct_ok & induced_ok).float().mean().item())

    # Most important matched-control number:
    # only look at examples where clean AT transfer works.
    print(
        "Induced replay GIVEN clean succeeds:",
        induced_ok[clean_ok].float().mean().item(),
    )


if __name__ == "__main__":
    main()
