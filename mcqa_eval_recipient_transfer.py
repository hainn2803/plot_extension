"""Cross-question recipient transfer for frozen Standard DAS C -> AP -> AT -> Y.

For each intervention pair, draw a model-correct, held-out recipient from a
different question. "same" preserves the recipient's natural mediator value;
"different" must override that value. All draws are deterministic per eval seed.
"""

import random

import torch

from mcqa_gradual_das import (
    answer_label_ids,
    collect_joint_token_states_in_single_layer,
    collect_layer_states,
    output_label_ids,
)
from mcqa_gradual_das_ap_cross_tokens import run_ap_intervention_and_capture_at
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import (
    LETTER_TO_CLASS,
    color_to_class,
    factual_bank,
    factual_rows_correct,
    load_generated_dataset,
)
from mcqa_train_grid_standard_das import output_correct, readout_prediction


def prepare_recipient_pool(model, tokenizer, data_path, batch_size, seed):
    dataset = load_generated_dataset(data_path)
    rows = [dataset["test"][i] for i in range(len(dataset["test"]))]
    random.Random(seed + 4000).shuffle(rows)
    rows = factual_rows_correct(model, tokenizer, rows, batch_size)
    print(f"[recipient] {len(rows)} model-correct natural test rows")
    return rows, color_to_class(dataset)


def select_rows(bank, pool, kind, condition, seed):
    rng = random.Random(seed + {"AP_AT": 4301, "C_AP": 4302, "C_AT": 4303}[kind] + (0 if condition == "same" else 10000))
    picked = []
    targets = []

    for pair in bank["pairs"]:
        if kind == "AP_AT":
            value = str(pair["AT_after_AP_intervene"])
        elif kind == "C_AT":
            value = str(pair["AT_after_C_intervene"])
        else:
            value = int(pair["AP_after_C_intervene"])

        eligible = []
        for row in pool:
            if row["question_key"] == pair["question"]:
                continue
            natural = int(row["answer_position"]) if kind == "C_AP" else str(row["answer"])
            if (natural == value) == (condition == "same"):
                eligible.append(row)

        if not eligible:
            raise RuntimeError(f"No recipient for kind={kind}, condition={condition}, target={value}")

        row = rng.choice(eligible)
        picked.append(row)

        if kind == "C_AP":
            intended_letter = str(row["choices"]["label"][value])
        else:
            intended_letter = value
        targets.append(LETTER_TO_CLASS[intended_letter])

    return picked, torch.tensor(targets, dtype=torch.long)


def make_recipient_bank(rows, tokenizer, color_map):
    data = factual_bank(rows, tokenizer, color_map)
    return {
        "base_input_ids": data["input_ids"],
        "base_attention_mask": data["attention_mask"],
        "base_position_by_id": data["position_by_id"],
    }


def prepare_schedules(ap_bank, c_bank, pool, tokenizer, color_map, seed):
    schedules = {}
    for kind, donor in (("AP_AT", ap_bank), ("C_AP", c_bank), ("C_AT", c_bank)):
        schedules[kind] = {}
        for condition in ("same", "different"):
            rows, target = select_rows(donor, pool, kind, condition, seed)
            schedules[kind][condition] = (make_recipient_bank(rows, tokenizer, color_map), target)
    return schedules


@torch.no_grad()
def eval_recipient(model, tokenizer, ap_bank, c_bank, schedules,
                   Q_at, at_layer, pos_at, W_at, b_at,
                   Q_ap, ap_layer, pos_ap,
                   Q_c, c_layer, pos_c, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    source_ap = collect_joint_token_states_in_single_layer(
        model, ap_bank["source_input_ids"], ap_bank["source_attention_mask"],
        ap_bank["source_position_by_id"], ap_layer, pos_ap, batch_size,
    )
    _, induced_ap_at = run_ap_intervention_and_capture_at(
        model, ap_bank, source_ap, Q_ap, ap_layer, at_layer, output_ids,
        pos_ap, pos_at, batch_size, False,
    )

    source_c = collect_layer_states(
        model, c_bank["source_input_ids"], c_bank["source_attention_mask"],
        c_bank["source_position_by_id"], c_layer, pos_c, batch_size,
    )
    _, induced_c_ap, induced_c_at = run_c_intervention_and_capture(
        model, c_bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids,
        pos_c, pos_ap, pos_at, batch_size, False,
    )

    metrics = {"AP_only": {}, "C_only": {}}
    for condition in ("same", "different"):
        recipient, target = schedules["AP_AT"][condition]
        logits = run_frozen_mediator(
            model, recipient, induced_ap_at, Q_at, at_layer, output_ids,
            [pos_at], batch_size, False,
        )
        metrics["AP_only"][f"Y_via_AT_replay_recipient_{condition}"] = float(output_correct(logits, target, tokenizer, output_space).float().mean())

        recipient, target = schedules["C_AT"][condition]
        logits = run_frozen_mediator(
            model, recipient, induced_c_at, Q_at, at_layer, output_ids,
            [pos_at], batch_size, False,
        )
        metrics["C_only"][f"Y_via_AT_replay_recipient_{condition}"] = float(output_correct(logits, target, tokenizer, output_space).float().mean())

        recipient, target = schedules["C_AP"][condition]
        logits, generated_at = run_ap_intervention_and_capture_at(
            model, recipient, induced_c_ap, Q_ap, ap_layer, at_layer,
            output_ids, pos_ap, pos_at, batch_size, False,
        )
        metrics["C_only"][f"Y_via_AP_replay_recipient_{condition}"] = float(output_correct(logits, target, tokenizer, output_space).float().mean())
        prediction_at = readout_prediction(generated_at, Q_at, W_at, b_at)
        metrics["C_only"][f"AT_via_AP_replay_recipient_{condition}"] = float((prediction_at == target.cpu()).float().mean())

    return metrics
