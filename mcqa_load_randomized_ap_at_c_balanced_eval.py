import random

from mcqa_load_randomized_ap_at_c import *


def natural_row_to_pair(base_row, source_row, split, pair_type):
    pair = {
        "split": split,
        "question": base_row["question_key"],
        "pair_type": pair_type,
        "base_prompt": base_row["prompt"],
        "base_choices": base_row["choices"],
        "base_color": str(base_row["color"]),
        "base_AP": int(base_row["answer_position"]),
        "base_AT": str(base_row["answer"]),
        "base_Y": str(base_row["answer"]),
        "source_prompt": source_row["prompt"],
        "source_choices": source_row["choices"],
        "source_color": str(source_row["color"]),
        "source_AP": int(source_row["answer_position"]),
        "source_AT": str(source_row["answer"]),
        "source_Y": str(source_row["answer"]),
    }

    pair.update(intervention_targets(pair))
    return pair


def build_preservation_pairs(dataset_split, split, variable, seed):
    rows_by_state = {}

    for index in range(len(dataset_split)):
        row = dataset_split[index]

        state = (
            str(row["color"]).strip().casefold(),
            int(row["answer_position"]),
            str(row["answer"]),
        )

        if state not in rows_by_state:
            rows_by_state[state] = []

        rows_by_state[state].append(row)

    pairs = []

    for state in rows_by_state:
        rows = rows_by_state[state]

        if len(rows) < 2:
            continue

        for base_index in range(len(rows)):
            for source_index in range(len(rows)):
                if base_index == source_index:
                    continue

                base_row = rows[base_index]
                source_row = rows[source_index]

                if base_row["question_key"] == source_row["question_key"]:
                    continue

                pair = natural_row_to_pair(base_row, source_row, split, f"{variable}_preserve")

                if pair["base_color"] != pair["source_color"]:
                    raise ValueError("Preservation pair changed C")

                if pair["base_AP"] != pair["source_AP"]:
                    raise ValueError("Preservation pair changed AP")

                if pair["base_AT"] != pair["source_AT"]:
                    raise ValueError("Preservation pair changed AT")

                pairs.append(pair)

    random.Random(seed).shuffle(pairs)
    return pairs


def build_balanced_eval_bank(model, tokenizer, data_path, split, variable, size, batch_size, seed, change_ratio=0.5):
    if change_ratio < 0.0 or change_ratio > 1.0:
        raise ValueError(f"change_ratio must be in [0, 1], got {change_ratio}")

    dataset = load_generated_dataset(data_path)

    change_pair_types = {
        "AP": "AP_only",
        "AT": "AT_only",
        "C": "C_only",
    }

    if variable not in change_pair_types:
        raise ValueError(f"Unknown variable: {variable}")

    change_pair_type = change_pair_types[variable]

    change_size = int(round(size * change_ratio))
    preserve_size = size - change_size

    indices = list(range(len(dataset[split])))
    random.Random(seed).shuffle(indices)

    change_pairs = []
    cache = {}

    for start in range(0, len(indices), 64):
        batch_indices = indices[start:start + 64]
        row_pairs = rows_to_pairs(dataset[split], batch_indices, split)
        candidates = []

        for pair in row_pairs:
            if pair["pair_type"] == change_pair_type:
                candidates.append(pair)

        correct_pairs = predict_correct(model, tokenizer, candidates, batch_size, cache)
        change_pairs.extend(correct_pairs)

        if len(change_pairs) >= change_size:
            break

    if len(change_pairs) < change_size:
        raise ValueError(f"Requested {change_size} {variable}-change pairs, only found {len(change_pairs)}")

    change_pairs = change_pairs[:change_size]

    preservation_candidates = build_preservation_pairs(dataset[split], split, variable, seed + 1000)
    preservation_pairs = []

    for start in range(0, len(preservation_candidates), 64):
        candidates = preservation_candidates[start:start + 64]
        correct_pairs = predict_correct(model, tokenizer, candidates, batch_size, cache)
        preservation_pairs.extend(correct_pairs)

        if len(preservation_pairs) >= preserve_size:
            break

    if len(preservation_pairs) < preserve_size:
        raise ValueError(f"Requested {preserve_size} {variable}-preserve pairs, only found {len(preservation_pairs)}")

    preservation_pairs = preservation_pairs[:preserve_size]

    pairs = []
    pairs.extend(change_pairs)
    pairs.extend(preservation_pairs)
    random.Random(seed).shuffle(pairs)

    bank = create_bank(pairs, tokenizer, size, seed)

    num_change = 0
    num_preserve = 0

    for pair in bank["pairs"]:
        if pair["pair_type"] == change_pair_type:
            num_change += 1
        elif pair["pair_type"] == f"{variable}_preserve":
            num_preserve += 1

    print(f"[{variable} {split}] size={size} change={num_change} preserve={num_preserve} change_ratio={change_ratio:.2f}")

    return bank


def build_all_banks(model, tokenizer, data_path="randomized_copycolors", ft_size=800, cal_size=400, te_size=400, batch_size=32, seed=0, change_ratio=0.5):
    pair_types = {
        "AP": "AP_only",
        "AT": "AT_only",
        "C": "C_only",
    }

    result = {}

    for variable, pair_type in pair_types.items():
        train_bank = build_pair_bank(model, tokenizer, data_path, "train", pair_type, ft_size, batch_size, seed)

        cal_bank = build_balanced_eval_bank(model, tokenizer, data_path, "validation", variable, cal_size, batch_size, seed + 1, change_ratio)

        test_bank = build_balanced_eval_bank(model, tokenizer, data_path, "test", variable, te_size, batch_size, seed + 2, change_ratio)

        result[variable] = (
            train_bank,
            {variable: cal_bank},
            {variable: test_bank},
        )

    return result
