"""Dataset helpers for AP -> AT -> Y and C -> AP -> AT -> Y experiments."""

import random
import re
import string
from pathlib import Path

import torch
from datasets import load_from_disk

from mcqa_constants import NUM_CHOICES

LETTERS = string.ascii_uppercase
LETTER_TO_CLASS = {}
for index, letter in enumerate(LETTERS):
    LETTER_TO_CLASS[letter] = index

PAIR_FIELDS = {
    "AP_only": "answerPosition_counterfactual",
    "AT_only": "randomLetter_counterfactual",
    "C_only": "color_counterfactual",
    "AP_AT": "answerPosition_letter_counterfactual",
    "C_AP_AT": "color_position_letter_counterfactual",
}

COLOR_PATTERN = re.compile(r"\b(?:is|are)\s+(.+?)\.\s*What color\b", re.IGNORECASE)


def option_color_positions(prompt, tokenizer):
    matches = list(re.finditer(r"(?m)^([A-Z])\. (.+?)\r?$", prompt))

    if len(matches) != 4:
        raise ValueError(f"Expected 4 option rows, got {len(matches)}")

    offsets = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)["offset_mapping"]
    color_positions = []

    for match in matches:
        color_start, color_end = match.span(2)
        candidates = []

        for index, offset in enumerate(offsets):
            start, end = offset

            if end > start and start < color_end and end > color_start:
                candidates.append(index)

        if len(candidates) == 0:
            raise ValueError(f"Cannot map color {match.group(2)!r}")

        color_positions.append(candidates[-1])

    return color_positions


def option_label_positions(prompt, tokenizer):
    matches = list(re.finditer(r"(?m)^([A-Z])\. (.+?)\r?$", prompt))

    if len(matches) != 4:
        raise ValueError(f"Expected 4 option rows, got {len(matches)}")

    offsets = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)["offset_mapping"]
    label_positions = []

    for match in matches:
        label_start, label_end = match.span(1)
        candidates = []

        for index, offset in enumerate(offsets):
            start, end = offset

            if end > start and start < label_end and end > label_start:
                candidates.append(index)

        if len(candidates) != 1:
            raise ValueError(f"Cannot uniquely map label {match.group(1)!r}")

        label_positions.append(candidates[0])

    return label_positions

    
def load_generated_dataset(path="randomized_copycolors"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path.resolve())

    dataset = load_from_disk(str(path))
    assert set(dataset) == {"train", "validation", "test"}

    questions = {}
    for split in dataset:
        questions[split] = set(dataset[split]["question_key"])

    assert questions["train"].isdisjoint(questions["validation"])
    assert questions["train"].isdisjoint(questions["test"])
    assert questions["validation"].isdisjoint(questions["test"])
    return dataset


def label_at_position(pair, position):
    return str(pair["base_choices"]["label"][int(position)])


def position_of_color_in_base(pair, color):
    color = str(color).strip().casefold()
    colors = []

    for text in pair["base_choices"]["text"]:
        colors.append(str(text).strip().casefold())

    matches = []
    for index, text in enumerate(colors):
        if text == color:
            matches.append(index)

    if len(matches) != 1:
        raise ValueError(f"Expected {color!r} once in base choices, got {matches}")

    return matches[0]


def intervention_targets(pair):
    ap_after_c = position_of_color_in_base(pair, pair["source_color"])
    at_after_c = label_at_position(pair, ap_after_c)
    at_after_ap = label_at_position(pair, pair["source_AP"])

    return {
        "AP_after_AP_intervene": pair["source_AP"],
        "AT_after_AP_intervene": at_after_ap,
        "Y_after_AP_intervene": at_after_ap,
        "AT_after_AT_intervene": pair["source_AT"],
        "Y_after_AT_intervene": pair["source_AT"],
        "C_after_C_intervene": pair["source_color"],
        "AP_after_C_intervene": ap_after_c,
        "AT_after_C_intervene": at_after_c,
        "Y_after_C_intervene": at_after_c,
    }


def row_to_pairs(row, split):
    base = {
        "prompt": row["prompt"],
        "choices": row["choices"],
        "color": str(row["color"]),
        "AP": int(row["answer_position"]),
        "AT": str(row["answer"]),
    }

    pairs = []

    for pair_type, field in PAIR_FIELDS.items():
        source = row[field]
        pair = {
            "split": split,
            "question": row["question_key"],
            "pair_type": pair_type,
            "base_prompt": base["prompt"],
            "base_choices": base["choices"],
            "base_color": base["color"],
            "base_AP": base["AP"],
            "base_AT": base["AT"],
            "base_Y": base["AT"],
            "source_prompt": source["prompt"],
            "source_choices": source["choices"],
            "source_color": str(source["color"]),
            "source_AP": int(source["answer_position"]),
            "source_AT": str(source["answer"]),
            "source_Y": str(source["answer"]),
        }
        pair.update(intervention_targets(pair))
        pairs.append(pair)

    return pairs


def rows_to_pairs(dataset_split, indices, split):
    pairs = []

    for index in indices:
        row = dataset_split[int(index)]
        row_pairs = row_to_pairs(row, split)
        for pair in row_pairs:
            pairs.append(pair)

    return pairs


@torch.no_grad()
def predict_correct(model, tokenizer, pairs, batch_size=32, cache=None):
    """Keep pairs where Gemma answers both natural base and source prompts correctly."""
    if cache is None:
        cache = {}

    gold = {}
    for pair in pairs:
        for side in ("base", "source"):
            prompt = pair[f"{side}_prompt"]
            gold[prompt] = pair[f"{side}_Y"]

    prompts = []
    for prompt in gold:
        if prompt not in cache:
            prompts.append(prompt)

    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        position_ids = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)

        try:
            logits = model(**inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1).logits[:, -1]
        except TypeError:
            logits = model(**inputs, position_ids=position_ids, use_cache=False).logits[:, -1]

        predicted_ids = logits.argmax(-1).tolist()

        for prompt, token_id in zip(batch, predicted_ids):
            prediction = tokenizer.decode([int(token_id)]).strip()
            cache[prompt] = prediction == gold[prompt]

    kept = []
    for pair in pairs:
        base_correct = cache[pair["base_prompt"]]
        source_correct = cache[pair["source_prompt"]]
        if base_correct and source_correct:
            kept.append(pair)

    return kept


def symbol_position(prompt, letter, tokenizer):
    match = re.search(rf"(?m)^[ \t]*{re.escape(letter)}\.", prompt)
    if match is None:
        raise ValueError(f"Cannot find choice label {letter!r}")

    prefix = prompt[:match.end() - 1]
    input_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    return len(input_ids) - 1


def color_position(prompt, tokenizer):
    first_line = prompt.splitlines()[0]
    match = COLOR_PATTERN.search(first_line)

    if match is None:
        raise ValueError(f"Cannot find color in {first_line!r}")

    color_start = match.start(1)
    offsets = tokenizer(first_line, add_special_tokens=True, return_offsets_mapping=True)["offset_mapping"]

    for index, offset in enumerate(offsets):
        start, end = offset
        if start <= color_start < end:
            return index

    raise ValueError("Cannot map color to token")


def ap_slot_label_position(pair, side, tokenizer):
    """Find the label token at the slot selected by source_AP, in either the base or source prompt."""
    ap = pair["source_AP"]
    letter = pair[f"{side}_choices"]["label"][ap]
    return symbol_position(pair[f"{side}_prompt"], letter, tokenizer)


def create_bank(pairs, tokenizer, size=None, seed=0):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)

    if size is not None:
        if len(pairs) < size:
            raise ValueError(f"Requested {size} pairs, only {len(pairs)} available")
        pairs = pairs[:size]

    if len(pairs) == 0:
        raise ValueError("No pairs available")

    tokenizer.padding_side = "left"

    base_prompts = []
    source_prompts = []
    for pair in pairs:
        base_prompts.append(pair["base_prompt"])
        source_prompts.append(pair["source_prompt"])

    base = tokenizer(base_prompts, padding=True, return_tensors="pt")
    source = tokenizer(source_prompts, padding=True, return_tensors="pt")

    base_golden_letter_positions = []
    source_golden_letter_positions = []
    base_source_ap_slot_positions = []
    source_source_ap_slot_positions = []

    c_targets = []
    ap_after_ap_targets = []
    at_after_at_targets = []
    at_after_ap_targets = []
    y_after_ap_targets = []
    y_after_at_targets = []
    ap_after_c_targets = []
    at_after_c_targets = []
    y_after_c_targets = []

    pair_types = []
    base_answer_label_ids = []
    source_answer_label_ids = []
    base_answer_pointer_ids = []
    source_answer_pointer_ids = []

    base_color_positions = []
    source_color_positions = []

    base_all_label_positions = [[] for i in range(NUM_CHOICES)]
    source_all_label_positions = [[] for i in range(NUM_CHOICES)]

    base_all_color_positions = [[] for i in range(NUM_CHOICES)]
    source_all_color_positions = [[] for i in range(NUM_CHOICES)]

    for pair in pairs:

        base_golden_letter_positions.append(symbol_position(pair["base_prompt"], pair["base_AT"], tokenizer))
        source_golden_letter_positions.append(symbol_position(pair["source_prompt"], pair["source_AT"], tokenizer))

        base_source_ap_slot_positions.append(ap_slot_label_position(pair, "base", tokenizer))
        source_source_ap_slot_positions.append(ap_slot_label_position(pair, "source", tokenizer))

        base_label_positions = option_label_positions(pair["base_prompt"], tokenizer)
        source_label_positions = option_label_positions(pair["source_prompt"], tokenizer)

        list_base_color_positions = option_color_positions(pair["base_prompt"], tokenizer)
        list_source_color_positions = option_color_positions(pair["source_prompt"], tokenizer)

        for option_index in range(NUM_CHOICES):
            base_all_label_positions[option_index].append(base_label_positions[option_index])
            source_all_label_positions[option_index].append(source_label_positions[option_index])

            base_all_color_positions[option_index].append(list_base_color_positions[option_index])
            source_all_color_positions[option_index].append(list_source_color_positions[option_index])

        c_targets.append(pair["C_after_C_intervene"])
        ap_after_ap_targets.append(pair["AP_after_AP_intervene"])
        at_after_at_targets.append(LETTER_TO_CLASS[pair["AT_after_AT_intervene"]])
        at_after_ap_targets.append(LETTER_TO_CLASS[pair["AT_after_AP_intervene"]])
        y_after_ap_targets.append(LETTER_TO_CLASS[pair["Y_after_AP_intervene"]])
        y_after_at_targets.append(LETTER_TO_CLASS[pair["Y_after_AT_intervene"]])
        ap_after_c_targets.append(pair["AP_after_C_intervene"])
        at_after_c_targets.append(LETTER_TO_CLASS[pair["AT_after_C_intervene"]])
        y_after_c_targets.append(LETTER_TO_CLASS[pair["Y_after_C_intervene"]])

        pair_types.append(pair["pair_type"])
        base_answer_label_ids.append(LETTER_TO_CLASS[pair["base_AT"]])
        source_answer_label_ids.append(LETTER_TO_CLASS[pair["source_AT"]])
        base_answer_pointer_ids.append(pair["base_AP"])
        source_answer_pointer_ids.append(pair["source_AP"])

        base_color_positions.append(color_position(pair["base_prompt"], tokenizer))
        source_color_positions.append(color_position(pair["source_prompt"], tokenizer))

    targets = {
        "C_after_C_intervene": c_targets,
        "AP_after_AP_intervene": torch.tensor(ap_after_ap_targets),
        "AT_after_AT_intervene": torch.tensor(at_after_at_targets),
        "AT_after_AP_intervene": torch.tensor(at_after_ap_targets),
        "Y_after_AP_intervene": torch.tensor(y_after_ap_targets),
        "Y_after_AT_intervene": torch.tensor(y_after_at_targets),
        "AP_after_C_intervene": torch.tensor(ap_after_c_targets),
        "AT_after_C_intervene": torch.tensor(at_after_c_targets),
        "Y_after_C_intervene": torch.tensor(y_after_c_targets),
    }

    base_positions = {
        "last_token": base["attention_mask"].sum(1).long() - 1,
        "color": torch.tensor(base_color_positions),
        "golden_letter": torch.tensor(base_golden_letter_positions),
        "source_ap_slot_label": torch.tensor(base_source_ap_slot_positions),
    }

    for i in range(NUM_CHOICES):
        base_positions[f"label_{i}"] = torch.tensor(base_all_label_positions[i])
        base_positions[f"color_{i}"] = torch.tensor(base_all_color_positions[i])

    source_positions = {
        "last_token": source["attention_mask"].sum(1).long() - 1,
        "color": torch.tensor(source_color_positions),
        "golden_letter": torch.tensor(source_golden_letter_positions),
        "source_ap_slot_label": torch.tensor(source_source_ap_slot_positions),
    }

    for i in range(NUM_CHOICES):
        source_positions[f"label_{i}"] = torch.tensor(source_all_label_positions[i])
        source_positions[f"color_{i}"] = torch.tensor(source_all_color_positions[i])

    return {
        "pairs": pairs,
        "pair_types": pair_types,
        "base_input_ids": base["input_ids"].long(),
        "base_attention_mask": base["attention_mask"].long(),
        "source_input_ids": source["input_ids"].long(),
        "source_attention_mask": source["attention_mask"].long(),
        "base_position_by_id": base_positions,
        "source_position_by_id": source_positions,
        "base_answer_label_ids": torch.tensor(base_answer_label_ids),
        "source_answer_label_ids": torch.tensor(source_answer_label_ids),
        "base_answer_pointer_ids": torch.tensor(base_answer_pointer_ids),
        "source_answer_pointer_ids": torch.tensor(source_answer_pointer_ids),
        "intervention_targets": targets,
    }


def build_pair_bank(model, tokenizer, data_path, split, pair_type, size, batch_size, seed):
    dataset = load_generated_dataset(data_path)
    indices = list(range(len(dataset[split])))
    random.Random(seed).shuffle(indices)

    pairs = []
    cache = {}

    for start in range(0, len(indices), 64):
        batch_indices = indices[start:start + 64]
        row_pairs = rows_to_pairs(dataset[split], batch_indices, split)
        candidates = []

        for pair in row_pairs:
            if pair["pair_type"] == pair_type:
                candidates.append(pair)

        correct_pairs = predict_correct(model, tokenizer, candidates, batch_size, cache)
        pairs.extend(correct_pairs)

        if len(pairs) >= size:
            break

    return create_bank(pairs, tokenizer, size, seed)


def build_all_banks(model, tokenizer, data_path="randomized_copycolors", ft_size=800, cal_size=400, te_size=400, batch_size=32, seed=0):
    sizes = {"train": ft_size, "validation": cal_size, "test": te_size}
    pair_types = {"AP": "AP_only", "AT": "AT_only", "C": "C_only"}
    banks = {}

    for variable in pair_types:
        banks[variable] = {}

    splits = ("train", "validation", "test")
    for offset, split in enumerate(splits):
        for variable, pair_type in pair_types.items():
            banks[variable][split] = build_pair_bank(model, tokenizer, data_path, split, pair_type, sizes[split], batch_size, seed + offset)

    result = {}
    for variable, variable_banks in banks.items():
        cal = {variable: variable_banks["validation"]}
        test = {variable: variable_banks["test"]}
        result[variable] = (variable_banks["train"], cal, test)

    return result


def color_to_class(dataset):
    colors = set()

    for split in dataset:
        for row in dataset[split]:
            color = str(row["color"]).strip().casefold()
            colors.add(color)

    sorted_colors = sorted(colors)
    result = {}

    for index, color in enumerate(sorted_colors):
        result[color] = index

    return result


@torch.no_grad()
def factual_rows_correct(model, tokenizer, rows, batch_size):
    """Keep only original factual rows that Gemma answers correctly."""
    device = next(model.parameters()).device
    kept = []
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = []

        for row in batch:
            prompts.append(row["prompt"])

        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        position_ids = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)

        try:
            logits = model(**inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1).logits[:, -1]
        except TypeError:
            logits = model(**inputs, position_ids=position_ids, use_cache=False).logits[:, -1]

        predicted_ids = logits.argmax(-1).tolist()

        for row, token_id in zip(batch, predicted_ids):
            prediction = tokenizer.decode([int(token_id)]).strip()
            if prediction == str(row["answer"]):
                kept.append(row)

    return kept


def factual_bank(rows, tokenizer, color_map):
    prompts = []
    questions = []
    ap_targets = []
    at_targets = []
    c_targets = []
    color_positions = []
    base_all_label_positions = [[] for i in range(NUM_CHOICES)]
    base_all_color_positions = [[] for i in range(NUM_CHOICES)]

    for row in rows:
        prompt = row["prompt"]

        label_positions = option_label_positions(prompt, tokenizer)
        option_color_positions_list = option_color_positions(prompt, tokenizer)

        for option_index in range(NUM_CHOICES):
            base_all_label_positions[option_index].append(label_positions[option_index])
            base_all_color_positions[option_index].append(option_color_positions_list[option_index])

        prompts.append(prompt)
        questions.append(row["question_key"])
        ap_targets.append(int(row["answer_position"]))
        at_targets.append(LETTER_TO_CLASS[str(row["answer"])])
        c_targets.append(color_map[str(row["color"]).strip().casefold()])
        color_positions.append(color_position(prompt, tokenizer))

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    positions = {
        "last_token": inputs["attention_mask"].sum(1).long() - 1,
        "color": torch.tensor(color_positions),
    }

    for i in range(NUM_CHOICES):
        positions[f"label_{i}"] = torch.tensor(base_all_label_positions[i])
        positions[f"color_{i}"] = torch.tensor(base_all_color_positions[i])

    return {
        "prompts": prompts,
        "questions": questions,
        "input_ids": inputs["input_ids"].long(),
        "attention_mask": inputs["attention_mask"].long(),
        "position_by_id": positions,
        "AP_target_ids": torch.tensor(ap_targets),
        "AT_target_ids": torch.tensor(at_targets),
        "C_target_ids": torch.tensor(c_targets),
        "color_to_class": color_map,
    }


@torch.no_grad()
def build_shared_readout_banks(model, tokenizer, data_path="randomized_copycolors", train_size=1300, val_size=520, batch_size=32, seed=0):
    dataset = load_generated_dataset(data_path)
    colors = color_to_class(dataset)
    banks = []

    settings = (("train", train_size, seed), ("validation", val_size, seed + 1))

    for split, size, split_seed in settings:
        rows = []
        for index in range(len(dataset[split])):
            rows.append(dataset[split][index])

        random.Random(split_seed).shuffle(rows)
        rows = factual_rows_correct(model, tokenizer, rows, batch_size)

        if len(rows) < size:
            raise RuntimeError(f"{split}: requested {size} correct factual rows, found {len(rows)}")

        banks.append(factual_bank(rows[:size], tokenizer, colors))
        print(f"[readout] {split}: {size} shared factual prompts")

    return tuple(banks)


def readout_view(shared_banks, target):
    result = []

    for bank in shared_banks:
        view = dict(bank)
        view["target_ids"] = bank[f"{target}_target_ids"]
        result.append(view)

    return tuple(result)


def build_ap_readout_datasets(shared_banks):
    return readout_view(shared_banks, "AP")


def build_at_readout_datasets(shared_banks):
    return readout_view(shared_banks, "AT")


def build_c_readout_datasets(shared_banks):
    return readout_view(shared_banks, "C")
