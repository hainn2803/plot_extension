"""Question-disjoint MCQA banks from jchang153/copycolors_mcqa.

Follows the author's three original-to-counterfactual pairs and shared,
unfiltered-by-target DAS train bank. Calibration and test require a change
in the variable under intervention. Unlike the author's base-row grouping,
all rows containing the same full question stay in the same split. The
question split is cached and reused for both DAS and factual readouts.
"""

import json
import os
import random
import re
import string
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset


DATASET_PATH = "jchang153/copycolors_mcqa"
CACHE_DIR = Path(os.environ.get("MCQA_CACHE_DIR", Path(__file__).resolve().parent / "hf_cache"))
PAIR_SPECS = {
    "answerPosition_counterfactual": ("original", "answerPosition_counterfactual"),
    "randomLetter_counterfactual": ("original", "randomLetter_counterfactual"),
    "answerPosition_randomLetter_counterfactual": (
        "original", "answerPosition_randomLetter_counterfactual"
    ),
}
LETTERS = string.ascii_uppercase
LETTER_TO_CLASS = {letter: i for i, letter in enumerate(LETTERS)}
COLOR_IN_QUESTION = re.compile(
    r"\b(?:is|are)\s+(.+?)\.\s*What color\b", re.IGNORECASE
)


def question_key(row):
    """Use the full question, including the stated color, but no choices."""
    first_line = row["prompt"].splitlines()[0].strip()
    if first_line.lower().startswith("question:"):
        first_line = first_line[len("question:"):].strip()
    if "?" not in first_line:
        raise ValueError(f"No question mark in prompt: {first_line!r}")
    return " ".join(first_line.split("?", 1)[0].split()).casefold() + "?"


def split_question_indices(dataset, seed=0, train_fraction=0.7, cal_fraction=0.15):
    """Assign each complete question and all its rows to one split."""
    if not (0 < train_fraction < 1 and 0 < cal_fraction < 1 - train_fraction):
        raise ValueError("Expected 0 < train_fraction, cal_fraction and sum < 1")
    grouped = defaultdict(list)
    for index, row in enumerate(dataset):
        grouped[question_key(row)].append(index)
    questions = list(grouped)
    random.Random(seed).shuffle(questions)
    row_limits = (int(len(dataset) * train_fraction), int(len(dataset) * cal_fraction))
    indices = {"train": [], "validation": [], "test": []}
    for question in questions:
        if len(indices["train"]) < row_limits[0]:
            split = "train"
        elif len(indices["validation"]) < row_limits[1]:
            split = "validation"
        else:
            split = "test"
        indices[split].extend(grouped[question])
    return indices


def check_disjoint_questions_cross_dataset(dataset_train, dataset_validation, dataset_test):
    """Raise with examples if any complete question occurs in two splits."""
    datasets = {
        "train": dataset_train,
        "validation": dataset_validation,
        "test": dataset_test,
    }
    question_sets = {}
    for name, dataset in datasets.items():
        question_sets[name] = {question_key(row) for row in dataset}
        print(f"[split] {name}: rows={len(dataset)} questions={len(question_sets[name])}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = question_sets[left] & question_sets[right]
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(f"Question overlap between {left} and {right}: {examples}")
    print("[split] train/validation/test questions are disjoint")


def load_question_splits(seed=0, train_fraction=0.7, cal_fraction=0.15):
    """Load the public dataset once; cache row-index splits on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(path=DATASET_PATH, split="train", cache_dir=str(CACHE_DIR))
    manifest_path = CACHE_DIR / (
        f"jchang_question_split_seed{seed}_train{train_fraction}_cal{cal_fraction}.json"
    )
    expected = {
        "dataset_path": DATASET_PATH,
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "num_rows": len(dataset),
        "seed": seed,
        "train_fraction": train_fraction,
        "cal_fraction": cal_fraction,
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if all(saved.get(key) == value for key, value in expected.items()):
            indices = saved["indices"]
            print(f"[split] reusing {manifest_path}")
        else:
            indices = None
    else:
        indices = None
    if indices is None:
        indices = split_question_indices(dataset, seed, train_fraction, cal_fraction)
        tmp_path = manifest_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({**expected, "indices": indices}))
        tmp_path.replace(manifest_path)
        print(f"[split] saved {manifest_path}")
    all_indices = [i for name in ("train", "validation", "test") for i in indices[name]]
    if len(all_indices) != len(dataset) or set(all_indices) != set(range(len(dataset))):
        raise ValueError("Question split manifest does not cover each original row once")
    subsets = {name: dataset.select(indices[name]) for name in indices}
    check_disjoint_questions_cross_dataset(
        subsets["train"], subsets["validation"], subsets["test"]
    )
    return dataset, indices


def _question_color(row):
    first_line = row["prompt"].splitlines()[0]
    match = COLOR_IN_QUESTION.search(first_line)
    if match is None:
        raise ValueError(f"Cannot parse stated color from {first_line!r}")
    color = match.group(1).strip().casefold()
    if sum(str(x).strip().casefold() == color for x in row["choices"]["text"]) != 1:
        raise ValueError(f"Stated color {color!r} must occur exactly once in base choices")
    return color


def _answer(choices, color):
    labels = [str(label) for label in choices["label"]]
    texts = [str(value).strip().casefold() for value in choices["text"]]
    if len(labels) != 4 or len(texts) != 4 or len(set(labels)) != 4:
        raise ValueError("Expected four choices with distinct labels")
    if any(label not in LETTER_TO_CLASS for label in labels):
        raise ValueError(f"Expected A-Z choice labels, got {labels}")
    positions = [i for i, text in enumerate(texts) if text == color]
    if len(positions) != 1:
        raise ValueError(f"Stated color {color!r} must occur once in source choices")
    position = positions[0]
    return position, labels[position]


def pairs_for_indices(dataset, indices, split, seed=0):
    """Create pairs only for the requested original rows, after splitting."""
    pairs_by_family = {family: [] for family in PAIR_SPECS}
    for row_idx in indices:
        row = dataset[int(row_idx)]
        color = _question_color(row)
        for name, (base_kind, source_kind) in PAIR_SPECS.items():
            base = row if base_kind == "original" else row[base_kind]
            source = row if source_kind == "original" else row[source_kind]
            base_ap, base_at = _answer(base["choices"], color)
            source_ap, source_at = _answer(source["choices"], color)
            pair = {
                "split": split,
                "base_row_id": f"jchang:train:{row_idx}",
                "question": question_key(row),
                "base_family": base_kind,
                "source_family": source_kind,
                "pair_family": name,
                "base_prompt": base["prompt"],
                "base_choices": base["choices"],
                "base_AP": base_ap,
                "base_AT": base_at,
                "base_Y": base_at,
                "source_prompt": source["prompt"],
                "source_choices": source["choices"],
                "source_AP": source_ap,
                "source_AT": source_at,
                "source_Y": source_at,
            }
            pairs_by_family[name].append(pair)
    return pairs_by_family


def load_mcqa_datasets(split, seed=0):
    """Return (pairs_by_family, question-disjoint subset) for inspection."""
    dataset, indices = load_question_splits(seed)
    if split not in indices:
        raise ValueError(f"Unknown split: {split}")
    return (
        pairs_for_indices(dataset, indices[split], split, seed),
        dataset.select(indices[split]),
    )


@torch.no_grad()
def predict_correct(model, tokenizer, pair_examples, batch_size=32, device=None,
                    correct_by_prompt=None):
    """Retain pairs where full-vocabulary argmax equals both gold letters."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = device or next(model.parameters()).device
    if correct_by_prompt is None:
        correct_by_prompt = {}
    gold_by_prompt = {}
    for family_pairs in pair_examples.values():
        for pair in family_pairs:
            for side in ("base", "source"):
                prompt = pair[f"{side}_prompt"]
                gold = pair[f"{side}_Y"]
                if prompt in gold_by_prompt and gold_by_prompt[prompt] != gold:
                    raise ValueError("Identical prompts have conflicting answers")
                gold_by_prompt[prompt] = gold
    unknown = [prompt for prompt in gold_by_prompt if prompt not in correct_by_prompt]
    gold_token_ids = {}
    for letter in set(gold_by_prompt.values()):
        variants = set()
        for spelling in (" " + letter, letter):
            ids = tokenizer(spelling, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                variants.add(int(ids[0]))
        if not variants:
            raise ValueError(f"No single-token encoding for answer {letter!r}")
        gold_token_ids[letter] = variants
    for start in range(0, len(unknown), batch_size):
        prompts = unknown[start:start + batch_size]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        position_ids = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
        try:
            outputs = model(**inputs, position_ids=position_ids,
                            use_cache=False, logits_to_keep=1)
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            outputs = model(**inputs, position_ids=position_ids, use_cache=False)
        predicted = outputs.logits[:, -1].argmax(dim=-1).tolist()
        for prompt, token_id in zip(prompts, predicted):
            text = tokenizer.decode([int(token_id)]).strip()
            gold = gold_by_prompt[prompt]
            correct_by_prompt[prompt] = token_id in gold_token_ids[gold] or text == gold
    filtered = {}
    for family, family_pairs in pair_examples.items():
        filtered[family] = [
            pair for pair in family_pairs
            if correct_by_prompt[pair["base_prompt"]]
            and correct_by_prompt[pair["source_prompt"]]
        ]
    return filtered


def create_bank(filtered_pairs, target, tokenizer, size=None, seed=0):
    """Keep the exact field names and label class conventions used by DAS."""
    pairs = [pair for group in filtered_pairs.values() for pair in group]
    if target == "shared":
        pass
    elif target == "AP":
        pairs = [p for p in pairs if p["base_AP"] != p["source_AP"]]
    elif target == "AT":
        pairs = [p for p in pairs if p["base_AT"] != p["source_AT"]]
    else:
        raise ValueError(f"Unknown target: {target}")
    random.Random(seed).shuffle(pairs)
    if size is not None:
        if len(pairs) < size:
            raise ValueError(f"{target}: requested {size} pairs but only {len(pairs)} passed")
        pairs = pairs[:size]
    if not pairs:
        raise ValueError(f"{target}: no model-correct changing pairs")

    tokenizer.padding_side = "left"
    base_tokens = tokenizer([p["base_prompt"] for p in pairs], padding=True, return_tensors="pt")
    source_tokens = tokenizer([p["source_prompt"] for p in pairs], padding=True, return_tensors="pt")

    def symbol_position(prompt, letter):
        match = re.search(rf"(?m)^[ \t]*{re.escape(letter)}\.", prompt)
        if match is None:
            raise ValueError(f"Cannot locate choice label {letter!r} in prompt")
        prefix = prompt[:match.end() - 1]
        return len(tokenizer(prefix, add_special_tokens=True)["input_ids"]) - 1

    base_symbol = torch.tensor([
        symbol_position(p["base_prompt"], p["base_AT"]) for p in pairs
    ], dtype=torch.long)
    source_symbol = torch.tensor([
        symbol_position(p["source_prompt"], p["source_AT"]) for p in pairs
    ], dtype=torch.long)
    after_ap = [p["base_choices"]["label"][p["source_AP"]] for p in pairs]
    return {
        "pairs": pairs,
        "pair_source_families": [p["source_family"] for p in pairs],
        "pair_families": [p["pair_family"] for p in pairs],
        "base_input_ids": base_tokens["input_ids"].long(),
        "base_attention_mask": base_tokens["attention_mask"].long(),
        "source_input_ids": source_tokens["input_ids"].long(),
        "source_attention_mask": source_tokens["attention_mask"].long(),
        "base_position_by_id": {
            "last_token": base_tokens["attention_mask"].sum(dim=1).long() - 1,
            "correct_symbol": base_symbol,
            "correct_symbol_period": base_symbol + 1,
        },
        "source_position_by_id": {
            "last_token": source_tokens["attention_mask"].sum(dim=1).long() - 1,
            "correct_symbol": source_symbol,
            "correct_symbol_period": source_symbol + 1,
        },
        "base_answer_label_ids": torch.tensor([
            LETTER_TO_CLASS[p["base_AT"]] for p in pairs
        ], dtype=torch.long),
        "base_answer_pointer_ids": torch.tensor([p["base_AP"] for p in pairs], dtype=torch.long),
        "source_answer_pointer_ids": torch.tensor([p["source_AP"] for p in pairs], dtype=torch.long),
        "counterfactual_label_ids": {
            "answer_pointer": torch.tensor([p["source_AP"] for p in pairs], dtype=torch.long),
            "answer_token": torch.tensor([
                LETTER_TO_CLASS[p["source_AT"]] for p in pairs
            ], dtype=torch.long),
            "answer_token_after_pointer_interchange": torch.tensor([
                LETTER_TO_CLASS[letter] for letter in after_ap
            ], dtype=torch.long),
        },
        "changed_mask": {
            "answer_pointer": torch.tensor([
                p["base_AP"] != p["source_AP"] for p in pairs
            ], dtype=torch.bool),
            "answer_token": torch.tensor([
                p["base_AT"] != p["source_AT"] for p in pairs
            ], dtype=torch.bool),
        },
    }


def build_all_mcqa_banks(model, tokenizer, ft_size, cal_size, te_size,
                         batch_size=32, seed=0):
    """Shared train bank; changed-pair AP/AT calibration and test banks."""
    dataset, split_indices = load_question_splits(seed)
    device = next(model.parameters()).device
    correct_cache = {}
    banks = {}
    for split, size in (("train", ft_size), ("validation", cal_size), ("test", te_size)):
        indices = list(split_indices[split])
        random.Random(seed + {"train": 0, "validation": 1, "test": 2}[split]).shuffle(indices)
        kept = {"shared": [], "AP": [], "AT": []}
        rows_scored = 0
        # Score a small set of rows at a time instead of running Gemma on all 10k.
        for start in range(0, len(indices), 128):
            current = indices[start:start + 128]
            candidate = pairs_for_indices(dataset, current, split, seed + start)
            filtered = predict_correct(
                model, tokenizer, candidate, batch_size, device, correct_cache
            )
            rows_scored += len(current)
            for pairs in filtered.values():
                for pair in pairs:
                    kept["shared"].append(pair)
                    if pair["base_AP"] != pair["source_AP"]:
                        kept["AP"].append(pair)
                    if pair["base_AT"] != pair["source_AT"]:
                        kept["AT"].append(pair)
            required = ("shared",) if split == "train" else ("AP", "AT")
            if size is not None and all(len(kept[target]) >= size for target in required):
                break
        banks[split] = {}
        for target in required:
            bank = create_bank({target: kept[target]}, target, tokenizer,
                               size=size, seed=seed + {"train": 0, "validation": 1, "test": 2}[split])
            banks[split][target] = bank
            print(f"[bank] {split} {target}: {len(bank['pairs'])} pairs "
                  f"(available={len(kept[target])}, rows_scored={rows_scored})")
        if split == "train":
            banks[split]["AP"] = banks[split]["shared"]
            banks[split]["AT"] = banks[split]["shared"]
    # Final assertion over selected pairs, in addition to the row split check.
    selected = {
        split: {p["question"] for target in ("AP", "AT") for p in banks[split][target]["pairs"]}
        for split in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert selected[left].isdisjoint(selected[right]), f"{left}/{right} question overlap"
    return {
        "AP": (banks["train"]["AP"],
               {"answer_pointer": banks["validation"]["AP"]},
               {"answer_pointer": banks["test"]["AP"]}),
        "AT": (banks["train"]["AT"],
               {"answer_token": banks["validation"]["AT"]},
               {"answer_token": banks["test"]["AT"]}),
    }


READOUT_CHOICE_LINE = re.compile(r"(?m)^([ \t]*)([A-Z])\.[ \t]+([^\r\n]+)$")


def _make_readout_example(row, target, requested_class, rng):
    """Move only choice colors and labels; keep the complete question fixed."""
    matches = list(READOUT_CHOICE_LINE.finditer(row["prompt"]))
    labels_in_prompt = [match.group(2) for match in matches]
    if len(matches) != 4 or labels_in_prompt != list(row["choices"]["label"]):
        raise ValueError("Expected four choice lines matching choices.label")
    colors = [match.group(3) for match in matches]
    original_position, _ = _answer(row["choices"], _question_color(row))
    position = requested_class if target == "AP" else rng.randrange(4)
    colors[original_position], colors[position] = colors[position], colors[original_position]
    if target == "AT":
        correct_letter = LETTERS[requested_class]
        labels = rng.sample([letter for letter in LETTERS if letter != correct_letter], 3)
        labels.insert(position, correct_letter)
    else:
        labels = rng.sample(LETTERS, 4)
        correct_letter = labels[position]
    prompt = row["prompt"]
    for i in range(3, -1, -1):
        match = matches[i]
        replacement = f"{match.group(1)}{labels[i]}. {colors[i]}"
        prompt = prompt[:match.start()] + replacement + prompt[match.end():]
    return {
        "prompt": prompt,
        "answer_letter": correct_letter,
        "answer_position": position,
        "target_id": requested_class,
        "question": question_key(row),
    }


@torch.no_grad()
def _readout_correct(model, tokenizer, proposals, device, batch_size):
    """Score the proposed factual prompts over the full next-token vocabulary."""
    correct = []
    for start in range(0, len(proposals), batch_size):
        batch = proposals[start:start + batch_size]
        inputs = tokenizer([p["prompt"] for p in batch], padding=True,
                           return_tensors="pt").to(device)
        position_ids = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
        try:
            outputs = model(**inputs, position_ids=position_ids,
                            use_cache=False, logits_to_keep=1)
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            outputs = model(**inputs, position_ids=position_ids, use_cache=False)
        predicted_ids = outputs.logits[:, -1].argmax(dim=-1).tolist()
        correct.extend(
            tokenizer.decode([int(token_id)]).strip() == item["answer_letter"]
            for item, token_id in zip(batch, predicted_ids)
        )
    return correct


def _readout_symbol_position(prompt, letter, tokenizer):
    match = re.search(rf"(?m)^[ \t]*{re.escape(letter)}\.", prompt)
    if match is None:
        raise ValueError(f"Cannot find correct choice {letter!r}")
    return len(tokenizer(prompt[:match.end() - 1],
                         add_special_tokens=True)["input_ids"]) - 1


def _tokenize_readout(examples, tokenizer):
    tokens = tokenizer([item["prompt"] for item in examples],
                       padding=True, return_tensors="pt")
    symbols = torch.tensor([
        _readout_symbol_position(item["prompt"], item["answer_letter"], tokenizer)
        for item in examples
    ], dtype=torch.long)
    return {
        "prompts": [item["prompt"] for item in examples],
        "questions": [item["question"] for item in examples],
        "answer_letters": [item["answer_letter"] for item in examples],
        "answer_positions": torch.tensor([
            item["answer_position"] for item in examples
        ], dtype=torch.long),
        "target_ids": torch.tensor([
            item["target_id"] for item in examples
        ], dtype=torch.long),
        "input_ids": tokens["input_ids"].long(),
        "attention_mask": tokens["attention_mask"].long(),
        "position_by_id": {
            "last_token": tokens["attention_mask"].sum(dim=1).long() - 1,
            "correct_symbol": symbols,
            "correct_symbol_period": symbols + 1,
        },
    }


def build_readout_datasets(
    model, tokenizer, target, examples_per_class=50, *,
    val_examples_per_class=20, train_split="train", val_split="validation",
    batch_size=32, max_attempts_per_class=500, seed=0, cache_dir=None,
):
    """Create balanced factual (train, validation) banks from the SAME split.

    `cache_dir` remains in the signature for compatibility with the old
    readout builder; the question split uses MCQA_CACHE_DIR for both callers.
    """
    if target not in ("AT", "AP"):
        raise ValueError("target must be AT or AP")
    if train_split != "train" or val_split != "validation":
        raise ValueError("This builder uses the cached train/validation question split")
    if min(examples_per_class, val_examples_per_class,
           max_attempts_per_class, batch_size) < 1:
        raise ValueError("Readout counts and batch_size must be positive")
    dataset, indices = load_question_splits(seed)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device
    classes = range(26 if target == "AT" else 4)
    banks = {}
    for offset, (name, quota) in enumerate((
        ("train", examples_per_class), ("validation", val_examples_per_class)
    )):
        rng = random.Random(seed + offset)
        row_indices = indices[name]
        accepted = {i: [] for i in classes}
        seen_prompts = set()
        for _ in range(max_attempts_per_class):
            proposals = []
            for class_id in classes:
                if len(accepted[class_id]) >= quota:
                    continue
                for _ in range(20):
                    row = dataset[row_indices[rng.randrange(len(row_indices))]]
                    item = _make_readout_example(row, target, class_id, rng)
                    if item["prompt"] not in seen_prompts:
                        seen_prompts.add(item["prompt"])
                        proposals.append(item)
                        break
            if not proposals:
                break
            for item, correct in zip(
                proposals, _readout_correct(model, tokenizer, proposals, device, batch_size)
            ):
                if correct and len(accepted[item["target_id"]]) < quota:
                    accepted[item["target_id"]].append(item)
            if all(len(items) == quota for items in accepted.values()):
                break
        missing = {
            (LETTERS[i] if target == "AT" else i): quota - len(items)
            for i, items in accepted.items() if len(items) < quota
        }
        if missing:
            raise RuntimeError(
                f"{name}: insufficient Gemma-correct {target} prompts "
                f"after {max_attempts_per_class} attempts per class: {missing}"
            )
        examples = [item for i in classes for item in accepted[i]]
        rng.shuffle(examples)
        banks[name] = _tokenize_readout(examples, tokenizer)
        print(f"[readout {target}] {name}: {quota} per class, {len(examples)} prompts")
    assert set(banks["train"]["questions"]).isdisjoint(banks["validation"]["questions"])
    return banks["train"], banks["validation"]


if __name__ == "__main__":
    data, indices = load_question_splits(seed=0)
    print("[split] cached sizes:", {name: len(rows) for name, rows in indices.items()})
