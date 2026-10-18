"""Simple loader for the randomized AP / AT / C CopyColors dataset.

Each generated row gives four intervention pairs:
    AP_only : AP changes, C/AT stay fixed
    AT_only : AT changes, C/AP stay fixed
    C_only  : C changes, AP/AT stay fixed
    C_AP_AT : C, AP, and AT all change

Training/validation/test:
    AP -> only AP_only pairs
    AT -> only AT_only pairs
    C  -> only C_only pairs
"""

import random
import re
import string
from pathlib import Path

import torch
from datasets import load_from_disk


LETTERS = string.ascii_uppercase
LETTER_TO_CLASS = {letter: i for i, letter in enumerate(LETTERS)}

PAIR_FIELDS = {
    "AP_only": "answerPosition_counterfactual",
    "AT_only": "randomLetter_counterfactual",
    "C_only": "color_counterfactual",
    "C_AP_AT": "color_all_counterfactual",
}

COLOR_PATTERN = re.compile(r"\b(?:is|are)\s+(.+?)\.\s*What color\b", re.IGNORECASE)


def load_generated_dataset(path="randomized_copycolors"):
    """Load the saved DatasetDict and verify question-disjoint splits."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    ds = load_from_disk(str(path))
    expected = {"train", "validation", "test"}
    if set(ds.keys()) != expected:
        raise ValueError(f"Expected splits {sorted(expected)}, got {list(ds.keys())}")

    question_sets = {split: set(ds[split]["question_key"]) for split in expected}

    for split in ("train", "validation", "test"):
        print(f"[load] {split}: rows={len(ds[split])} questions={len(question_sets[split])}")

    assert question_sets["train"].isdisjoint(question_sets["validation"])
    assert question_sets["train"].isdisjoint(question_sets["test"])
    assert question_sets["validation"].isdisjoint(question_sets["test"])
    print("[load] train/validation/test questions are disjoint")

    return ds


def row_to_pairs(row, split):
    """Convert one generated row into the four base/source intervention pairs."""
    base = {
        "prompt": row["prompt"],
        "choices": row["choices"],
        "color": str(row["color"]),
        "AP": int(row["answer_position"]),
        "AT": str(row["answer"]),
    }

    pairs = []
    for pair_type, field in PAIR_FIELDS.items():
        cf = row[field]
        pairs.append({
            "split": split,
            "question": row["question_key"],
            "base_row_id": row["base_row_id"],
            "repeat_idx": int(row["repeat_idx"]),
            "pair_type": pair_type,
            "base_prompt": base["prompt"],
            "base_choices": base["choices"],
            "base_color": base["color"],
            "base_AP": base["AP"],
            "base_AT": base["AT"],
            "base_Y": base["AT"],
            "source_prompt": cf["prompt"],
            "source_choices": cf["choices"],
            "source_color": str(cf["color"]),
            "source_AP": int(cf["answer_position"]),
            "source_AT": str(cf["answer"]),
            "source_Y": str(cf["answer"]),
        })

    return pairs


def rows_to_pairs(dataset_split, row_indices, split):
    pairs = []
    for row_idx in row_indices:
        pairs.extend(row_to_pairs(dataset_split[int(row_idx)], split))
    return pairs


@torch.no_grad()
def predict_correct(model, tokenizer, pairs, batch_size=32, correct_cache=None):
    """Keep only pairs for which Gemma answers both base and source correctly."""
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    correct_cache = {} if correct_cache is None else correct_cache

    gold_by_prompt = {}
    for pair in pairs:
        gold_by_prompt[pair["base_prompt"]] = pair["base_Y"]
        gold_by_prompt[pair["source_prompt"]] = pair["source_Y"]

    unknown_prompts = [prompt for prompt in gold_by_prompt if prompt not in correct_cache]

    gold_token_ids = {}
    for letter in set(gold_by_prompt.values()):
        ids = set()
        for text in (" " + letter, letter):
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))
        if not ids:
            raise ValueError(f"No single-token encoding for answer {letter!r}")
        gold_token_ids[letter] = ids

    for start in range(0, len(unknown_prompts), batch_size):
        prompts = unknown_prompts[start:start + batch_size]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        position_ids = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)

        try:
            outputs = model(**inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1)
        except TypeError:
            outputs = model(**inputs, position_ids=position_ids, use_cache=False)

        predicted_ids = outputs.logits[:, -1].argmax(-1).tolist()
        for prompt, token_id in zip(prompts, predicted_ids):
            gold = gold_by_prompt[prompt]
            decoded = tokenizer.decode([int(token_id)]).strip()
            correct_cache[prompt] = token_id in gold_token_ids[gold] or decoded == gold

    return [
        pair for pair in pairs
        if correct_cache[pair["base_prompt"]] and correct_cache[pair["source_prompt"]]
    ]


def symbol_position(prompt, letter, tokenizer):
    """Token position of the answer-choice letter, e.g. X in 'X. magenta'."""
    match = re.search(rf"(?m)^[ \t]*{re.escape(letter)}\.", prompt)
    if match is None:
        raise ValueError(f"Cannot find choice label {letter!r} in prompt")
    prefix = prompt[:match.end() - 1]
    return len(tokenizer(prefix, add_special_tokens=True)["input_ids"]) - 1


def color_position(prompt, tokenizer):
    """Token position of the color in the question sentence."""
    first_line = prompt.splitlines()[0]
    match = COLOR_PATTERN.search(first_line)
    if match is None:
        raise ValueError(f"Cannot find color in prompt: {first_line!r}")

    color_start = match.start(1)
    encoded = tokenizer(first_line, add_special_tokens=True, return_offsets_mapping=True)
    for token_idx, (start, end) in enumerate(encoded["offset_mapping"]):
        if start <= color_start < end:
            return token_idx

    raise ValueError(f"Cannot map color to token position: {first_line!r}")


def question_mark_position(prompt, tokenizer):
    first_line = prompt.splitlines()[0]
    char_idx = first_line.rfind("?")
    if char_idx < 0:
        raise ValueError(f"Cannot find question mark in prompt: {first_line!r}")
    prefix = first_line[:char_idx + 1]
    return len(tokenizer(prefix, add_special_tokens=True)["input_ids"]) - 1


def ap_output_target(pair):
    """After AP <- source_AP, read the label at that position in the BASE choices."""
    return str(pair["base_choices"]["label"][pair["source_AP"]])


def at_output_target(pair):
    return pair["source_AT"]


def c_output_target(pair):
    """Changing C follows the source answer; unchanged C is a no-op."""
    return pair["source_Y"] if pair["base_color"] != pair["source_color"] else pair["base_Y"]


def create_bank(pairs, tokenizer, size=None, seed=0):
    """Tokenize a list of pairs and create the tensor bank used by DAS."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)

    if size is not None:
        if len(pairs) < size:
            raise ValueError(f"Requested {size} pairs, only {len(pairs)} available")
        pairs = pairs[:size]
    if not pairs:
        raise ValueError("No pairs available")

    tokenizer.padding_side = "left"
    base = tokenizer([p["base_prompt"] for p in pairs], padding=True, return_tensors="pt")
    source = tokenizer([p["source_prompt"] for p in pairs], padding=True, return_tensors="pt")

    base_symbol = torch.tensor([symbol_position(p["base_prompt"], p["base_AT"], tokenizer) for p in pairs])
    source_symbol = torch.tensor([symbol_position(p["source_prompt"], p["source_AT"], tokenizer) for p in pairs])
    base_color = torch.tensor([color_position(p["base_prompt"], tokenizer) for p in pairs])
    source_color = torch.tensor([color_position(p["source_prompt"], tokenizer) for p in pairs])
    base_qmark = torch.tensor([question_mark_position(p["base_prompt"], tokenizer) for p in pairs])
    source_qmark = torch.tensor([question_mark_position(p["source_prompt"], tokenizer) for p in pairs])

    ap_targets = [LETTER_TO_CLASS[ap_output_target(p)] for p in pairs]
    at_targets = [LETTER_TO_CLASS[at_output_target(p)] for p in pairs]
    c_targets = [LETTER_TO_CLASS[c_output_target(p)] for p in pairs]

    return {
        "pairs": pairs,
        "pair_types": [p["pair_type"] for p in pairs],
        "base_input_ids": base["input_ids"].long(),
        "base_attention_mask": base["attention_mask"].long(),
        "source_input_ids": source["input_ids"].long(),
        "source_attention_mask": source["attention_mask"].long(),
        "base_position_by_id": {
            "last_token": base["attention_mask"].sum(dim=1).long() - 1,
            "correct_symbol": base_symbol.long(),
            "correct_symbol_period": (base_symbol + 1).long(),
            "color": base_color.long(),
            "question_mark": base_qmark.long(),
        },
        "source_position_by_id": {
            "last_token": source["attention_mask"].sum(dim=1).long() - 1,
            "correct_symbol": source_symbol.long(),
            "correct_symbol_period": (source_symbol + 1).long(),
            "color": source_color.long(),
            "question_mark": source_qmark.long(),
        },
        "base_answer_label_ids": torch.tensor([LETTER_TO_CLASS[p["base_AT"]] for p in pairs]),
        "base_answer_pointer_ids": torch.tensor([p["base_AP"] for p in pairs]),
        "source_answer_pointer_ids": torch.tensor([p["source_AP"] for p in pairs]),
        "counterfactual_label_ids": {
            "answer_pointer": torch.tensor([p["source_AP"] for p in pairs]),
            "answer_token": torch.tensor([LETTER_TO_CLASS[p["source_AT"]] for p in pairs]),
            "answer_token_after_pointer_interchange": torch.tensor(ap_targets),
            "answer_token_direct": torch.tensor(at_targets),
            "color": torch.tensor(c_targets),
        },
        "changed_mask": {
            "answer_pointer": torch.tensor([p["base_AP"] != p["source_AP"] for p in pairs]),
            "answer_token": torch.tensor([p["base_AT"] != p["source_AT"] for p in pairs]),
            "color": torch.tensor([p["base_color"] != p["source_color"] for p in pairs]),
        },
    }


def build_all_banks(model, tokenizer, data_path="randomized_copycolors", ft_size=800,
                    cal_size=400, te_size=400, batch_size=32, seed=0, rows_per_chunk=64):
    """Build variable-specific train/validation/test banks for AP, AT, and C."""
    ds = load_generated_dataset(data_path)
    sizes = {"train": ft_size, "validation": cal_size, "test": te_size}
    split_seed = {"train": seed, "validation": seed + 1, "test": seed + 2}
    correct_cache = {}
    banks = {}

    for split in ("train", "validation", "test"):
        wanted = sizes[split]
        row_indices = list(range(len(ds[split])))
        random.Random(split_seed[split]).shuffle(row_indices)

        ap_pairs, at_pairs, c_pairs = [], [], []

        for start in range(0, len(row_indices), rows_per_chunk):
            indices = row_indices[start:start + rows_per_chunk]
            candidates = rows_to_pairs(ds[split], indices, split)
            filtered = predict_correct(model, tokenizer, candidates, batch_size, correct_cache)

            ap_pairs.extend(p for p in filtered if p["pair_type"] == "AP_only")
            at_pairs.extend(p for p in filtered if p["pair_type"] == "AT_only")
            c_pairs.extend(p for p in filtered if p["pair_type"] == "C_only")

            if min(len(ap_pairs), len(at_pairs), len(c_pairs)) >= wanted:
                break

        print(
            f"\n[{split}] model-correct candidates: "
            f"AP_only={len(ap_pairs)} AT_only={len(at_pairs)} C_only={len(c_pairs)}"
        )

        banks[split] = {
            "AP": create_bank(ap_pairs, tokenizer, wanted, split_seed[split]),
            "AT": create_bank(at_pairs, tokenizer, wanted, split_seed[split]),
            "C": create_bank(c_pairs, tokenizer, wanted, split_seed[split]),
        }
        print(f"[bank] {split}: AP={wanted} AT={wanted} C={wanted}")

    return {
        "AP": (
            banks["train"]["AP"],
            {"answer_pointer": banks["validation"]["AP"]},
            {"answer_pointer": banks["test"]["AP"]},
        ),
        "AT": (
            banks["train"]["AT"],
            {"answer_token": banks["validation"]["AT"]},
            {"answer_token": banks["test"]["AT"]},
        ),
        "C": (
            banks["train"]["C"],
            {"color": banks["validation"]["C"]},
            {"color": banks["test"]["C"]},
        ),
    }


if __name__ == "__main__":
    ds = load_generated_dataset("randomized_copycolors")
    for pair in row_to_pairs(ds["train"][0], "train"):
        print(
            f"{pair['pair_type']:8s} | "
            f"C {pair['base_color']} -> {pair['source_color']} | "
            f"AP {pair['base_AP']} -> {pair['source_AP']} | "
            f"AT {pair['base_AT']} -> {pair['source_AT']}"
        )
