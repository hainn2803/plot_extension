"""Dataset helpers for AP -> AT -> Y and C -> AP -> AT -> Y experiments."""

import random, re, string
from pathlib import Path
import torch
from datasets import load_from_disk

LETTERS = string.ascii_uppercase
LETTER_TO_CLASS = {x: i for i, x in enumerate(LETTERS)}
PAIR_FIELDS = {
    "AP_only": "answerPosition_counterfactual",
    "AT_only": "randomLetter_counterfactual",
    "C_only": "color_counterfactual",
    "AP_AT": "answerPosition_letter_counterfactual",
    "C_AP_AT": "color_position_letter_counterfactual",
}
COLOR_PATTERN = re.compile(r"\b(?:is|are)\s+(.+?)\.\s*What color\b", re.IGNORECASE)


def load_generated_dataset(path="randomized_copycolors"):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(path.resolve())
    ds = load_from_disk(str(path))
    assert set(ds) == {"train", "validation", "test"}
    questions = {split: set(ds[split]["question_key"]) for split in ds}
    assert questions["train"].isdisjoint(questions["validation"])
    assert questions["train"].isdisjoint(questions["test"])
    assert questions["validation"].isdisjoint(questions["test"])
    return ds


def label_at_position(pair, position):
    return str(pair["base_choices"]["label"][int(position)])


def position_of_color_in_base(pair, color):
    color = str(color).strip().casefold()
    colors = [str(x).strip().casefold() for x in pair["base_choices"]["text"]]
    matches = [i for i, x in enumerate(colors) if x == color]
    if len(matches) != 1: raise ValueError(f"Expected {color!r} once in base choices, got {matches}")
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
    base = {"prompt": row["prompt"], "choices": row["choices"], "color": str(row["color"]), "AP": int(row["answer_position"]), "AT": str(row["answer"])}
    pairs = []
    for pair_type, field in PAIR_FIELDS.items():
        source = row[field]
        pair = {
            "split": split, "question": row["question_key"], "pair_type": pair_type,
            "base_prompt": base["prompt"], "base_choices": base["choices"], "base_color": base["color"], "base_AP": base["AP"], "base_AT": base["AT"], "base_Y": base["AT"],
            "source_prompt": source["prompt"], "source_choices": source["choices"], "source_color": str(source["color"]), "source_AP": int(source["answer_position"]), "source_AT": str(source["answer"]), "source_Y": str(source["answer"]),
        }
        pair.update(intervention_targets(pair))
        pairs.append(pair)
    return pairs


def rows_to_pairs(dataset_split, indices, split):
    return [pair for i in indices for pair in row_to_pairs(dataset_split[int(i)], split)]


@torch.no_grad()
def predict_correct(model, tokenizer, pairs, batch_size=32, cache=None):
    """Keep pairs where Gemma answers both natural base and source prompts correctly."""
    cache = {} if cache is None else cache
    gold = {p[f"{side}_prompt"]: p[f"{side}_Y"] for p in pairs for side in ("base", "source")}
    prompts = [p for p in gold if p not in cache]
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        x = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        pos = (x["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
        try: logits = model(**x, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1]
        except TypeError: logits = model(**x, position_ids=pos, use_cache=False).logits[:, -1]
        for prompt, token in zip(batch, logits.argmax(-1).tolist()):
            cache[prompt] = tokenizer.decode([int(token)]).strip() == gold[prompt]

    return [p for p in pairs if cache[p["base_prompt"]] and cache[p["source_prompt"]]]


def symbol_position(prompt, letter, tokenizer):
    match = re.search(rf"(?m)^[ \t]*{re.escape(letter)}\.", prompt)
    if match is None: raise ValueError(f"Cannot find choice label {letter!r}")
    return len(tokenizer(prompt[:match.end() - 1], add_special_tokens=True)["input_ids"]) - 1


def color_position(prompt, tokenizer):
    first_line = prompt.splitlines()[0]
    match = COLOR_PATTERN.search(first_line)
    if match is None: raise ValueError(f"Cannot find color in {first_line!r}")
    color_start = match.start(1)
    offsets = tokenizer(first_line, add_special_tokens=True, return_offsets_mapping=True)["offset_mapping"]
    for i, (start, end) in enumerate(offsets):
        if start <= color_start < end: return i
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

    if not pairs:
        raise ValueError("No pairs available")

    tokenizer.padding_side = "left"
    base = tokenizer([p["base_prompt"] for p in pairs], padding=True, return_tensors="pt")
    source = tokenizer([p["source_prompt"] for p in pairs], padding=True, return_tensors="pt")

    base_golden_letter_positions = []
    source_golden_letter_positions = []

    for p in pairs:
        base_golden_letter_positions.append(symbol_position(p["base_prompt"], p["base_AT"], tokenizer))
        source_golden_letter_positions.append(symbol_position(p["source_prompt"], p["source_AT"], tokenizer))

    targets = {
        "C_after_C_intervene": [p["C_after_C_intervene"] for p in pairs],
        "AP_after_AP_intervene": torch.tensor([p["AP_after_AP_intervene"] for p in pairs]),
        "AT_after_AT_intervene": torch.tensor([LETTER_TO_CLASS[p["AT_after_AT_intervene"]] for p in pairs]),
        "AT_after_AP_intervene": torch.tensor([LETTER_TO_CLASS[p["AT_after_AP_intervene"]] for p in pairs]),
        "Y_after_AP_intervene": torch.tensor([LETTER_TO_CLASS[p["Y_after_AP_intervene"]] for p in pairs]),
        "Y_after_AT_intervene": torch.tensor([LETTER_TO_CLASS[p["Y_after_AT_intervene"]] for p in pairs]),
        "AP_after_C_intervene": torch.tensor([p["AP_after_C_intervene"] for p in pairs]),
        "AT_after_C_intervene": torch.tensor([LETTER_TO_CLASS[p["AT_after_C_intervene"]] for p in pairs]),
        "Y_after_C_intervene": torch.tensor([LETTER_TO_CLASS[p["Y_after_C_intervene"]] for p in pairs]),
    }

    return {
        "pairs": pairs,
        "pair_types": [p["pair_type"] for p in pairs],
        "base_input_ids": base["input_ids"].long(),
        "base_attention_mask": base["attention_mask"].long(),
        "source_input_ids": source["input_ids"].long(),
        "source_attention_mask": source["attention_mask"].long(),
        "base_position_by_id": {
            "last_token": base["attention_mask"].sum(1).long() - 1,
            "color": torch.tensor([color_position(p["base_prompt"], tokenizer) for p in pairs]),
            "golden_letter": torch.tensor(base_golden_letter_positions),
            "source_ap_slot_label": torch.tensor([ap_slot_label_position(p, "base", tokenizer) for p in pairs]),
        },
        "source_position_by_id": {
            "last_token": source["attention_mask"].sum(1).long() - 1,
            "color": torch.tensor([color_position(p["source_prompt"], tokenizer) for p in pairs]),
            "golden_letter": torch.tensor(source_golden_letter_positions),
            "source_ap_slot_label": torch.tensor([ap_slot_label_position(p, "source", tokenizer) for p in pairs]),
        },
        "base_answer_label_ids": torch.tensor([LETTER_TO_CLASS[p["base_AT"]] for p in pairs]),
        "source_answer_label_ids": torch.tensor([LETTER_TO_CLASS[p["source_AT"]] for p in pairs]),
        "base_answer_pointer_ids": torch.tensor([p["base_AP"] for p in pairs]),
        "source_answer_pointer_ids": torch.tensor([p["source_AP"] for p in pairs]),
        "intervention_targets": targets,
    }


def build_pair_bank(model, tokenizer, data_path, split, pair_type, size, batch_size, seed):
    ds = load_generated_dataset(data_path); ids = list(range(len(ds[split]))); random.Random(seed).shuffle(ids)
    pairs, cache = [], {}
    for start in range(0, len(ids), 64):
        candidates = [p for p in rows_to_pairs(ds[split], ids[start:start + 64], split) if p["pair_type"] == pair_type]
        pairs += predict_correct(model, tokenizer, candidates, batch_size, cache)
        if len(pairs) >= size: break
    return create_bank(pairs, tokenizer, size, seed)


def build_all_banks(model, tokenizer, data_path="randomized_copycolors", ft_size=800, cal_size=400, te_size=400, batch_size=32, seed=0):
    sizes = {"train": ft_size, "validation": cal_size, "test": te_size}
    pair_type = {"AP": "AP_only", "AT": "AT_only", "C": "C_only"}
    banks = {v: {} for v in pair_type}
    for offset, split in enumerate(("train", "validation", "test")):
        for variable, kind in pair_type.items():
            banks[variable][split] = build_pair_bank(model, tokenizer, data_path, split, kind, sizes[split], batch_size, seed + offset)
    return {v: (b["train"], {v: b["validation"]}, {v: b["test"]}) for v, b in banks.items()}


# ---------------- Readout dataset: same factual prompts for AP, AT, and C ----------------

def color_to_class(ds):
    colors = sorted({str(row["color"]).strip().casefold() for split in ds for row in ds[split]})
    return {color: i for i, color in enumerate(colors)}


@torch.no_grad()
def factual_rows_correct(model, tokenizer, rows, batch_size):
    """Keep only original factual rows that Gemma answers correctly."""
    device = next(model.parameters()).device; kept = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        x = tokenizer([r["prompt"] for r in batch], padding=True, return_tensors="pt").to(device)
        pos = (x["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
        try: logits = model(**x, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1]
        except TypeError: logits = model(**x, position_ids=pos, use_cache=False).logits[:, -1]
        for row, token in zip(batch, logits.argmax(-1).tolist()):
            if tokenizer.decode([int(token)]).strip() == str(row["answer"]): kept.append(row)
    return kept


def factual_bank(rows, tokenizer, color_map):
    prompts = [r["prompt"] for r in rows]; answers = [str(r["answer"]) for r in rows]
    x = tokenizer(prompts, padding=True, return_tensors="pt")
    return {
        "prompts": prompts, "questions": [r["question_key"] for r in rows],
        "input_ids": x["input_ids"].long(), "attention_mask": x["attention_mask"].long(),
        "position_by_id": {"last_token": x["attention_mask"].sum(1).long() - 1, "color": torch.tensor([color_position(p, tokenizer) for p in prompts])},
        "AP_target_ids": torch.tensor([int(r["answer_position"]) for r in rows]),
        "AT_target_ids": torch.tensor([LETTER_TO_CLASS[a] for a in answers]),
        "C_target_ids": torch.tensor([color_map[str(r["color"]).strip().casefold()] for r in rows]),
        "color_to_class": color_map,
    }


@torch.no_grad()
def build_shared_readout_banks(model, tokenizer, data_path="randomized_copycolors", train_size=1300, val_size=520, batch_size=32, seed=0):
    ds = load_generated_dataset(data_path); colors = color_to_class(ds)
    banks = []
    for split, size, split_seed in (("train", train_size, seed), ("validation", val_size, seed + 1)):
        rows = [ds[split][i] for i in range(len(ds[split]))]; random.Random(split_seed).shuffle(rows)
        rows = factual_rows_correct(model, tokenizer, rows, batch_size)
        if len(rows) < size: raise RuntimeError(f"{split}: requested {size} correct factual rows, found {len(rows)}")
        banks.append(factual_bank(rows[:size], tokenizer, colors))
        print(f"[readout] {split}: {size} shared factual prompts")
    return tuple(banks)


def readout_view(shared_banks, target):
    result = []
    for bank in shared_banks:
        view = dict(bank); view["target_ids"] = bank[f"{target}_target_ids"]; result.append(view)
    return tuple(result)


def build_ap_readout_datasets(shared_banks): 
    return readout_view(shared_banks, "AP")

def build_at_readout_datasets(shared_banks): 
    return readout_view(shared_banks, "AT")

def build_c_readout_datasets(shared_banks): 
    return readout_view(shared_banks, "C")



# if __name__ == "__main__":
#     from mcqa_neural_net import load_gemma_model


#     ds = load_generated_dataset("randomized_copycolors")
#     model, tokenizer = load_gemma_model()
#     row = ds["train"][0]
#     pairs = row_to_pairs(row, "train")

#     for pair in pairs:
#         if pair["pair_type"] != "AP_only":
#             continue

#         ap = pair["source_AP"]

#         base_letter = pair["base_choices"]["label"][ap]
#         source_letter = pair["source_choices"]["label"][ap]

#         base_pos = ap_slot_label_position(pair, "base", tokenizer)
#         source_pos = ap_slot_label_position(pair, "source", tokenizer)

#         base_ids = tokenizer(pair["base_prompt"], add_special_tokens=True)["input_ids"]
#         source_ids = tokenizer(pair["source_prompt"], add_special_tokens=True)["input_ids"]

#         base_token = tokenizer.decode([base_ids[base_pos]]).strip()
#         source_token = tokenizer.decode([source_ids[source_pos]]).strip()

#         print("\n================ AP SLOT SANITY CHECK ================")

#         print("\nBASE")
#         print(pair["base_prompt"])
#         print(f"base_AP={pair['base_AP']}")
#         print(f"source_AP={pair['source_AP']}")
#         print(f"label at source_AP slot={base_letter}")
#         print(f"token position={base_pos}")
#         print(f"decoded token={base_token!r}")

#         print("\nSOURCE")
#         print(pair["source_prompt"])
#         print(f"source_AP={pair['source_AP']}")
#         print(f"label at source_AP slot={source_letter}")
#         print(f"token position={source_pos}")
#         print(f"decoded token={source_token!r}")

#         print("\nEXPECTED")
#         print(f"source_AT={pair['source_AT']}")
#         print(f"Y_after_AP={pair['Y_after_AP_intervene']}")

#         assert base_token == base_letter
#         assert source_token == source_letter
#         assert source_letter == pair["source_AT"]
#         assert base_letter == pair["Y_after_AP_intervene"]

#         print("\nSanity check passed.")

# if __name__ == "__main__":

#     ds = load_generated_dataset("randomized_copycolors")

#     for i in range(3):
#         row = ds["train"][i]
#         pairs = row_to_pairs(row, "train")

#         print(f"\n================ ROW {i} ================")

#         for pair in pairs:
#             print(f"\n---------- {pair['pair_type']} ----------")

#             print("\nBASE")
#             print(pair["base_prompt"])
#             print(f"C={pair['base_color']} AP={pair['base_AP']} AT={pair['base_AT']} Y={pair['base_Y']}")

#             print("\nSOURCE")
#             print(pair["source_prompt"])
#             print(f"C={pair['source_color']} AP={pair['source_AP']} AT={pair['source_AT']} Y={pair['source_Y']}")

#             print("\nINTERVENTION TARGETS")

#             if pair["pair_type"] == "AP_only":
#                 print(f"AP_after_AP = {pair['AP_after_AP_intervene']}")
#                 print(f"AT_after_AP = {pair['AT_after_AP_intervene']}")
#                 print(f"Y_after_AP  = {pair['Y_after_AP_intervene']}")

#             elif pair["pair_type"] == "AT_only":
#                 print(f"AT_after_AT = {pair['AT_after_AT_intervene']}")
#                 print(f"Y_after_AT  = {pair['Y_after_AT_intervene']}")

#             elif pair["pair_type"] in ("C_only", "C_AP_AT"):
#                 print(f"C_after_C   = {pair['C_after_C_intervene']}")
#                 print(f"AP_after_C  = {pair['AP_after_C_intervene']}")
#                 print(f"AT_after_C  = {pair['AT_after_C_intervene']}")
#                 print(f"Y_after_C   = {pair['Y_after_C_intervene']}")