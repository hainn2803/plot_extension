import random
import string

import torch
from datasets import load_dataset

from mcqa_data_load_all import letter_token_id


DATASET = "mib-bench/arc_challenge"
LETTERS = string.ascii_uppercase
FAMILIES = {
    "answer_pointer": "answerPosition_counterfactual",
    "answer_token": "randomLetter_counterfactual",
    "both": "answerPosition_randomLetter_counterfactual",
}


def load_pairs(split, max_rows=5):
    # dataset = load_dataset(DATASET, split=split)
    dataset = load_dataset(DATASET, split=split, streaming=True)
    pairs = []

    for i, row in enumerate(dataset):
        if max_rows is not None and i >= max_rows:
            break
        base_symbols = list(row["choices"]["label"])
        # if base_symbols != ["A", "B", "C", "D"]:
        #     continue

        base_pointer = int(row["answerKey"])

        for family, key in FAMILIES.items():
            source = row[key]
            source_symbols = list(source["choices"]["label"])
            if len(source_symbols) != 4:
                continue

            source_pointer = int(source["answerKey"])

            pairs.append({
                "family": family,
                "base_prompt": row["prompt"],
                "source_prompt": source["prompt"],
                "base_symbols": base_symbols,
                "source_symbols": source_symbols,
                "base_pointer": base_pointer,
                "source_pointer": source_pointer,
                "base_answer": base_symbols[base_pointer],
                "source_answer": source_symbols[source_pointer],
                "pointer_answer": base_symbols[source_pointer],
            })

    return pairs


@torch.no_grad()
def filter_correct(model, tokenizer, pairs, batch_size=16):
    device = next(model.parameters()).device
    kept = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]

        def correct(side):
            prompts = [x[f"{side}_prompt"] for x in batch]
            answers = [x[f"{side}_answer"] for x in batch]
            enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            pred = model(**enc, logits_to_keep=1).logits[:, -1].argmax(dim=-1)
            gold = torch.tensor([letter_token_id(tokenizer, x) for x in answers], device=device)
            return pred == gold

        ok = correct("base") & correct("source")
        kept.extend(pair for pair, keep in zip(batch, ok.tolist()) if keep)

    return kept


def build_bank(tokenizer, pairs, target):
    base = tokenizer([x["base_prompt"] for x in pairs], padding=True, return_tensors="pt")
    source = tokenizer([x["source_prompt"] for x in pairs], padding=True, return_tensors="pt")

    base_mask = base["attention_mask"].long()
    source_mask = source["attention_mask"].long()

    # base_answer = torch.tensor([LETTERS.index(x["base_answer"]) for x in pairs])
    # source_answer = torch.tensor([LETTERS.index(x["source_answer"]) for x in pairs])
    # source_pointer = torch.tensor([x["source_pointer"] for x in pairs])
    # pointer_answer = torch.tensor([LETTERS.index(x["pointer_answer"]) for x in pairs])
    base_answer = torch.tensor([LETTERS.index(x["base_answer"]) for x in pairs])
    source_answer = torch.tensor([LETTERS.index(x["source_answer"]) for x in pairs])
    base_pointer = torch.tensor([x["base_pointer"] for x in pairs])
    source_pointer = torch.tensor([x["source_pointer"] for x in pairs])
    pointer_answer = torch.tensor([LETTERS.index(x["pointer_answer"]) for x in pairs])

    # Keep the exact key names expected by the existing MCQA DAS code.
    answer_target = source_answer if target == "answer_token" else pointer_answer

    return {
        "base_input_ids": base["input_ids"].long(),
        "base_attention_mask": base_mask,
        "source_input_ids": source["input_ids"].long(),
        "source_attention_mask": source_mask,
        "base_position_by_id": {"last_token": base_mask.sum(dim=1) - 1},
        "source_position_by_id": {"last_token": source_mask.sum(dim=1) - 1},
        "base_answer_label_ids": base_answer,
        "source_answer_label_ids": source_answer,
        "counterfactual_label_ids": {
            "answer_pointer": source_pointer,
            "answer_token": answer_target,
        },
        "base_answer_pointer_ids": base_pointer,
        "source_answer_pointer_ids": source_pointer,
    }


def build_arc_banks(model, tokenizer, fit_size=400, cal_size=200, test_size=200, batch_size=16, seed=0):
    tokenizer.padding_side = "left"
    sizes = {"fit": fit_size, "cal": cal_size, "test": test_size}
    split_names = {"fit": "train", "cal": "validation", "test": "test"}

    filtered = {}
    for split, hf_split in split_names.items():
        pairs = load_pairs(hf_split)
        random.Random(f"{seed}:{split}").shuffle(pairs)
        filtered[split] = filter_correct(model, tokenizer, pairs, batch_size)
        print(f"[{split}] correct={len(filtered[split])}/{len(pairs)}")

    banks = {"answer_pointer": {}, "answer_token": {}}

    for target in banks:
        for split in ("fit", "cal", "test"):
            rows = filtered[split]

            if target == "answer_pointer":
                rows = [x for x in rows if x["source_pointer"] != x["base_pointer"]]
            else:
                rows = [x for x in rows if x["source_answer"] != x["base_answer"]]

            n = sizes[split]
            if len(rows) < n:
                raise ValueError(f"{target}/{split}: need {n}, only {len(rows)} available")

            banks[target][split] = build_bank(tokenizer, rows[:n], target)

    return banks

if __name__ == "__main__":
    pairs = load_pairs("train", max_rows=5)

    for x in pairs:
        print("family:", x["family"])
        print("base symbols:", x["base_symbols"])
        print("source symbols:", x["source_symbols"])
        print("base AP:", x["base_pointer"])
        print("source AP:", x["source_pointer"])
        print("source AT:", x["source_answer"])
        print("AT-after-AP:", x["pointer_answer"])
        print()