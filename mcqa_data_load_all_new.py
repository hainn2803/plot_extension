import os
import random
import re
from functools import lru_cache
import torch
from datasets import load_dataset
from collections import defaultdict

from mcqa_constants import DATASET_PATH, HF_TOKEN, DATASET_ORIGINAL_PATH, DATASET_ORIGINAL_NAME, LETTER_TO_CLASS, CF_FAMILIES
from mcqa_neural_net import load_gemma_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_TOKEN"] = HF_TOKEN



def check_disjoint_questions_within_dataset(dataset):
    rows_by_question = defaultdict(list)
    for i, row in enumerate(dataset):
        rows_by_question[row["question"]].append(i)

    duplicates = {q: ids for q, ids in rows_by_question.items() if len(ids) > 1}

    print("num rows:", len(dataset))
    print("num unique questions:", len(rows_by_question))
    print("num duplicate questions:", len(duplicates))


def check_disjoint_questions_cross_dataset(dataset_train, dataset_validation, dataset_test):
    splits = {
        "train": dataset_train,
        "validation": dataset_validation,
        "test": dataset_test,
    }
    for dataset in splits.values():
        check_disjoint_questions_within_dataset(dataset)
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        for i, ques1 in enumerate(splits[left]):
            for j, ques2 in enumerate(splits[right]):
                if ques1["question"] == ques2["question"]:
                    print(f"Overlap found: {ques1['question']}")
                    print(ques1["prompt"])
                    print(ques2["prompt"])
                    print("********")



def load_mcqa_datasets(split):
    dataset = load_dataset(path=DATASET_ORIGINAL_PATH, name=DATASET_ORIGINAL_NAME, split=split, cache_dir=CACHE_DIR)

    pairs_by_cf_family = {family: [] for family in CF_FAMILIES}

    for row_idx, row in enumerate(dataset):
        base_AP = int(row["answerKey"])
        base_AT = str(row["choices"]["label"][base_AP])
        base_Y = base_AT

        for family_cf in CF_FAMILIES:
            source = row[family_cf]
            source_AP = int(source["answerKey"])
            source_AT = str(source["choices"]["label"][source_AP])
            source_Y = source_AT

            pairs_by_cf_family[family_cf].append({
                "split": split,
                "base_row_id": f"{split}:{row_idx}",
                "question": row["question"],
                "source_family": family_cf,

                "base_prompt": row["prompt"],
                "base_choices": row["choices"],
                "base_AP": base_AP,
                "base_AT": base_AT,
                "base_Y": base_Y,

                "source_prompt": source["prompt"],
                "source_choices": source["choices"],
                "source_AP": source_AP,
                "source_AT": source_AT,
                "source_Y": source_Y,
            })
    
    return pairs_by_cf_family, dataset


@torch.no_grad()
def predict_correct(model, tokenizer, pair_examples, batch_size=32, device="cuda"):
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gom prompt trùng để Gemma chỉ chấm một lần.
    gold_by_prompt = {}
    for family_cf in CF_FAMILIES:
        for pair in pair_examples[family_cf]:
            for side in ("base", "source"):
                prompt = pair[f"{side}_prompt"]
                gold = str(pair[f"{side}_Y"])

                if prompt in gold_by_prompt and gold_by_prompt[prompt] != gold:
                    raise ValueError("Same prompt has different gold answers")
                gold_by_prompt[prompt] = gold

    gold_ids_by_answer = {}
    for gold in set(gold_by_prompt.values()):
        valid_ids = set()
        for text in (gold, " " + gold):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                valid_ids.add(ids[0])

        if not valid_ids:
            raise ValueError(f"No single-token encoding for answer {gold!r}")
        gold_ids_by_answer[gold] = valid_ids

    prompts = list(gold_by_prompt)
    correct_by_prompt = {}

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        encoding = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)

        predicted_ids = model(**encoding, logits_to_keep=1).logits[:, -1].argmax(dim=-1).tolist()

        for prompt, predicted_id in zip(batch_prompts, predicted_ids):
            gold = gold_by_prompt[prompt]
            correct_by_prompt[prompt] = (predicted_id in gold_ids_by_answer[gold])

    filtered = {}
    for family_cf in CF_FAMILIES:
        filtered[family_cf] = [
            pair for pair in pair_examples[family_cf]
            if correct_by_prompt[pair["base_prompt"]]
            and correct_by_prompt[pair["source_prompt"]]
        ]
        print(f"{family_cf}: {len(filtered[family_cf])}/{len(pair_examples[family_cf])}")

    return filtered




def create_bank(filtered_pairs, target, tokenizer, size=None, seed=0):
    pairs = [
        pair
        for family_pairs in filtered_pairs.values()
        for pair in family_pairs
    ]

    if target == "AP":
        pairs = [p for p in pairs if p["base_AP"] != p["source_AP"]]
    elif target == "AT":
        pairs = [p for p in pairs if p["base_AT"] != p["source_AT"]]
    else:
        raise ValueError(f"Unknown target: {target}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    if size is not None:
        pairs = pairs[:size]

    # Code train/readout hiện tại dùng 26 classes A–Z.
    for p in pairs:
        if p["base_AT"] not in LETTER_TO_CLASS or p["source_AT"] not in LETTER_TO_CLASS:
            raise ValueError(
                f"Family {p['source_family']} has a label outside A–Z. "
                "The current AT readout cannot use numeric answer labels."
            )

    after_ap = [
        str(p["base_choices"]["label"][p["source_AP"]])
        for p in pairs
    ]

    tokenizer.padding_side = "left"
    base_tokens = tokenizer(
        [p["base_prompt"] for p in pairs],
        padding=True,
        return_tensors="pt",
    )
    source_tokens = tokenizer(
        [p["source_prompt"] for p in pairs],
        padding=True,
        return_tensors="pt",
    )

    base_mask = base_tokens["attention_mask"].long()
    source_mask = source_tokens["attention_mask"].long()
    n = len(pairs)
    base_width = base_tokens["input_ids"].shape[1]
    source_width = source_tokens["input_ids"].shape[1]

    def symbol_position(prompt, letter, width, attention_mask):
        match = re.search(
            rf"(?m)^[ \t]*{re.escape(letter)}\.",
            prompt,
        )
        if match is None:
            raise ValueError(f"Cannot find choice {letter!r} in prompt")

        # Prefix kết thúc ngay sau chữ cái, trước dấu chấm.
        prefix = prompt[:match.end() - 1]
        unpadded_position = len(
            tokenizer(prefix, add_special_tokens=True)["input_ids"]
        ) - 1
        left_padding = width - int(attention_mask.sum())
        return left_padding + unpadded_position

    base_symbol = torch.tensor([
        symbol_position(p["base_prompt"], p["base_AT"], base_width, base_mask[i])
        for i, p in enumerate(pairs)
    ], dtype=torch.long)

    source_symbol = torch.tensor([
        symbol_position(p["source_prompt"], p["source_AT"], source_width, source_mask[i])
        for i, p in enumerate(pairs)
    ], dtype=torch.long)

    return {
        "pairs": pairs,
        "pair_source_families": [p["source_family"] for p in pairs],

        "base_input_ids": base_tokens["input_ids"].long(),
        "base_attention_mask": base_mask,
        "source_input_ids": source_tokens["input_ids"].long(),
        "source_attention_mask": source_mask,

        "base_position_by_id": {
            "correct_symbol": base_symbol,
            "correct_symbol_period": base_symbol + 1,
            "last_token": base_tokens["attention_mask"].sum(dim=1).long() - 1
        },
        "source_position_by_id": {
            "correct_symbol": source_symbol,
            "correct_symbol_period": source_symbol + 1,
            "last_token": source_tokens["attention_mask"].sum(dim=1).long() - 1
        },

        # Đây là class index, KHÔNG phải Gemma vocabulary token ID.
        "base_answer_label_ids": torch.tensor([
            LETTER_TO_CLASS[p["base_AT"]] for p in pairs
        ], dtype=torch.long),
        "base_answer_pointer_ids": torch.tensor([
            p["base_AP"] for p in pairs
        ], dtype=torch.long),
        "source_answer_pointer_ids": torch.tensor([
            p["source_AP"] for p in pairs
        ], dtype=torch.long),

        "counterfactual_label_ids": {
            "answer_pointer": torch.tensor([
                p["source_AP"] for p in pairs
            ], dtype=torch.long),
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



def build_all_mcqa_banks(
    model, tokenizer, ft_size, cal_size, te_size,
    batch_size=32, seed=0,
):
    device = next(model.parameters()).device
    banks = {}

    for split, size in (
        ("train", ft_size),
        ("validation", cal_size),
        ("test", te_size),
    ):
        pairs_by_family, _ = load_mcqa_datasets(split)

        filtered_pairs = predict_correct(
            model=model,
            tokenizer=tokenizer,
            pair_examples=pairs_by_family,
            batch_size=batch_size,
            device=device,
        )

        banks[split] = {
            target: create_bank(
                filtered_pairs, target, tokenizer,
                size=size, seed=seed,
            )
            for target in ("AP", "AT")
        }

        for target, bank in banks[split].items():
            print(f"[bank] {split} {target}: {len(bank['base_input_ids'])} pairs")

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
    }

    
if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    pairs_by_family, _ = load_mcqa_datasets(split="train")
    filtered = predict_correct(model, tokenizer, pairs_by_family)

