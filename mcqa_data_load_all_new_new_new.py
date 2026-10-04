import os
import random
import re
import string
import torch
from datasets import load_dataset
from collections import defaultdict
from mcqa_constants import HF_TOKEN, DATASET_ORIGINAL_PATH, DATASET_ORIGINAL_NAME, LETTER_TO_CLASS, CF_FAMILIES
from mcqa_neural_net import load_gemma_model
# The only place to add or remove pairs. Each entry maps a pair name to
# (base row field, source row field); "original" means the row itself.
# Keep the three original keys for consumers that inspect source families.
PAIR_SPECS = {
    "answerPosition_counterfactual": (
        "original", "answerPosition_counterfactual"
    ),
    "randomLetter_counterfactual": (
        "original", "randomLetter_counterfactual"
    ),
    "answerPosition_randomLetter_counterfactual": (
        "original", "answerPosition_randomLetter_counterfactual"
    ),
    # "randomLetter_to_answerPosition_randomLetter": (
    #     "randomLetter_counterfactual", "answerPosition_randomLetter_counterfactual"
    # ),
    # "answerPosition_to_answerPosition_randomLetter": (
    #     "answerPosition_counterfactual", "answerPosition_randomLetter_counterfactual"
    ),
}
# Total versions of each pair: the original plus three label-remapped copies.
LABEL_VARIANTS_PER_PAIR = 4
CHOICE_LINE = re.compile(r"(?m)^([ \t]*)([A-Z])(\.[ \t]+)")
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


def relabel_pair(pair, rng, row_idx, family_idx, variant_idx, copies):
    """Relabel choices consistently in base and source; leave colors/positions intact."""
    alphabet = string.ascii_uppercase
    base_labels = [str(x) for x in pair["base_choices"]["label"]]
    source_labels = [str(x) for x in pair["source_choices"]["label"]]
    old_labels = set(base_labels + source_labels)
    if len(base_labels) != 4 or len(source_labels) != 4:
        raise ValueError("Expected four choices on both sides of a pair")
    if not old_labels.issubset(set(alphabet)):
        raise ValueError(f"Choice labels must be A-Z: {old_labels}")

    mapping = {}
    used = set()

    def assign(old, start):
        if old in mapping:
            return
        for offset in range(26):
            new = alphabet[(start + offset) % 26]
            if new not in used and new != old:
                mapping[old] = new
                used.add(new)
                return
        raise ValueError("Could not assign distinct answer labels")

    # Rotate the correct label through A-Z across rows/copies, then spread
    # the source's correct label when it is a different original letter.
    start = (row_idx * copies + variant_idx + family_idx * 7) % 26
    assign(pair["base_AT"], start)
    assign(pair["source_AT"], (start + 13) % 26)
    remaining = list(alphabet)
    rng.shuffle(remaining)
    for old in sorted(old_labels - mapping.keys()):
        new = next(letter for letter in remaining if letter not in used and letter != old)
        mapping[old] = new
        used.add(new)

    augmented = pair.copy()
    augmented["label_variant"] = variant_idx
    for side, old_side_labels in (("base", base_labels), ("source", source_labels)):
        def replace_choice(match):
            old = match.group(2)
            if old in old_side_labels:
                seen.append(old)
                return match.group(1) + mapping[old] + match.group(3)
            return match.group(0)

        seen = []
        prompt = CHOICE_LINE.sub(replace_choice, pair[f"{side}_prompt"])
        if sorted(seen) != sorted(old_side_labels):
            raise ValueError(f"Cannot locate all {side} choice labels in prompt")
        choices = dict(pair[f"{side}_choices"])
        choices["label"] = [mapping[old] for old in old_side_labels]
        answer = choices["label"][pair[f"{side}_AP"]]
        augmented[f"{side}_prompt"] = prompt
        augmented[f"{side}_choices"] = choices
        augmented[f"{side}_AT"] = answer
        augmented[f"{side}_Y"] = answer
    return augmented


def load_mcqa_datasets(split, variants_per_pair=LABEL_VARIANTS_PER_PAIR, seed=0):
    if variants_per_pair < 1:
        raise ValueError("variants_per_pair must be at least 1")
    dataset = load_dataset(path=DATASET_ORIGINAL_PATH, name=DATASET_ORIGINAL_NAME, split=split, cache_dir=CACHE_DIR)
    pairs_by_cf_family = {family: [] for family in PAIR_SPECS}
    rng = random.Random(seed)
    for row_idx, row in enumerate(dataset):
        for family_idx, (pair_family, (base_family, source_family)) in enumerate(PAIR_SPECS.items()):
            base = row if base_family == "original" else row[base_family]
            source = row if source_family == "original" else row[source_family]
            base_AP = int(base["answerKey"])
            base_AT = str(base["choices"]["label"][base_AP])
            source_AP = int(source["answerKey"])
            source_AT = str(source["choices"]["label"][source_AP])
            pair = {
                "split": split,
                "base_row_id": f"{split}:{row_idx}",
                "question": row["question"],
                "source_family": source_family,
                "base_family": base_family,
                "pair_family": pair_family,
                "base_prompt": base["prompt"],
                "base_choices": base["choices"],
                "base_AP": base_AP,
                "base_AT": base_AT,
                "base_Y": base_AT,
                "source_prompt": source["prompt"],
                "source_choices": source["choices"],
                "source_AP": source_AP,
                "source_AT": source_AT,
                "source_Y": source_AT,
                "label_variant": 0,
            }
            pairs_by_cf_family[pair_family].append(pair)
            for variant_idx in range(1, variants_per_pair):
                pairs_by_cf_family[pair_family].append(relabel_pair(
                    pair, rng, row_idx, family_idx, variant_idx,
                    variants_per_pair - 1,
                ))
    return pairs_by_cf_family, dataset


@torch.no_grad()


def predict_correct(model, tokenizer, pair_examples, batch_size=32, device="cuda"):
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gom prompt trùng để Gemma chỉ chấm một lần.
    gold_by_prompt = {}
    for family_cf, family_pairs in pair_examples.items():
        for pair in family_pairs:
            for side in ("base", "source"):
                prompt = pair[f"{side}_prompt"]
                gold = str(pair[f"{side}_Y"])
                if prompt in gold_by_prompt and gold_by_prompt[prompt] != gold:
                    raise ValueError("Same prompt has different gold answers")
                gold_by_prompt[prompt] = gold
    prompts = list(gold_by_prompt)
    correct_by_prompt = {}
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        encoding = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        generated = model.generate(
            **encoding,
            max_new_tokens=1,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        output_texts = tokenizer.batch_decode(
            generated.sequences[:, -1:], skip_special_tokens=True
        )
        for prompt, output_text in zip(batch_prompts, output_texts):
            correct_by_prompt[prompt] = (" " + gold_by_prompt[prompt]) in output_text
    filtered = {}
    for family_cf, family_pairs in pair_examples.items():
        filtered[family_cf] = [
            pair for pair in family_pairs
            if correct_by_prompt[pair["base_prompt"]]
            and correct_by_prompt[pair["source_prompt"]]
        ]
        print(f"{family_cf}: {len(filtered[family_cf])}/{len(family_pairs)}")
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
    def symbol_position(prompt, letter):
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
        # collect_layer_states adds the left-padding offset at run time.
        return unpadded_position
    base_symbol = torch.tensor([
        symbol_position(p["base_prompt"], p["base_AT"])
        for p in pairs
    ], dtype=torch.long)
    source_symbol = torch.tensor([
        symbol_position(p["source_prompt"], p["source_AT"])
        for p in pairs
    ], dtype=torch.long)
    return {
        "pairs": pairs,
        "pair_source_families": [p["source_family"] for p in pairs],
        "pair_families": [p["pair_family"] for p in pairs],
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
        pairs_by_family, _ = load_mcqa_datasets(split, seed=seed)
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
