import os
import random
import re
import torch
from datasets import load_dataset
from collections import defaultdict
from mcqa_constants import HF_TOKEN, DATASET_ORIGINAL_PATH, DATASET_ORIGINAL_NAME, LETTER_TO_CLASS, CF_FAMILIES
from mcqa_neural_net import load_gemma_model
import re

# The only place to add or remove pairs. Each entry maps a pair name to
# (base row field, source row field); "original" means the row itself.
# Keep the three original keys for consumers that inspect source families.
PAIR_SPECS = {
    # "answerPosition_counterfactual": (
    #     "original", "answerPosition_counterfactual"
    # ),
    # "randomLetter_counterfactual": (
    #     "original", "randomLetter_counterfactual"
    # ),
    # "answerPosition_randomLetter_counterfactual": (
    #     "original", "answerPosition_randomLetter_counterfactual"
    "answerColor_counterfactual": (
        "original", "color_counterfactual"
    ),
}
COLOR_IN_QUESTION = re.compile(
    r"\b(?:is|are)\s+(.+?)\.\s*What color\b",
    re.IGNORECASE,
)

COLORS = [
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
]

COLOR_TO_CLASS = {
    color: i
    for i, color in enumerate(COLORS)
}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_TOKEN"] = HF_TOKEN


# create functions to find position of specific tokens: "?", color

def extract_color(prompt):
    first_line = prompt.splitlines()[0]
    match = COLOR_IN_QUESTION.search(first_line)
    if match is None:
        raise ValueError(f"Cannot find color in prompt: {first_line!r}")
    return match.group(1).strip().lower()

def ap_after_color_interchange(pair):
    source_color = pair["source_C"]
    base_choice_texts = [str(x).strip().lower() for x in pair["base_choices"]["text"]]
    if source_color not in base_choice_texts:
        raise ValueError(f"Source color {source_color!r} not found in base choices")
    return base_choice_texts.index(source_color)

def color_position(prompt, tokenizer):
    first_line = prompt.splitlines()[0]
    match = COLOR_IN_QUESTION.search(first_line)
    if match is None:
        raise ValueError(f"Cannot find color in prompt: {first_line!r}")
    prefix = first_line[:match.end(1)]
    return len(tokenizer(prefix, add_special_tokens=True)["input_ids"]) - 1

def symbol_position(prompt, letter, tokenizer):
    match = re.search(rf"(?m)^[ \t]*{re.escape(letter)}\.", prompt)
    if match is None:
        raise ValueError(f"Cannot find choice {letter!r} in prompt")
    prefix = prompt[:match.end() - 1]
    unpadded_position = len(tokenizer(prefix, add_special_tokens=True)["input_ids"]) - 1
    # collect_layer_states adds the left-padding offset at run time.
    return unpadded_position

def question_mark_position(prompt, tokenizer):
    first_line = prompt.splitlines()[0]
    q_idx = first_line.rfind("?")
    if q_idx == -1:
        raise ValueError(f"Cannot find '?' in prompt: {first_line!r}")
    prefix = first_line[:q_idx + 1]
    return len(tokenizer(prefix, add_special_tokens=True)["input_ids"]) - 1


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
    pairs_by_cf_family = {family: [] for family in PAIR_SPECS}
    for row_idx, row in enumerate(dataset):
        for pair_family, (base_family, source_family) in PAIR_SPECS.items():
            base = row if base_family == "original" else row[base_family]
            source = row if source_family == "original" else row[source_family]
            base_AP = int(base["answerKey"])
            base_AT = str(base["choices"]["label"][base_AP])
            source_AP = int(source["answerKey"])
            source_AT = str(source["choices"]["label"][source_AP])
            base_C = extract_color(base["prompt"])
            source_C = extract_color(source["prompt"])

            pairs_by_cf_family[pair_family].append({
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
                "base_C": base_C,
                "source_C": source_C,
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
    elif target == "C":
        pairs = [p for p in pairs if p["base_C"] != p["source_C"]]
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


    # metric for readout
    after_ap = [
        str(p["base_choices"]["label"][p["source_AP"]])
        for p in pairs
    ]
    # after_c_ap = [
    #     ap_after_color_interchange(p)
    #     for p in pairs
    # ]
    # after_c_at = [
    #     str(p["base_choices"]["label"][ap])
    #     for p, ap in zip(pairs, after_c_ap)
    # ]



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
    
    # symbol answer position
    base_symbol = torch.tensor([
        symbol_position(p["base_prompt"], p["base_AT"], tokenizer)
        for p in pairs
    ], dtype=torch.long)
    source_symbol = torch.tensor([
        symbol_position(p["source_prompt"], p["source_AT"], tokenizer)
        for p in pairs
    ], dtype=torch.long)

    # color token position
    base_color = torch.tensor([
        color_position(p["base_prompt"], tokenizer)
        for p in pairs
    ], dtype=torch.long)

    source_color = torch.tensor([
        color_position(p["source_prompt"], tokenizer)
        for p in pairs
    ], dtype=torch.long)

    # question mark position
    base_question_mark = torch.tensor([
        question_mark_position(p["base_prompt"], tokenizer)
        for p in pairs
    ], dtype=torch.long)

    source_question_mark = torch.tensor([
        question_mark_position(p["source_prompt"], tokenizer)
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
            "color": base_color,
            "question_mark": base_question_mark,
            "last_token": base_tokens["attention_mask"].sum(dim=1).long() - 1
        },
        "source_position_by_id": {
            "correct_symbol": source_symbol,
            "correct_symbol_period": source_symbol + 1,
            "color": source_color,
            "question_mark": source_question_mark,
            "last_token": source_tokens["attention_mask"].sum(dim=1).long() - 1
        },
        "base_answer_label_ids": torch.tensor([
            LETTER_TO_CLASS[p["base_AT"]] for p in pairs
        ], dtype=torch.long),
        "base_answer_pointer_ids": torch.tensor([
            p["base_AP"] for p in pairs
        ], dtype=torch.long),
        "source_answer_pointer_ids": torch.tensor([
            p["source_AP"] for p in pairs
        ], dtype=torch.long),
        "base_color_ids": torch.tensor([
            COLOR_TO_CLASS[p["base_C"]]
            for p in pairs
        ], dtype=torch.long),

        "source_color_ids": torch.tensor([
            COLOR_TO_CLASS[p["source_C"]]
            for p in pairs
        ], dtype=torch.long),
        "counterfactual_label_ids": {

            "answer_token": torch.tensor([
                LETTER_TO_CLASS[p["source_AT"]] for p in pairs
            ], dtype=torch.long),

            # when intervene on AP
            "answer_pointer": torch.tensor([
                p["source_AP"] for p in pairs
            ], dtype=torch.long),
            "answer_token_after_pointer_interchange": torch.tensor([
                LETTER_TO_CLASS[letter] for letter in after_ap
            ], dtype=torch.long),

            # when intervene on C
            "color": torch.tensor([
                COLOR_TO_CLASS[p["source_C"]]
                for p in pairs
            ], dtype=torch.long),

            # "answer_pointer_after_color_interchange": torch.tensor(
            #     after_c_ap,
            #     dtype=torch.long,
            # ),

            # "answer_token_after_color_interchange": torch.tensor([
            #     LETTER_TO_CLASS[letter]
            #     for letter in after_c_at
            # ], dtype=torch.long),

        },
        "changed_mask": {
            "answer_pointer": torch.tensor([
                p["base_AP"] != p["source_AP"] for p in pairs
            ], dtype=torch.bool),
            "answer_token": torch.tensor([
                p["base_AT"] != p["source_AT"] for p in pairs
            ], dtype=torch.bool),
            "color": torch.tensor([
                p["base_C"] != p["source_C"]
                for p in pairs
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
            # for target in ("AP", "AT", "C")
            for target in ("C")
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
        "C": (
        banks["train"]["C"],
        {"color": banks["validation"]["C"]},
        {"color": banks["test"]["C"]},
    ),
    }


if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    tokenizer.padding_side = "left"

    data = build_all_mcqa_banks(
        model=model,
        tokenizer=tokenizer,
        ft_size=5,
        cal_size=5,
        te_size=5,
        batch_size=2,
        seed=0,
    )

    # Lấy C train bank
    c_train_bank = data["C"][0]

    pair = c_train_bank["pairs"][0]

    print("\n===== PAIR =====")
    print("Base prompt:")
    print(pair["base_prompt"])

    print("\nSource prompt:")
    print(pair["source_prompt"])

    print("\n===== VARIABLES =====")
    print("base_C:", pair["base_C"])
    print("source_C:", pair["source_C"])

    print("base_AP:", pair["base_AP"])
    print("source_AP:", pair["source_AP"])

    print("base_AT:", pair["base_AT"])
    print("source_AT:", pair["source_AT"])

    print("\n===== C TARGETS =====")
    print(
        "color class:",
        c_train_bank["counterfactual_label_ids"]["color"][0].item()
    )

    print(
        "AP after C:",
        c_train_bank["counterfactual_label_ids"][
            "answer_pointer_after_color_interchange"
        ][0].item()
    )

    at_after_c_id = c_train_bank["counterfactual_label_ids"][
        "answer_token_after_color_interchange"
    ][0].item()

    print("AT after C class:", at_after_c_id)
    print("AT after C letter:", list(LETTER_TO_CLASS.keys())[at_after_c_id])

    print("\n===== POSITIONS =====")
    print(
        "base color pos:",
        c_train_bank["base_position_by_id"]["color"][0].item()
    )
    print(
        "source color pos:",
        c_train_bank["source_position_by_id"]["color"][0].item()
    )
    print(
        "base question mark pos:",
        c_train_bank["base_position_by_id"]["question_mark"][0].item()
    )
    print(
        "source question mark pos:",
        c_train_bank["source_position_by_id"]["question_mark"][0].item()
    )