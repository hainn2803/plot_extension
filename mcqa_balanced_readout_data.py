"""Build question-disjoint, balanced factual data for AT or AP readouts.

Example, after loading Gemma and its tokenizer:

    train, val = build_readout_datasets(
        model, tokenizer, target="AT", examples_per_class=5, seed=0,
    )
    # train["input_ids"], train["attention_mask"],
    # train["position_by_id"], train["target_ids"]

AT classes are A-Z (indices 0-25); AP classes are positions 0-3. Each
sample is one prompt. The model must answer that prompt correctly before
the sample enters either readout dataset.
"""

import random
import re
import string

import torch
from datasets import load_dataset

from mcqa_constants import DATASET_ORIGINAL_NAME, DATASET_ORIGINAL_PATH


LETTERS = string.ascii_uppercase
CHOICE_LINE = re.compile(r"(?m)^([ \t]*)([A-Z])\.[ \t]+([^\r\n]+)$")


def _choices(row):
    matches = list(CHOICE_LINE.finditer(row["prompt"]))
    if len(matches) != 4:
        raise ValueError("Expected four labeled choice lines in the prompt")
    labels = [m.group(2) for m in matches]
    expected = [str(x) for x in row["choices"]["label"]]
    if labels != expected or len(set(labels)) != 4:
        raise ValueError("Prompt labels differ from choices.label")
    answer_position = int(row["answerKey"])
    if answer_position not in range(4):
        raise ValueError("answerKey must be a position from 0 to 3")
    return matches, [m.group(3) for m in matches], answer_position


def _make_example(row, target, requested_class, rng):
    matches, colors, old_position = _choices(row)
    position = requested_class if target == "AP" else rng.randrange(4)
    colors[old_position], colors[position] = colors[position], colors[old_position]

    if target == "AT":
        correct_letter = LETTERS[requested_class]
        other_letters = rng.sample([x for x in LETTERS if x != correct_letter], 3)
        labels = other_letters[:]
        labels.insert(position, correct_letter)
    else:
        labels = rng.sample(LETTERS, 4)
        correct_letter = labels[position]

    # Replace only the four choice lines. Question text and Answer: stay intact.
    prompt = row["prompt"]
    for i in range(3, -1, -1):
        m = matches[i]
        line = f"{m.group(1)}{labels[i]}. {colors[i]}"
        prompt = prompt[:m.start()] + line + prompt[m.end():]
    return {
        "prompt": prompt,
        "answer_letter": correct_letter,
        "answer_position": position,
        "target_id": requested_class,
        "question": row["question"],
    }


@torch.no_grad()
def _model_correct(model, tokenizer, examples, device, batch_size):
    """Check the generated next token against the displayed gold letter."""
    result = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        tokens = tokenizer(
            [item["prompt"] for item in batch], padding=True,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **tokens,
            max_new_tokens=1,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        texts = tokenizer.batch_decode(generated[:, -1:], skip_special_tokens=True)
        result.extend(
            text.strip() == item["answer_letter"]
            for item, text in zip(batch, texts)
        )
    return result


def _symbol_position(prompt, letter, tokenizer):
    match = re.search(rf"(?m)^[ \t]*{re.escape(letter)}\.", prompt)
    if match is None:
        raise ValueError(f"Could not locate correct label {letter!r}")
    prefix = prompt[:match.end() - 1]
    return len(tokenizer(prefix, add_special_tokens=True)["input_ids"]) - 1


def _tokenize(examples, tokenizer):
    tokens = tokenizer(
        [item["prompt"] for item in examples],
        padding=True, return_tensors="pt",
    )
    symbol = torch.tensor([
        _symbol_position(item["prompt"], item["answer_letter"], tokenizer)
        for item in examples
    ], dtype=torch.long)
    return {
        "prompts": [item["prompt"] for item in examples],
        "questions": [item["question"] for item in examples],
        "answer_letters": [item["answer_letter"] for item in examples],
        "answer_positions": torch.tensor(
            [item["answer_position"] for item in examples], dtype=torch.long,
        ),
        "target_ids": torch.tensor(
            [item["target_id"] for item in examples], dtype=torch.long,
        ),
        "input_ids": tokens["input_ids"].long(),
        "attention_mask": tokens["attention_mask"].long(),
        "position_by_id": {
            "last_token": tokens["attention_mask"].sum(dim=1).long() - 1,
            "correct_symbol": symbol,
            "correct_symbol_period": symbol + 1,
        },
    }


def build_readout_datasets(
    model, tokenizer, target, examples_per_class=5, *,
    val_examples_per_class=None,
    train_split="train", val_split="validation", batch_size=32,
    max_attempts_per_class=100, seed=0, cache_dir=None,
):
    """Return balanced (train, val) prompt banks for one AP or AT readout.

    Training uses examples_per_class; validation uses val_examples_per_class
    (or the same quota when omitted). Raise an error if the attempt budget runs
    out; never silently return an imbalanced bank. The source splits must have
    disjoint question strings. No counterfactual pairs are constructed here.
    """
    if target not in ("AP", "AT"):
        raise ValueError("target must be 'AP' or 'AT'")
    if examples_per_class < 1 or max_attempts_per_class < 1 or batch_size < 1:
        raise ValueError("counts and batch_size must be positive")
    if val_examples_per_class is not None and val_examples_per_class < 1:
        raise ValueError("val_examples_per_class must be positive")
    if train_split == val_split:
        raise ValueError("train_split and val_split must differ")

    datasets = {
        name: load_dataset(
            path=DATASET_ORIGINAL_PATH, name=DATASET_ORIGINAL_NAME,
            split=name, cache_dir=cache_dir,
        )
        for name in (train_split, val_split)
    }
    train_questions = {row["question"] for row in datasets[train_split]}
    val_questions = {row["question"] for row in datasets[val_split]}
    overlap = train_questions & val_questions
    if overlap:
        raise ValueError(f"Train/validation questions overlap: {len(overlap)}")

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    classes = range(26 if target == "AT" else 4)
    banks = {}

    for split_index, split_name in enumerate((train_split, val_split)):
        rows = datasets[split_name]
        quota = (examples_per_class if split_index == 0 or val_examples_per_class is None
                 else val_examples_per_class)
        if len(rows) == 0:
            raise ValueError(f"Empty dataset split: {split_name}")
        rng = random.Random(seed + split_index)
        accepted = {i: [] for i in classes}
        seen_prompts = set()

        for _ in range(max_attempts_per_class):
            proposals = []
            for class_id in classes:
                if len(accepted[class_id]) >= quota:
                    continue
                # Try new row/label combinations without scoring duplicates.
                for _ in range(20):
                    item = _make_example(rows[rng.randrange(len(rows))], target, class_id, rng)
                    if item["prompt"] not in seen_prompts:
                        seen_prompts.add(item["prompt"])
                        proposals.append(item)
                        break
            if not proposals:
                break
            for item, correct in zip(
                proposals, _model_correct(model, tokenizer, proposals, device, batch_size)
            ):
                class_id = item["target_id"]
                if correct and len(accepted[class_id]) < quota:
                    accepted[class_id].append(item)
            if all(len(items) == quota for items in accepted.values()):
                break

        missing = {
            (LETTERS[i] if target == "AT" else i): quota - len(items)
            for i, items in accepted.items() if len(items) < quota
        }
        if missing:
            raise RuntimeError(
                f"{split_name}: insufficient model-correct {target} prompts "
                f"after {max_attempts_per_class} attempts per class: {missing}"
            )
        examples = [item for class_id in classes for item in accepted[class_id]]
        rng.shuffle(examples)
        banks[split_name] = _tokenize(examples, tokenizer)
        print(f"[readout {target}] {split_name}: "
              f"{quota} per class, {len(examples)} prompts")

    return banks[train_split], banks[val_split]
