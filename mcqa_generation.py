"""
Generate randomized MCQA data for the causal model:

    C -> AP -> AT -> Y

C  = queried color
AP = position of the correct color
AT = label at that position
Y  = AT

For each base example:
- AP_only: C same, AP changes, AT same
- AT_only: C same, AP same, AT changes
- C_only: C changes, AP same, AT same
- C_AP_AT: C changes, choices stay fixed, so AP and AT also change

AT_only uses a mixture controlled by distractor_portion:
- distractor_swap: swap the correct label with one distractor label
- novel_label: replace the correct label with a fresh A-Z label not already used

C_only and C_AP_AT use the same counterfactual color.
"""

import argparse
import random
import re
import string
from pathlib import Path

from datasets import Dataset, DatasetDict

import mcqa_data_load_jchang_disjoint as jchang


LETTERS = string.ascii_uppercase
COLOR_PATTERN = re.compile(r"\b(?:is|are)\s+(.+?)\.\s*What color\b", re.IGNORECASE)
CHOICE_PATTERN = re.compile(r"(?m)^([ \t]*)([A-Z])\.[ \t]+([^\r\n]+)$")
NUM_CHOICES = 4


def normalize(text):
    return str(text).strip().casefold()


def get_question_color(prompt):
    first_line = prompt.splitlines()[0]
    match = COLOR_PATTERN.search(first_line)

    if match is None:
        raise ValueError(f"Cannot parse color from prompt: {first_line!r}")

    return normalize(match.group(1))


def replace_question_color(prompt, new_color):
    lines = prompt.splitlines(keepends=True)
    first_line = lines[0]
    match = COLOR_PATTERN.search(first_line)

    if match is None:
        raise ValueError(f"Cannot parse color from prompt: {first_line!r}")

    start, end = match.span(1)
    lines[0] = first_line[:start] + str(new_color) + first_line[end:]
    return "".join(lines)


def rewrite_choices(prompt, labels, texts):
    matches = list(CHOICE_PATTERN.finditer(prompt))

    if len(matches) != NUM_CHOICES:
        raise ValueError(f"Expected {NUM_CHOICES} choice lines, got {len(matches)}")

    result = prompt

    for i in range(NUM_CHOICES - 1, -1, -1):
        match = matches[i]
        indent = match.group(1)
        new_line = f"{indent}{labels[i]}. {texts[i]}"
        result = result[:match.start()] + new_line + result[match.end():]

    return result


def find_color_position(texts, color):
    color = normalize(color)
    positions = []

    for i in range(len(texts)):
        if normalize(texts[i]) == color:
            positions.append(i)

    if len(positions) != 1:
        raise ValueError(f"Expected color {color!r} exactly once, got {positions}")

    return positions[0]


def make_state(prompt, labels, texts):
    color = get_question_color(prompt)
    answer_position = find_color_position(texts, color)
    answer = str(labels[answer_position])

    return {
        "prompt": prompt,
        "choices": {"label": list(labels), "text": list(texts)},
        "color": color,
        "answer_position": answer_position,
        "answer": answer,
    }


def make_base(row, rng):
    texts = []

    for text in row["choices"]["text"]:
        texts.append(str(text).strip())

    labels = rng.sample(list(LETTERS), NUM_CHOICES)
    prompt = rewrite_choices(row["prompt"], labels, texts)
    return make_state(prompt, labels, texts)


def sample_new_ap(base, rng):
    candidates = []
    base_ap = base["answer_position"]

    for position in range(NUM_CHOICES):
        if position != base_ap:
            candidates.append(position)

    return rng.choice(candidates)


def sample_novel_at(base, rng):
    candidates = []
    used_labels = set(base["choices"]["label"])

    for letter in LETTERS:
        if letter not in used_labels:
            candidates.append(letter)

    return rng.choice(candidates)


def sample_distractor_at(base, rng):
    candidates = []
    ap = int(base["answer_position"])

    for position in range(NUM_CHOICES):
        if position != ap:
            candidates.append(str(base["choices"]["label"][position]))

    return rng.choice(candidates)


def sample_new_color(base, rng):
    candidates = []

    for text in base["choices"]["text"]:
        color = normalize(text)

        if color != base["color"]:
            candidates.append(color)

    if len(candidates) != NUM_CHOICES - 1:
        raise ValueError(f"Expected {NUM_CHOICES - 1} alternative colors, got {candidates}")

    return rng.choice(candidates)


def make_ap_counterfactual(base, new_ap):
    """Change AP while keeping C and AT fixed."""
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])
    old_ap = base["answer_position"]

    labels[old_ap], labels[new_ap] = labels[new_ap], labels[old_ap]
    texts[old_ap], texts[new_ap] = texts[new_ap], texts[old_ap]

    prompt = rewrite_choices(base["prompt"], labels, texts)
    cf = make_state(prompt, labels, texts)

    assert cf["color"] == base["color"]
    assert cf["answer_position"] == new_ap
    assert cf["answer"] == base["answer"]

    return cf


def make_at_counterfactual(base, new_at, use_distractor):
    """Change AT while keeping C and AP fixed."""
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])
    ap = int(base["answer_position"])

    if use_distractor:
        distractor_position = None

        for position in range(NUM_CHOICES):
            if position != ap and labels[position] == new_at:
                distractor_position = position
                break

        if distractor_position is None:
            raise ValueError(f"Distractor label {new_at!r} not found")

        labels[ap], labels[distractor_position] = labels[distractor_position], labels[ap]
    else:
        if new_at in labels:
            raise ValueError(f"Novel AT {new_at!r} already appears in labels {labels}")

        labels[ap] = new_at

    prompt = rewrite_choices(base["prompt"], labels, texts)
    cf = make_state(prompt, labels, texts)

    assert cf["color"] == base["color"]
    assert cf["answer_position"] == base["answer_position"]
    assert cf["answer"] == new_at
    assert cf["answer"] != base["answer"]

    if use_distractor:
        assert set(cf["choices"]["label"]) == set(base["choices"]["label"])
    else:
        assert new_at not in set(base["choices"]["label"])

    return cf


def make_c_counterfactual(base, new_color):
    """Change C while keeping AP and AT fixed."""
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])

    old_ap = base["answer_position"]
    new_color_position = find_color_position(texts, new_color)

    texts[old_ap], texts[new_color_position] = texts[new_color_position], texts[old_ap]

    prompt = replace_question_color(base["prompt"], new_color)
    prompt = rewrite_choices(prompt, labels, texts)
    cf = make_state(prompt, labels, texts)

    assert cf["color"] == normalize(new_color)
    assert cf["answer_position"] == base["answer_position"]
    assert cf["answer"] == base["answer"]

    return cf


def make_c_all_counterfactual(base, new_color):
    """Change C and leave choices unchanged, so AP and AT naturally change."""
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])

    prompt = replace_question_color(base["prompt"], new_color)
    cf = make_state(prompt, labels, texts)

    assert cf["color"] == normalize(new_color)
    assert cf["answer_position"] != base["answer_position"]
    assert cf["answer"] != base["answer"]

    return cf


def make_ap_at_counterfactual(base, new_ap):
    """Change AP and let AT change naturally, while keeping C fixed."""
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])
    old_ap = base["answer_position"]

    texts[old_ap], texts[new_ap] = texts[new_ap], texts[old_ap]

    prompt = rewrite_choices(base["prompt"], labels, texts)
    cf = make_state(prompt, labels, texts)

    assert cf["color"] == base["color"]
    assert cf["answer_position"] == new_ap
    assert cf["answer"] != base["answer"]

    return cf


def generate_rows_for_jchang_row(row, row_idx, split, repeats_per_row=2, seed=0, distractor_portion=0.5):
    if distractor_portion < 0.0 or distractor_portion > 1.0:
        raise ValueError(f"distractor_portion must be in [0, 1], got {distractor_portion}")

    rng = random.Random(seed * 1_000_003 + int(row_idx))
    generated = []
    used_labelings = set()

    for repeat_idx in range(repeats_per_row):
        for _ in range(100):
            base = make_base(row, rng)
            labeling = tuple(base["choices"]["label"])

            if labeling not in used_labelings:
                used_labelings.add(labeling)
                break
        else:
            raise RuntimeError("Could not sample a fresh base labeling")

        new_ap = sample_new_ap(base, rng)
        new_color = sample_new_color(base, rng)

        use_distractor = rng.random() < distractor_portion

        if use_distractor:
            new_at = sample_distractor_at(base, rng)
            at_counterfactual_type = "distractor_swap"
        else:
            new_at = sample_novel_at(base, rng)
            at_counterfactual_type = "novel_label"

        ap_cf = make_ap_counterfactual(base, new_ap)
        ap_at_cf = make_ap_at_counterfactual(base, new_ap)
        at_cf = make_at_counterfactual(base, new_at, use_distractor)
        c_cf = make_c_counterfactual(base, new_color)
        c_all_cf = make_c_all_counterfactual(base, new_color)

        generated.append({
            "base_row_id": f"jchang:train:{row_idx}",
            "repeat_idx": repeat_idx,
            "split": split,
            "question_key": jchang.question_key(row),
            "prompt": base["prompt"],
            "choices": base["choices"],
            "color": base["color"],
            "answer_position": base["answer_position"],
            "answer": base["answer"],
            "answerPosition_counterfactual": ap_cf,
            "randomLetter_counterfactual": at_cf,
            "at_counterfactual_type": at_counterfactual_type,
            "color_counterfactual": c_cf,
            "color_position_letter_counterfactual": c_all_cf,
            "answerPosition_letter_counterfactual": ap_at_cf,
        })

    return generated


def build_dataset_dict(seed=0, repeats_per_row=2, distractor_portion=0.5):
    dataset, split_indices = jchang.load_question_splits(seed=seed)
    generated_splits = {}

    for split in ("train", "validation", "test"):
        rows = []

        for row_idx in split_indices[split]:
            generated = generate_rows_for_jchang_row(dataset[int(row_idx)], int(row_idx), split, repeats_per_row, seed, distractor_portion)
            rows.extend(generated)

        generated_splits[split] = Dataset.from_list(rows)

        distractor_count = 0
        novel_count = 0

        for row in rows:
            if row["at_counterfactual_type"] == "distractor_swap":
                distractor_count += 1
            else:
                novel_count += 1

        print(f"[generated] {split}: original_rows={len(split_indices[split])} generated_rows={len(rows)} distractor={distractor_count} novel={novel_count}")

    train_questions = set(generated_splits["train"]["question_key"])
    validation_questions = set(generated_splits["validation"]["question_key"])
    test_questions = set(generated_splits["test"]["question_key"])

    assert train_questions.isdisjoint(validation_questions)
    assert train_questions.isdisjoint(test_questions)
    assert validation_questions.isdisjoint(test_questions)

    print("[generated] train/validation/test questions are disjoint")
    return DatasetDict(generated_splits)


def save_to_disk(output_dir="randomized_copycolors", seed=0, repeats_per_row=2, distractor_portion=0.5):
    dataset = build_dataset_dict(seed, repeats_per_row, distractor_portion)
    output_dir = str(Path(output_dir))
    dataset.save_to_disk(output_dir)
    print(f"[saved] {output_dir}")
    return dataset


def push_to_hub(repo_id, seed=0, repeats_per_row=2, distractor_portion=0.5, private=False):
    dataset = build_dataset_dict(seed, repeats_per_row, distractor_portion)
    dataset.push_to_hub(repo_id, private=private)
    print(f"[pushed] {repo_id}")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats-per-row", type=int, default=2)
    parser.add_argument("--distractor-portion", type=float, default=0.5)
    args = parser.parse_args()

    save_to_disk(args.output_dir, args.seed, args.repeats_per_row, args.distractor_portion)


if __name__ == "__main__":
    main()
