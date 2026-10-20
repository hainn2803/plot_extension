"""
Generate randomized AP / AT / C data and save it in a JChang-like row format.

Each generated row contains ONE randomized base plus four nested counterfactuals:

{
    "base_row_id": ...,
    "repeat_idx": ...,
    "prompt": ...,
    "choices": {"label": [...], "text": [...]},
    "color": ...,
    "answer_position": ...,
    "answer": ...,

    "answerPosition_counterfactual": {...},   # AP only
    "randomLetter_counterfactual": {...},     # AT only
    "color_counterfactual": {...},            # C only
    "color_all_counterfactual": {...},        # C + AP + AT
}

Each original JChang row is generated `repeats_per_row` times (default 2),
with a fresh set of 4 distinct A-Z labels each time.

Question-disjoint train / validation / test splitting is inherited directly
from the existing JChang split BEFORE any generation.
"""

import random
import re
import string
from pathlib import Path

from datasets import Dataset, DatasetDict

import mcqa_data_load_jchang_disjoint as jchang


LETTERS = string.ascii_uppercase

COLOR_PATTERN = re.compile(
    r"\b(?:is|are)\s+(.+?)\.\s*What color\b",
    re.IGNORECASE,
)

CHOICE_LINE = re.compile(
    r"(?m)^([ \t]*)([A-Z])\.[ \t]+([^\r\n]+)$"
)


# ---------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------

def question_color(prompt):
    first_line = prompt.splitlines()[0]
    match = COLOR_PATTERN.search(first_line)
    if match is None:
        raise ValueError(f"Cannot parse color from prompt: {first_line!r}")
    return match.group(1).strip().casefold()


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
    matches = list(CHOICE_LINE.finditer(prompt))

    if len(matches) != 4:
        raise ValueError(f"Expected 4 choice lines, got {len(matches)}")

    out = prompt

    for i in range(3, -1, -1):
        match = matches[i]
        indent = match.group(1)
        replacement = f"{indent}{labels[i]}. {texts[i]}"
        out = out[:match.start()] + replacement + out[match.end():]

    return out


def find_answer(texts, labels, color):
    color = str(color).strip().casefold()
    normalized = [str(x).strip().casefold() for x in texts]

    positions = [i for i, text in enumerate(normalized) if text == color]

    if len(positions) != 1:
        raise ValueError(
            f"Color {color!r} must occur exactly once in choices: {normalized}"
        )

    ap = positions[0]
    at = str(labels[ap])
    return ap, at


def make_state(prompt, labels, texts):
    color = question_color(prompt)
    ap, at = find_answer(texts, labels, color)

    return {
        "prompt": prompt,
        "choices": {
            "label": list(labels),
            "text": list(texts),
        },
        "color": color,
        "answer_position": int(ap),
        "answer": str(at),
    }


# ---------------------------------------------------------------------
# Base + four counterfactual types
# ---------------------------------------------------------------------

def make_randomized_base(row, rng):
    texts = [str(x).strip() for x in row["choices"]["text"]]
    labels = rng.sample(list(LETTERS), 4)
    # labels = rng.sample(list("ABCD"), 4)

    prompt = rewrite_choices(
        row["prompt"],
        labels,
        texts,
    )

    return make_state(prompt, labels, texts)


def make_ap_only(base, rng):
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])

    old_pos = base["answer_position"]
    new_pos = rng.choice([i for i in range(4) if i != old_pos])

    # Move the correct choice row together: AP changes, AT fixed.
    labels[old_pos], labels[new_pos] = labels[new_pos], labels[old_pos]
    texts[old_pos], texts[new_pos] = texts[new_pos], texts[old_pos]

    prompt = rewrite_choices(base["prompt"], labels, texts)
    cf = make_state(prompt, labels, texts)

    assert cf["color"] == base["color"]
    assert cf["answer_position"] != base["answer_position"]
    assert cf["answer"] == base["answer"]

    return cf


def make_at_only(base, rng):
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])

    correct_pos = base["answer_position"]
    used = set(labels)

    new_letter = rng.choice([
        letter for letter in LETTERS
        if letter not in used
    ])

    labels[correct_pos] = new_letter

    prompt = rewrite_choices(base["prompt"], labels, texts)
    cf = make_state(prompt, labels, texts)

    assert cf["color"] == base["color"]
    assert cf["answer_position"] == base["answer_position"]
    assert cf["answer"] != base["answer"]

    return cf


def choose_other_color(base, rng):
    candidates = [
        str(text).strip().casefold()
        for text in base["choices"]["text"]
        if str(text).strip().casefold() != base["color"]
    ]

    if len(candidates) != 3:
        raise ValueError("Expected exactly 3 alternative colors")

    return rng.choice(candidates)


def make_c_only(base, rng):
    """
    C changes; AP and AT stay fixed.
    """
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])

    new_color = choose_other_color(base, rng)

    old_correct_pos = base["answer_position"]
    new_color_old_pos, _ = find_answer(texts, labels, new_color)

    # Move only the choice texts so the new color lands at the same AP.
    texts[old_correct_pos], texts[new_color_old_pos] = (
        texts[new_color_old_pos],
        texts[old_correct_pos],
    )

    prompt = replace_question_color(base["prompt"], new_color)
    prompt = rewrite_choices(prompt, labels, texts)

    cf = make_state(prompt, labels, texts)

    assert cf["color"] != base["color"]
    assert cf["answer_position"] == base["answer_position"]
    assert cf["answer"] == base["answer"]

    return cf


def make_c_all_change(base, rng):
    """
    C changes; choices stay fixed, therefore AP and AT also change.
    """
    labels = list(base["choices"]["label"])
    texts = list(base["choices"]["text"])

    new_color = choose_other_color(base, rng)

    prompt = replace_question_color(base["prompt"], new_color)
    cf = make_state(prompt, labels, texts)

    assert cf["color"] != base["color"]
    assert cf["answer_position"] != base["answer_position"]
    assert cf["answer"] != base["answer"]

    return cf


# ---------------------------------------------------------------------
# JChang-like generated rows
# ---------------------------------------------------------------------

def generate_rows_for_jchang_row(
    row,
    row_idx,
    split,
    *,
    repeats_per_row=2,
    seed=0,
):
    """
    One original JChang row -> `repeats_per_row` generated JChang-like rows.
    """
    rng = random.Random(seed * 1_000_003 + int(row_idx))

    generated = []
    used_labelings = set()

    for repeat_idx in range(repeats_per_row):
        # Fresh randomized A-Z labels for every repeat.
        for _ in range(100):
            base = make_randomized_base(row, rng)
            labeling = tuple(base["choices"]["label"])

            if labeling not in used_labelings:
                used_labelings.add(labeling)
                break
        else:
            raise RuntimeError("Could not sample fresh labels")

        ap_cf = make_ap_only(base, rng)
        at_cf = make_at_only(base, rng)
        c_only_cf = make_c_only(base, rng)
        c_all_cf = make_c_all_change(base, rng)

        generated.append({
            "base_row_id": f"jchang:train:{row_idx}",
            "repeat_idx": repeat_idx,
            "split": split,
            "question_key": jchang.question_key(row),

            # Base state.
            "prompt": base["prompt"],
            "choices": base["choices"],
            "color": base["color"],
            "answer_position": base["answer_position"],
            "answer": base["answer"],

            # Same naming spirit as the original JChang dataset.
            "answerPosition_counterfactual": ap_cf,
            "randomLetter_counterfactual": at_cf,

            # New COLOR fields.
            "color_counterfactual": c_only_cf,
            "color_all_counterfactual": c_all_cf,
        })

    return generated


def build_dataset_dict(
    *,
    seed=0,
    repeats_per_row=2,
):
    """
    Build a HuggingFace DatasetDict with train / validation / test splits.
    """
    dataset, split_indices = jchang.load_question_splits(seed=seed)

    generated_splits = {}

    for split in ("train", "validation", "test"):
        rows = []

        for row_idx in split_indices[split]:
            rows.extend(
                generate_rows_for_jchang_row(
                    dataset[int(row_idx)],
                    int(row_idx),
                    split,
                    repeats_per_row=repeats_per_row,
                    seed=seed,
                )
            )

        generated_splits[split] = Dataset.from_list(rows)

        print(
            f"[generated] {split}: "
            f"parent_rows={len(split_indices[split])} "
            f"generated_rows={len(rows)}"
        )

    # Final disjointness check.
    qsets = {
        split: set(generated_splits[split]["question_key"])
        for split in generated_splits
    }

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = qsets[left] & qsets[right]
        if overlap:
            raise ValueError(
                f"Question leakage {left}/{right}: {sorted(overlap)[:5]}"
            )

    print("[generated] train/validation/test questions are disjoint")

    return DatasetDict(generated_splits)


# ---------------------------------------------------------------------
# Save / push
# ---------------------------------------------------------------------

def save_to_disk(
    output_dir,
    *,
    seed=0,
    repeats_per_row=2,
):
    ds = build_dataset_dict(
        seed=seed,
        repeats_per_row=repeats_per_row,
    )

    output_dir = str(Path(output_dir))
    ds.save_to_disk(output_dir)

    print(f"[saved] {output_dir}")
    return ds


def push_to_hub(
    repo_id,
    *,
    seed=0,
    repeats_per_row=2,
    private=False,
):
    ds = build_dataset_dict(
        seed=seed,
        repeats_per_row=repeats_per_row,
    )

    ds.push_to_hub(
        repo_id,
        private=private,
    )

    print(f"[pushed] {repo_id}")
    return ds


# ---------------------------------------------------------------------
# CLI sanity / local save
# ---------------------------------------------------------------------

if __name__ == "__main__":
    save_to_disk(
        "randomized_copycolors",
        seed=0,
        repeats_per_row=2,
    )

    # print(ds)
    # print("\nExample row:")
    # print(ds["train"][0])
