#!/usr/bin/env python3
"""
Generate grouped COLOR counterfactuals from:
    jchang153/copycolors_mcqa

For each original row:
1. Find the base color stated in the question.
2. Read the 4 answer-choice colors.
3. Keep the base prompt.
4. Replace the base color in the question with each of the other 3 choice colors.
5. Save ONE output row containing:
       base, cf1, cf2, cf3

The answer choices are unchanged across all four variants.

Later, pairwise examples can be constructed from these grouped rows.

Usage:
    python3 generate_color_bank.py

Optional:
    python3 generate_color_bank.py --output color_bank_grouped.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


DATASET_PATH = "jchang153/copycolors_mcqa"

COLOR_PATTERN = re.compile(
    r"\b(?:is|are)\s+(.+?)\.\s*What color\b",
    re.IGNORECASE,
)


def extract_question_color(prompt: str) -> str:
    """Extract the stated color from the question line."""
    first_line = prompt.splitlines()[0]

    match = COLOR_PATTERN.search(first_line)
    if match is None:
        raise ValueError(f"Cannot extract color from prompt: {first_line!r}")

    return match.group(1).strip()


def replace_question_color(prompt: str, new_color: str) -> str:
    """
    Replace ONLY the stated color in the question.
    Answer choices and the rest of the prompt stay unchanged.
    """
    lines = prompt.splitlines()
    first_line = lines[0]

    match = COLOR_PATTERN.search(first_line)
    if match is None:
        raise ValueError(f"Cannot extract color from prompt: {first_line!r}")

    start, end = match.span(1)

    lines[0] = (
        first_line[:start]
        + new_color
        + first_line[end:]
    )

    return "\n".join(lines)


def build_group(row, row_idx):
    """
    Convert one original dataset row into:
        base + 3 COLOR counterfactuals.
    """
    base_prompt = row["prompt"]

    labels = [str(x) for x in row["choices"]["label"]]
    colors = [str(x).strip() for x in row["choices"]["text"]]

    if len(labels) != 4 or len(colors) != 4:
        raise ValueError(
            f"Row {row_idx}: expected exactly 4 choices, "
            f"got labels={labels}, colors={colors}"
        )

    base_color = extract_question_color(base_prompt)

    matching_positions = [
        i
        for i, color in enumerate(colors)
        if color.casefold() == base_color.casefold()
    ]

    if len(matching_positions) != 1:
        raise ValueError(
            f"Row {row_idx}: base color {base_color!r} must occur exactly once "
            f"in choices {colors}"
        )

    base_position = matching_positions[0]

    # Start with the original/base variant.
    variants = [
        {
            "kind": "base",
            "color": base_color,
            "prompt": base_prompt,
            "answer_position": base_position,
            "answer": labels[base_position],
        }
    ]

    # Create the three color counterfactuals.
    for position, color in enumerate(colors):
        if position == base_position:
            continue

        cf_prompt = replace_question_color(
            base_prompt,
            color,
        )

        variants.append(
            {
                "kind": f"cf{len(variants)}",
                "color": color,
                "prompt": cf_prompt,
                "answer_position": position,
                "answer": labels[position],
            }
        )

    assert len(variants) == 4

    # Sanity check: the four variants should correspond exactly
    # to the four colors in the choices.
    variant_colors = {v["color"].casefold() for v in variants}
    choice_colors = {c.casefold() for c in colors}

    if variant_colors != choice_colors:
        raise ValueError(
            f"Row {row_idx}: variant colors do not match choice colors"
        )

    return {
        "base_row_id": f"jchang:train:{row_idx}",
        "choices": row["choices"],
        "variants": variants,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        default="color_bank_grouped.jsonl",
        help="Output JSONL file",
    )

    args = parser.parse_args()

    print(f"[load] {DATASET_PATH}")
    dataset = load_dataset(
        DATASET_PATH,
        split="train",
    )
    print(f"[load] rows={len(dataset)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_rows = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row_idx, row in enumerate(dataset):
            grouped = build_group(row, row_idx)

            f.write(
                json.dumps(grouped, ensure_ascii=False)
                + "\n"
            )

            num_rows += 1

    assert num_rows == len(dataset)

    print(f"[done] grouped rows={num_rows}")
    print(f"[done] variants per row=4")
    print(f"[done] saved to {output_path}")


if __name__ == "__main__":
    main()
