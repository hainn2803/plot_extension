"""
Loader/bank builder for grouped COLOR counterfactual dataset.

Expected input:
    color_bank_grouped.jsonl

Each JSONL row should contain:
    {
        "base_row_id": ...,
        "choices": ...,
        "variants": [
            {"kind": "base", "color": ..., "prompt": ..., "answer_position": ..., "answer": ...},
            {"kind": "cf1", ...},
            {"kind": "cf2", ...},
            {"kind": "cf3", ...},
        ]
    }

Pipeline:
1. Load grouped COLOR rows.
2. Split rows question-disjoint into train / validation / test.
3. Expand every 4-state group into all ordered pairwise interventions:
       4 * 3 = 12 pairs per group.
4. Keep only pairs where the model answers BOTH prompts correctly.
5. Tokenize the selected pairs into a DAS-ready bank.

This mirrors the structure of the existing MCQA AP/AT loader, but is
specialized for the new COLOR variable.
"""

import json
import os
import random
import re
import string
from collections import defaultdict
from pathlib import Path

import torch


COLOR_DATA_PATH = Path(
    os.environ.get(
        "COLOR_BANK_PATH",
        Path(__file__).resolve().parent / "color_bank_grouped.jsonl",
    )
)

CACHE_DIR = Path(
    os.environ.get(
        "MCQA_CACHE_DIR",
        Path(__file__).resolve().parent / "hf_cache",
    )
)

COLOR_PATTERN = re.compile(
    r"\b(?:is|are)\s+(.+?)\.\s*What color\b",
    re.IGNORECASE,
)

LETTERS = string.ascii_uppercase
LETTER_TO_CLASS = {letter: i for i, letter in enumerate(LETTERS)}



def question_mark_position(prompt, tokenizer):
    first_line = prompt.splitlines()[0]

    q_idx = first_line.rfind("?")
    if q_idx == -1:
        raise ValueError(f"Cannot locate question mark in prompt: {first_line!r}")

    # prefix kết thúc đúng tại dấu ?
    prefix = first_line[:q_idx + 1]

    return len(
        tokenizer(
            prefix,
            add_special_tokens=True,
        )["input_ids"]
    ) - 1



# def color_position(prompt, tokenizer):
#     first_line = prompt.splitlines()[0]

#     match = COLOR_PATTERN.search(first_line)
#     if match is None:
#         raise ValueError(f"Cannot locate color in prompt: {first_line!r}")

#     # Prefix kết thúc ngay trước color
#     prefix = first_line[:match.start(1)]

#     # Token đầu tiên của color
#     return len(
#         tokenizer(
#             prefix,
#             add_special_tokens=True,
#         )["input_ids"]
#     )


def color_position(prompt, tokenizer):
    first_line = prompt.splitlines()[0]

    match = COLOR_PATTERN.search(first_line)
    if match is None:
        raise ValueError(
            f"Cannot locate color in prompt: {first_line!r}"
        )

    color_start = match.start(1)

    encoded = tokenizer(
        first_line,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]

    for token_idx, (start, end) in enumerate(offsets):
        if start <= color_start < end:
            return token_idx

    raise ValueError(
        f"Cannot map color to token position: "
        f"color={match.group(1)!r}, prompt={first_line!r}"
    )


def load_color_groups(path=COLOR_DATA_PATH):
    """Load grouped COLOR JSONL rows."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"COLOR dataset not found: {path}\n"
            f"Generate it first or set COLOR_BANK_PATH."
        )

    groups = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            variants = row.get("variants")
            if not isinstance(variants, list) or len(variants) != 4:
                raise ValueError(
                    f"Line {line_no}: expected exactly 4 variants"
                )

            prompts = [v["prompt"] for v in variants]
            colors = [str(v["color"]).strip().casefold() for v in variants]

            if len(set(prompts)) != 4:
                raise ValueError(
                    f"Line {line_no}: variants must have 4 distinct prompts"
                )

            if len(set(colors)) != 4:
                raise ValueError(
                    f"Line {line_no}: variants must have 4 distinct colors"
                )

            groups.append(row)

    print(f"[color load] {path}: {len(groups)} grouped rows")
    return groups


def question_key(group):
    """
    Same spirit as the original MCQA loader:
    use the full BASE question, excluding choices.

    All four COLOR variants in one group remain together automatically.
    """
    prompt = group["variants"][0]["prompt"]
    first_line = prompt.splitlines()[0].strip()

    if first_line.lower().startswith("question:"):
        first_line = first_line[len("question:"):].strip()

    if "?" not in first_line:
        raise ValueError(f"No question mark in prompt: {first_line!r}")

    return " ".join(first_line.split("?", 1)[0].split()).casefold() + "?"


def split_question_indices(
    groups,
    seed=0,
    train_fraction=0.7,
    cal_fraction=0.15,
):
    """Assign each complete base question and all repeated rows to one split."""
    if not (
        0 < train_fraction < 1
        and 0 < cal_fraction < 1 - train_fraction
    ):
        raise ValueError(
            "Expected 0 < train_fraction, cal_fraction and sum < 1"
        )

    grouped = defaultdict(list)

    for index, group in enumerate(groups):
        grouped[question_key(group)].append(index)

    questions = list(grouped)
    random.Random(seed).shuffle(questions)

    train_limit = int(len(groups) * train_fraction)
    cal_limit = int(len(groups) * cal_fraction)

    indices = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for question in questions:
        if len(indices["train"]) < train_limit:
            split = "train"
        elif len(indices["validation"]) < cal_limit:
            split = "validation"
        else:
            split = "test"

        indices[split].extend(grouped[question])

    return indices


def check_disjoint_questions(groups, indices):
    """Verify train / validation / test have no base-question overlap."""
    question_sets = {}

    for split in ("train", "validation", "test"):
        question_sets[split] = {
            question_key(groups[i])
            for i in indices[split]
        }

        print(
            f"[color split] {split}: "
            f"rows={len(indices[split])} "
            f"questions={len(question_sets[split])}"
        )

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = question_sets[left] & question_sets[right]

        if overlap:
            raise ValueError(
                f"Question overlap between {left} and {right}: "
                f"{sorted(overlap)[:5]}"
            )

    print("[color split] train/validation/test questions are disjoint")


def load_color_splits(
    seed=0,
    train_fraction=0.7,
    cal_fraction=0.15,
    path=COLOR_DATA_PATH,
):
    """
    Load grouped COLOR dataset and cache row-index splits.
    """
    groups = load_color_groups(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    path = Path(path)
    stat = path.stat()

    manifest_path = CACHE_DIR / (
        f"color_question_split_seed{seed}_"
        f"train{train_fraction}_cal{cal_fraction}.json"
    )

    expected = {
        "data_path": str(path.resolve()),
        "num_rows": len(groups),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "seed": seed,
        "train_fraction": train_fraction,
        "cal_fraction": cal_fraction,
    }

    indices = None

    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())

        if all(
            saved.get(key) == value
            for key, value in expected.items()
        ):
            indices = saved["indices"]
            print(f"[color split] reusing {manifest_path}")

    if indices is None:
        indices = split_question_indices(
            groups,
            seed=seed,
            train_fraction=train_fraction,
            cal_fraction=cal_fraction,
        )

        tmp_path = manifest_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    **expected,
                    "indices": indices,
                }
            )
        )
        tmp_path.replace(manifest_path)

        print(f"[color split] saved {manifest_path}")

    all_indices = [
        i
        for split in ("train", "validation", "test")
        for i in indices[split]
    ]

    if (
        len(all_indices) != len(groups)
        or set(all_indices) != set(range(len(groups)))
    ):
        raise ValueError(
            "COLOR split does not cover every grouped row exactly once"
        )

    check_disjoint_questions(groups, indices)

    return groups, indices


def pairwise_pairs_for_indices(groups, indices, split):
    """
    Expand each 4-state COLOR group into ALL ordered pairs.

    For variants v0,v1,v2,v3:
        v0 -> v1
        v0 -> v2
        ...
        v3 -> v2

    Total = 4 * 3 = 12 pairs per grouped row.
    """
    pairs = []

    for row_idx in indices:
        group = groups[int(row_idx)]
        variants = group["variants"]

        if len(variants) != 4:
            raise ValueError(
                f"Row {row_idx}: expected 4 variants"
            )

        for base_idx, base in enumerate(variants):
            for source_idx, source in enumerate(variants):
                if base_idx == source_idx:
                    continue

                if (
                    str(base["color"]).strip().casefold()
                    == str(source["color"]).strip().casefold()
                ):
                    raise ValueError(
                        f"Row {row_idx}: pair does not change color"
                    )

                pair = {
                    "split": split,
                    "base_row_id": group["base_row_id"],
                    "question": question_key(group),

                    "base_variant": base["kind"],
                    "source_variant": source["kind"],

                    "base_prompt": base["prompt"],
                    "source_prompt": source["prompt"],

                    "base_choices": group["choices"],
                    "source_choices": group["choices"],

                    # COLOR intervention itself.
                    "base_color": base["color"],
                    "source_color": source["color"],

                    # Downstream answer variables.
                    "base_AP": int(base["answer_position"]),
                    "source_AP": int(source["answer_position"]),

                    "base_AT": str(base["answer"]),
                    "source_AT": str(source["answer"]),

                    "base_Y": str(base["answer"]),
                    "source_Y": str(source["answer"]),
                }

                pairs.append(pair)

    expected = 12 * len(indices)

    if len(pairs) != expected:
        raise RuntimeError(
            f"Expected {expected} pairwise pairs, got {len(pairs)}"
        )

    return pairs


def load_color_dataset(split, seed=0, path=COLOR_DATA_PATH):
    """
    Convenience loader.

    Returns:
        pairs, grouped_rows_for_split
    """
    groups, indices = load_color_splits(
        seed=seed,
        path=path,
    )

    if split not in indices:
        raise ValueError(f"Unknown split: {split}")

    pairs = pairwise_pairs_for_indices(
        groups,
        indices[split],
        split,
    )

    subset = [
        groups[i]
        for i in indices[split]
    ]

    return pairs, subset


@torch.no_grad()
def predict_correct(
    model,
    tokenizer,
    pairs,
    batch_size=32,
    device=None,
    correct_by_prompt=None,
):
    """
    Keep pairs where full-vocabulary next-token argmax equals the gold
    answer letter for BOTH base and source.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model.eval()
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = device or next(model.parameters()).device

    if correct_by_prompt is None:
        correct_by_prompt = {}

    gold_by_prompt = {}

    for pair in pairs:
        for side in ("base", "source"):
            prompt = pair[f"{side}_prompt"]
            gold = pair[f"{side}_Y"]

            if (
                prompt in gold_by_prompt
                and gold_by_prompt[prompt] != gold
            ):
                raise ValueError(
                    "Identical prompts have conflicting answers"
                )

            gold_by_prompt[prompt] = gold

    unknown = [
        prompt
        for prompt in gold_by_prompt
        if prompt not in correct_by_prompt
    ]

    gold_token_ids = {}

    for letter in set(gold_by_prompt.values()):
        variants = set()

        for spelling in (" " + letter, letter):
            ids = tokenizer(
                spelling,
                add_special_tokens=False,
            )["input_ids"]

            if len(ids) == 1:
                variants.add(int(ids[0]))

        if not variants:
            raise ValueError(
                f"No single-token encoding for answer {letter!r}"
            )

        gold_token_ids[letter] = variants

    for start in range(0, len(unknown), batch_size):
        prompts = unknown[start:start + batch_size]

        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        ).to(device)

        position_ids = (
            inputs["attention_mask"]
            .long()
            .cumsum(-1)
            .sub(1)
            .clamp(min=0)
        )

        try:
            outputs = model(
                **inputs,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=1,
            )
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise

            outputs = model(
                **inputs,
                position_ids=position_ids,
                use_cache=False,
            )

        predicted = (
            outputs.logits[:, -1]
            .argmax(dim=-1)
            .tolist()
        )

        for prompt, token_id in zip(prompts, predicted):
            text = tokenizer.decode(
                [int(token_id)]
            ).strip()

            gold = gold_by_prompt[prompt]

            correct_by_prompt[prompt] = (
                token_id in gold_token_ids[gold]
                or text == gold
            )

    filtered = [
        pair
        for pair in pairs
        if correct_by_prompt[pair["base_prompt"]]
        and correct_by_prompt[pair["source_prompt"]]
    ]

    return filtered


def _symbol_position(prompt, letter, tokenizer):
    match = re.search(
        rf"(?m)^[ \t]*{re.escape(letter)}\.",
        prompt,
    )

    if match is None:
        raise ValueError(
            f"Cannot locate choice label {letter!r} in prompt"
        )

    prefix = prompt[:match.end() - 1]

    return (
        len(
            tokenizer(
                prefix,
                add_special_tokens=True,
            )["input_ids"]
        )
        - 1
    )


def create_color_bank(
    pairs,
    tokenizer,
    size=None,
    seed=0,
):
    """
    Convert model-correct COLOR pairs into a DAS-ready tokenized bank.

    Every pair already changes COLOR, so no additional changed-mask
    filtering is required.
    """
    pairs = list(pairs)

    random.Random(seed).shuffle(pairs)

    if size is not None:
        if len(pairs) < size:
            raise ValueError(
                f"COLOR: requested {size} pairs "
                f"but only {len(pairs)} passed"
            )

        pairs = pairs[:size]

    if not pairs:
        raise ValueError(
            "COLOR: no model-correct pairs"
        )

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

    base_symbol = torch.tensor(
        [
            _symbol_position(
                p["base_prompt"],
                p["base_AT"],
                tokenizer,
            )
            for p in pairs
        ],
        dtype=torch.long,
    )

    source_symbol = torch.tensor(
        [
            _symbol_position(
                p["source_prompt"],
                p["source_AT"],
                tokenizer,
            )
            for p in pairs
        ],
        dtype=torch.long,
    )

    base_color_pos = torch.tensor([
        color_position(p["base_prompt"], tokenizer)
        for p in pairs
    ], dtype=torch.long)

    source_color_pos = torch.tensor([
        color_position(p["source_prompt"], tokenizer)
        for p in pairs
    ], dtype=torch.long)


    # for i in range(min(20, len(pairs))):
    #     base_ids = tokenizer(
    #         pairs[i]["base_prompt"],
    #         add_special_tokens=True,
    #     )["input_ids"]

    #     source_ids = tokenizer(
    #         pairs[i]["source_prompt"],
    #         add_special_tokens=True,
    #     )["input_ids"]

    #     bpos = base_color_pos[i].item()
    #     spos = source_color_pos[i].item()

    #     print("\n=== COLOR TOKEN CHECK ===")

    #     print(
    #         "BASE",
    #         "| color =", pairs[i]["base_color"],
    #         "| pos =", bpos,
    #         "| token =", repr(tokenizer.decode([base_ids[bpos]])),
    #     )

    #     print(
    #         "SOURCE",
    #         "| color =", pairs[i]["source_color"],
    #         "| pos =", spos,
    #         "| token =", repr(tokenizer.decode([source_ids[spos]])),
    #     )


    base_question_mark_pos = torch.tensor([
        question_mark_position(p["base_prompt"], tokenizer)
        for p in pairs
    ], dtype=torch.long)

    source_question_mark_pos = torch.tensor([
        question_mark_position(p["source_prompt"], tokenizer)
        for p in pairs
    ], dtype=torch.long)

    return {
        "pairs": pairs,

        "base_input_ids": base_tokens["input_ids"].long(),
        "base_attention_mask": base_tokens["attention_mask"].long(),

        "source_input_ids": source_tokens["input_ids"].long(),
        "source_attention_mask": source_tokens["attention_mask"].long(),

        "base_position_by_id": {
            "last_token":
                base_tokens["attention_mask"].sum(dim=1).long() - 1,
            "correct_symbol":
                base_symbol,
            "correct_symbol_period":
                base_symbol + 1,
            "color": base_color_pos,
            "question_mark": base_question_mark_pos,
        },

        "source_position_by_id": {
            "last_token":
                source_tokens["attention_mask"].sum(dim=1).long() - 1,
            "correct_symbol":
                source_symbol,
            "correct_symbol_period":
                source_symbol + 1,
            "color": source_color_pos,
            "question_mark": source_question_mark_pos,
        },

        "base_answer_label_ids": torch.tensor(
            [
                LETTER_TO_CLASS[p["base_AT"]]
                for p in pairs
            ],
            dtype=torch.long,
        ),

        "base_answer_pointer_ids": torch.tensor(
            [
                p["base_AP"]
                for p in pairs
            ],
            dtype=torch.long,
        ),

        "source_answer_pointer_ids": torch.tensor(
            [
                p["source_AP"]
                for p in pairs
            ],
            dtype=torch.long,
        ),

        # After a COLOR interchange, the correct model output should
        # be the answer letter associated with the SOURCE color.
        "counterfactual_label_ids": {
            "color": torch.tensor(
                [
                    LETTER_TO_CLASS[p["source_Y"]]
                    for p in pairs
                ],
                dtype=torch.long,
            ),

            # Aliases useful if existing evaluation code expects these names.
            "answer_pointer": torch.tensor(
                [
                    p["source_AP"]
                    for p in pairs
                ],
                dtype=torch.long,
            ),

            "answer_token": torch.tensor(
                [
                    LETTER_TO_CLASS[p["source_AT"]]
                    for p in pairs
                ],
                dtype=torch.long,
            ),
        },

        "changed_mask": {
            "color": torch.ones(
                len(pairs),
                dtype=torch.bool,
            ),
        },
    }


def build_all_color_banks(
    model,
    tokenizer,
    ft_size,
    cal_size,
    te_size,
    batch_size=32,
    seed=0,
    path=COLOR_DATA_PATH,
):
    """
    Build train / validation / test COLOR banks.

    Each grouped row first becomes 12 ordered pairwise COLOR interventions.
    Only model-correct base/source pairs are retained.
    """
    groups, split_indices = load_color_splits(
        seed=seed,
        path=path,
    )

    device = next(model.parameters()).device
    correct_cache = {}

    banks = {}

    for offset, (split, size) in enumerate(
        (
            ("train", ft_size),
            ("validation", cal_size),
            ("test", te_size),
        )
    ):
        row_indices = list(split_indices[split])

        random.Random(
            seed + offset
        ).shuffle(row_indices)

        kept = []
        rows_scored = 0

        # Score incrementally instead of materializing all pairs at once.
        for start in range(0, len(row_indices), 128):
            current = row_indices[start:start + 128]

            candidate = pairwise_pairs_for_indices(
                groups,
                current,
                split,
            )

            filtered = predict_correct(
                model,
                tokenizer,
                candidate,
                batch_size=batch_size,
                device=device,
                correct_by_prompt=correct_cache,
            )

            kept.extend(filtered)
            rows_scored += len(current)

            if size is not None and len(kept) >= size:
                break

        bank = create_color_bank(
            kept,
            tokenizer,
            size=size,
            seed=seed + offset,
        )

        banks[split] = bank

        print(
            f"[color bank] {split}: "
            f"{len(bank['pairs'])} pairs "
            f"(available={len(kept)}, "
            f"rows_scored={rows_scored})"
        )

    selected_questions = {
        split: {
            p["question"]
            for p in banks[split]["pairs"]
        }
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        assert selected_questions[left].isdisjoint(
            selected_questions[right]
        ), f"{left}/{right} question overlap"

    return (
        banks["train"],
        {"color": banks["validation"]},
        {"color": banks["test"]},
    )


if __name__ == "__main__":
    groups, indices = load_color_splits(seed=0)

    row_idx = indices["train"][0]

    pairs = pairwise_pairs_for_indices(
        groups,
        [row_idx],
        split="train",
    )

    print("num_pairs:", len(pairs))

    from mcqa_neural_net import load_gemma_model

    _, tokenizer = load_gemma_model()

    print("\n" + "#" * 80)
    print("QUESTION MARK POSITION SANITY CHECK")
    print("#" * 80)

    for i, pair in enumerate(pairs[:10]):
        base_ids = tokenizer(
            pair["base_prompt"],
            add_special_tokens=True,
        )["input_ids"]

        source_ids = tokenizer(
            pair["source_prompt"],
            add_special_tokens=True,
        )["input_ids"]

        bpos = question_mark_position(
            pair["base_prompt"],
            tokenizer,
        )

        spos = question_mark_position(
            pair["source_prompt"],
            tokenizer,
        )

        print("\n" + "=" * 70)
        print(f"PAIR {i}")

        print(
            "BASE:",
            "| pos =", bpos,
            "| token =", repr(tokenizer.decode([base_ids[bpos]])),
        )

        print("BASE context:")
        for j in range(
            max(0, bpos - 3),
            min(len(base_ids), bpos + 4),
        ):
            marker = "<<< QUESTION MARK" if j == bpos else ""
            print(
                j,
                repr(tokenizer.decode([base_ids[j]])),
                marker,
            )

        print(
            "\nSOURCE:",
            "| pos =", spos,
            "| token =", repr(tokenizer.decode([source_ids[spos]])),
        )

        print("SOURCE context:")
        for j in range(
            max(0, spos - 3),
            min(len(source_ids), spos + 4),
        ):
            marker = "<<< QUESTION MARK" if j == spos else ""
            print(
                j,
                repr(tokenizer.decode([source_ids[j]])),
                marker,
            )


#     # ------------------------------------------------------------
#     # COLOR POSITION SANITY CHECK
#     # ------------------------------------------------------------
#     for i, pair in enumerate(pairs[:10]):
#         base_ids = tokenizer(
#             pair["base_prompt"],
#             add_special_tokens=True,
#         )["input_ids"]

#         source_ids = tokenizer(
#             pair["source_prompt"],
#             add_special_tokens=True,
#         )["input_ids"]

#         bpos = color_position(
#             pair["base_prompt"],
#             tokenizer,
#         )

#         spos = color_position(
#             pair["source_prompt"],
#             tokenizer,
#         )

#         print("\n" + "=" * 70)
#         print(f"PAIR {i}")

#         print(
#             "BASE:",
#             "color =", pair["base_color"],
#             "| pos =", bpos,
#             "| token =", repr(
#                 tokenizer.decode([base_ids[bpos]])
#             ),
#         )

#         print("BASE context:")
#         for j in range(
#             max(0, bpos - 3),
#             min(len(base_ids), bpos + 4),
#         ):
#             marker = "<<< COLOR" if j == bpos else ""
#             print(
#                 j,
#                 repr(tokenizer.decode([base_ids[j]])),
#                 marker,
#             )

#         print(
#             "\nSOURCE:",
#             "color =", pair["source_color"],
#             "| pos =", spos,
#             "| token =", repr(
#                 tokenizer.decode([source_ids[spos]])
#             ),
#         )

#         print("SOURCE context:")
#         for j in range(
#             max(0, spos - 3),
#             min(len(source_ids), spos + 4),
#         ):
#             marker = "<<< COLOR" if j == spos else ""
#             print(
#                 j,
#                 repr(tokenizer.decode([source_ids[j]])),
#                 marker,
#             )


