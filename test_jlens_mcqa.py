# import torch
# import transformers
# import jlens
# from mcqa_neural_net import load_gemma_model


# LENS_REPO = "neuronpedia/jacobian-lens"
# LENS_FILE = (
#     "gemma-2-2b/jlens/Salesforce-wikitext/"
#     "gemma-2-2b_jacobian_lens.pt"
# )

# TOP_K = 5

# PROMPT = """A mushroom is brown. What color is a mushroom?
# T. burgundy
# N. black
# P. purple
# W. brown
# Answer in one word:
# """


# def main():
#     # 1. Load Gemma-2-2B
#     hf_model, tokenizer = load_gemma_model()

#     model = jlens.from_hf(hf_model, tokenizer)

#     # 2. Load the published J-lens for Gemma-2-2B
#     lens = jlens.JacobianLens.from_pretrained(
#         LENS_REPO,
#         filename=LENS_FILE,
#     )

#     # 3. Apply J-lens at the last prompt token ("Answer:")
#     lens_logits, model_logits, input_ids = lens.apply(
#         model,
#         PROMPT,
#         positions=[-1],
#     )

#     print("\nPROMPT")
#     print("=" * 60)
#     print(PROMPT)

#     print("\nLAST INPUT TOKEN")
#     print("=" * 60)
#     print(repr(tokenizer.decode([input_ids[0, -1].item()])))

#     # 4. Print top-k J-lens tokens at every fitted layer
#     print("\nJ-LENS TOP TOKENS BY LAYER")
#     print("=" * 60)

#     for layer, logits in sorted(lens_logits.items()):
#         top_ids = logits[0].topk(TOP_K).indices.tolist()
#         top_tokens = [repr(tokenizer.decode([token_id])) for token_id in top_ids]
#         print(f"L{layer:02d}: {', '.join(top_tokens)}")

#     # Actual model next-token prediction for comparison
#     print("\nACTUAL MODEL NEXT-TOKEN TOP TOKENS")
#     print("=" * 60)

#     top_ids = model_logits[0].topk(TOP_K).indices.tolist()
#     top_tokens = [repr(tokenizer.decode([token_id])) for token_id in top_ids]
#     print(", ".join(top_tokens))


# if __name__ == "__main__":
#     main()



import torch
import jlens
from mcqa_neural_net import load_gemma_model

# ------------------------------------------------------------
# J-lens MCQA experiment: sweep the position of the correct
# answer W while keeping its label and answer content fixed.
# ------------------------------------------------------------

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = (
    "gemma-2-2b/jlens/Salesforce-wikitext/"
    "gemma-2-2b_jacobian_lens.pt"
)

TOP_K = 5
CHOICES = ["T", "N", "P", "W"]
CORRECT_CHOICE = "W"

# The order of T, N, P remains fixed.
DISTRACTORS = [
    ("T", "burgundy"),
    ("N", "black"),
    ("P", "purple"),
]
CORRECT_OPTION = ("W", "brown")

AP_POSITIONS = range(4)


def build_prompt(w_position):
    """Insert W at zero-based row position w_position."""
    options = DISTRACTORS.copy()
    options.insert(w_position, CORRECT_OPTION)

    option_text = "\n".join(
        f"{label}. {answer}"
        for label, answer in options
    )

    prompt = (
        "A mushroom is brown. What color is a mushroom?\n"
        f"{option_text}\n"
        "Answer:"
    )

    return prompt


def get_top_tokens(logits, tokenizer, k=5):
    token_ids = logits[0].topk(k).indices.tolist()
    return [
        repr(tokenizer.decode([token_id]))
        for token_id in token_ids
    ]


def get_choice_scores(logits, choice_ids):
    """Extract scores for T, N, P, W from one vocabulary distribution."""
    return {
        choice: float(logits[0, choice_ids[choice]].item())
        for choice in CHOICES
    }


def get_prediction(scores):
    """Return all tied winners, if any."""
    max_score = max(scores.values())
    winners = [
        choice
        for choice in CHOICES
        if abs(scores[choice] - max_score) < 1e-6
    ]
    return "/".join(winners)


def get_correct_margin(scores):
    """
    Positive: W beats all distractors.
    Zero: W ties the strongest distractor.
    Negative: a distractor beats W.
    """
    best_distractor = max(
        scores[c]
        for c in CHOICES
        if c != CORRECT_CHOICE
    )
    return scores[CORRECT_CHOICE] - best_distractor


def format_scores(scores):
    return "  ".join(
        f"{c}={scores[c]:8.3f}"
        for c in CHOICES
    )


def main():
    # 1. Load model and J-lens once.
    hf_model, tokenizer = load_gemma_model()
    model = jlens.from_hf(hf_model, tokenizer)

    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO,
        filename=LENS_FILE,
    )

    # 2. Get token IDs for answer labels.
    choice_ids = {}

    print("\nCHOICE TOKENIZATION")
    print("=" * 80)

    for choice in CHOICES:
        ids = tokenizer.encode(
            choice,
            add_special_tokens=False,
        )

        print(
            f"{choice}: ids={ids}, "
            f"tokens={[repr(tokenizer.decode([i])) for i in ids]}"
        )

        if len(ids) != 1:
            raise ValueError(
                f"{choice!r} is not a single token: {ids}"
            )

        choice_ids[choice] = ids[0]

    # Store results for all four AP conditions.
    all_results = {}

    # --------------------------------------------------------
    # 3. Sweep W across all four positions.
    # --------------------------------------------------------
    for ap in AP_POSITIONS:
        prompt = build_prompt(ap)

        print("\n")
        print("#" * 90)
        print(f"EXPERIMENT AP={ap} | W IS OPTION ROW {ap + 1} OF 4")
        print("#" * 90)

        print("\nPROMPT")
        print("-" * 80)
        print(prompt)

        # Apply J-lens at the final prompt token.
        lens_logits, model_logits, input_ids = lens.apply(
            model,
            prompt,
            positions=[-1],
        )

        print("LAST INPUT TOKEN:")
        print(repr(tokenizer.decode([input_ids[0, -1].item()])))

        # ----------------------------------------------------
        # 3a. Full-vocabulary top-k from J-lens.
        # ----------------------------------------------------
        print("\nJ-LENS TOP TOKENS BY LAYER")
        print("-" * 80)

        for layer, logits in sorted(lens_logits.items()):
            top_tokens = get_top_tokens(
                logits,
                tokenizer,
                TOP_K,
            )
            print(f"L{layer:02d}: {', '.join(top_tokens)}")

        # ----------------------------------------------------
        # 3b. J-lens restricted-choice scores.
        # ----------------------------------------------------
        print("\nJ-LENS MCQA SCORES BY LAYER")
        print("-" * 100)
        print("Scores are raw logits; margin = W - best distractor.")

        layer_results = {}

        for layer, logits in sorted(lens_logits.items()):
            scores = get_choice_scores(logits, choice_ids)
            pred = get_prediction(scores)
            margin = get_correct_margin(scores)

            layer_results[layer] = {
                "scores": scores,
                "pred": pred,
                "margin": margin,
            }

            print(
                f"L{layer:02d}: "
                f"{format_scores(scores)}"
                f"  | pred={pred:>3}"
                f"  | W_margin={margin:+8.3f}"
            )

        # ----------------------------------------------------
        # 3c. Actual model full-vocabulary prediction.
        # ----------------------------------------------------
        model_top_tokens = get_top_tokens(
            model_logits,
            tokenizer,
            TOP_K,
        )

        print("\nACTUAL MODEL NEXT-TOKEN TOP TOKENS")
        print("-" * 80)
        print(", ".join(model_top_tokens))

        # ----------------------------------------------------
        # 3d. Actual model restricted-choice scores.
        # ----------------------------------------------------
        model_scores = get_choice_scores(
            model_logits,
            choice_ids,
        )
        model_pred = get_prediction(model_scores)
        model_margin = get_correct_margin(model_scores)

        print("\nACTUAL MODEL MCQA SCORES")
        print("-" * 80)
        print(format_scores(model_scores))
        print(f"pred={model_pred} | W_margin={model_margin:+.3f}")

        all_results[ap] = {
            "prompt": prompt,
            "layers": layer_results,
            "model": {
                "scores": model_scores,
                "pred": model_pred,
                "margin": model_margin,
                "top_tokens": model_top_tokens,
            },
        }

    # --------------------------------------------------------
    # 4. Cross-AP summary: actual model.
    # --------------------------------------------------------
    print("\n\n")
    print("=" * 100)
    print("SUMMARY 1: ACTUAL MODEL ACROSS ALL AP POSITIONS")
    print("=" * 100)
    print(
        f"{'AP':<6} {'W row':<8} {'Prediction':<12} "
        f"{'W margin':>12} {'Full-vocab top tokens'}"
    )

    for ap in AP_POSITIONS:
        result = all_results[ap]["model"]

        print(
            f"{ap:<6} {ap + 1:<8} {result['pred']:<12} "
            f"{result['margin']:>+12.3f} "
            f"{', '.join(result['top_tokens'])}"
        )

    # --------------------------------------------------------
    # 5. Cross-AP summary: J-lens W margin at every layer.
    # --------------------------------------------------------
    print("\n\n")
    print("=" * 110)
    print("SUMMARY 2: J-LENS W MARGIN BY LAYER AND AP")
    print("=" * 110)
    print("Each cell = score(W) - max(score(T), score(N), score(P)).")
    print("Positive means W wins; negative means a distractor wins.")
    print()

    header = (
        f"{'Layer':<8}"
        f"{'AP=0 (row 1)':>20}"
        f"{'AP=1 (row 2)':>20}"
        f"{'AP=2 (row 3)':>20}"
        f"{'AP=3 (row 4)':>20}"
    )
    print(header)
    print("-" * len(header))

    layers = sorted(all_results[0]["layers"])

    for layer in layers:
        cells = []

        for ap in AP_POSITIONS:
            result = all_results[ap]["layers"][layer]
            cells.append(
                f"{result['margin']:+.3f} ({result['pred']})"
            )

        print(
            f"L{layer:02d}    "
            + "".join(f"{cell:>20}" for cell in cells)
        )

    # --------------------------------------------------------
    # 6. Cross-AP summary: J-lens predictions by layer.
    # --------------------------------------------------------
    print("\n\n")
    print("=" * 90)
    print("SUMMARY 3: J-LENS RESTRICTED-CHOICE PREDICTIONS")
    print("=" * 90)
    print(
        f"{'Layer':<8}"
        f"{'AP=0':>12}"
        f"{'AP=1':>12}"
        f"{'AP=2':>12}"
        f"{'AP=3':>12}"
    )
    print("-" * 56)

    for layer in layers:
        predictions = [
            all_results[ap]["layers"][layer]["pred"]
            for ap in AP_POSITIONS
        ]

        print(
            f"L{layer:02d}    "
            + "".join(f"{pred:>12}" for pred in predictions)
        )


if __name__ == "__main__":
    main()
