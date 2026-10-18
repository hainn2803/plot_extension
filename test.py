# from mcqa_load_randomized_ap_at_not_c import load_generated_dataset, row_to_pairs

# ds = load_generated_dataset("randomized_copycolors")

# count = 0
# for row in ds["train"]:
#     pairs = row_to_pairs(row, "train")

#     for p in pairs:
#         if p["pair_type"] != "AP_only":
#             continue

#         print("=" * 80)
#         print("BASE")
#         print(p["base_prompt"])
#         print("C :", p["base_color"])
#         print("AP:", p["base_AP"])
#         print("AT:", p["base_AT"])

#         print("\nSOURCE AP_only")
#         print(p["source_prompt"])
#         print("C :", p["source_color"])
#         print("AP:", p["source_AP"])
#         print("AT:", p["source_AT"])

#         print("\nAP intervention target:",
#               p["base_choices"]["label"][p["source_AP"]])

#         count += 1
#         if count == 10:
#             raise SystemExit


# import string
# import unicodedata
# from collections import Counter, defaultdict

# import torch

# from mcqa_neural_net import load_gemma_model
# from mcqa_load_randomized_ap_at_not_c import build_all_banks
# from mcqa_gradual_das import (
#     answer_label_ids,
#     collect_layer_states,
#     output_label_ids,
#     run_das_intervention,
# )


# CKPT = "results_joint_standard_L17/das_ap.pt"
# DATA_PATH = "randomized_copycolors"


# def decode_predictions(logits, tokenizer, output_space):
#     pred = logits.argmax(dim=-1).detach().cpu().tolist()

#     if output_space == "az":
#         return [string.ascii_uppercase[i] for i in pred]

#     return [
#         unicodedata.normalize("NFKC", tokenizer.decode([int(i)]))
#         .strip()
#         .upper()
#         for i in pred
#     ]


# def main():
#     model, tokenizer = load_gemma_model()

#     ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
#     layer = int(ckpt["layer"])
#     Q = ckpt["basis"].to(next(model.parameters()).device).float()
#     token_position = ckpt["token_position"]
#     output_space = ckpt.get("output_space", "full")
#     config = ckpt["config"]

#     print("AP layer:", layer)
#     print("output_space:", output_space)

#     all_banks = build_all_banks(
#         model=model,
#         tokenizer=tokenizer,
#         data_path=DATA_PATH,
#         ft_size=config["ft_size"],
#         cal_size=config["cal_size"],
#         te_size=config["te_size"],
#         batch_size=config["eval_batch_size"],
#         seed=config["seed"],
#     )

#     # test_bank = all_banks["AP"][2]["answer_pointer"]

#     # assert all(p["pair_type"] == "AP_only" for p in test_bank["pairs"])
#     test_bank = all_banks["AT"][2]["answer_pointer"]

#     assert all(p["pair_type"] == "AT_only" for p in test_bank["pairs"])

#     source_states = collect_layer_states(
#         model,
#         test_bank["source_input_ids"],
#         test_bank["source_attention_mask"],
#         test_bank["source_position_by_id"],
#         layer,
#         token_position,
#         config["eval_batch_size"],
#     )

#     labels = answer_label_ids(tokenizer)
#     output_ids = output_label_ids(labels, output_space)

#     with torch.no_grad():
#         logits = run_das_intervention(
#             model,
#             test_bank,
#             source_states,
#             Q,
#             layer,
#             output_ids,
#             token_position,
#             config["eval_batch_size"],
#             require_grad=False,
#         )

#     predictions = decode_predictions(logits, tokenizer, output_space)

#     # counts = Counter()
#     # transitions = defaultdict(Counter)
#     # failures = []

#     # for i, (pair, pred) in enumerate(zip(test_bank["pairs"], predictions)):
#     #     base_labels = [str(x) for x in pair["base_choices"]["label"]]

#     #     base_ap = pair["base_AP"]
#     #     source_ap = pair["source_AP"]

#     #     old_at = pair["base_AT"]
#     #     source_at = pair["source_AT"]
#     #     base_ap_label = base_labels[base_ap]
#     #     target = base_labels[source_ap]

#     #     assert old_at == source_at
#     #     assert old_at == base_ap_label
#     #     assert target != old_at

#     #     if pred == target:
#     #         category = "TARGET"
#     #     elif pred == old_at:
#     #         category = "OLD_AT"
#     #     elif pred in base_labels:
#     #         category = "OTHER_BASE_CHOICE"
#     #     else:
#     #         category = "OUTSIDE_BASE_CHOICES"

#     #     counts[category] += 1

#     #     key = (base_ap, source_ap)
#     #     transitions[key]["n"] += 1
#     #     transitions[key][category] += 1

#     #     if category != "TARGET" and len(failures) < 20:
#     #         failures.append({
#     #             "i": i,
#     #             "AP": f"{base_ap}->{source_ap}",
#     #             "labels": base_labels,
#     #             "old_AT": old_at,
#     #             "target": target,
#     #             "pred": pred,
#     #             "category": category,
#     #         })

#     counts = Counter()
#     transitions = defaultdict(Counter)
#     failures = []

#     for i, (pair, pred) in enumerate(zip(test_bank["pairs"], predictions)):
#         base_labels = [str(x) for x in pair["base_choices"]["label"]]

#         base_ap = pair["base_AP"]
#         source_ap = pair["source_AP"]

#         base_at = pair["base_AT"]
#         source_at = pair["source_AT"]

#         # Case 2:
#         # AP fixed, AT changes
#         assert base_ap == source_ap
#         assert base_at != source_at

#         # Sanity check: base answer letter should be the label
#         # at the correct base position
#         assert base_labels[base_ap] == base_at

#         if pred == base_at:
#             category = "BASE_AT"
#         elif pred == source_at:
#             category = "SOURCE_AT"
#         elif pred in base_labels:
#             category = "OTHER_BASE_CHOICE"
#         else:
#             category = "OUTSIDE_BASE_CHOICES"

#         counts[category] += 1

#         key = (base_at, source_at)
#         transitions[key]["n"] += 1
#         transitions[key][category] += 1

#         if category != "BASE_AT" and len(failures) < 20:
#             failures.append({
#                 "i": i,
#                 "AP": base_ap,
#                 "labels": base_labels,
#                 "base_AT": base_at,
#                 "source_AT": source_at,
#                 "pred": pred,
#                 "category": category,
#             })

#     n = len(predictions)

#     print("\n" + "=" * 70)
#     print("OVERALL")
#     print("=" * 70)

#     for category in (
#         "TARGET",
#         "OLD_AT",
#         "OTHER_BASE_CHOICE",
#         "OUTSIDE_BASE_CHOICES",
#     ):
#         c = counts[category]
#         print(f"{category:22s}: {c:4d}/{n} = {c / n:.4f}")

#     print("\n" + "=" * 70)
#     print("BY AP TRANSITION")
#     print("=" * 70)

#     for key in sorted(transitions):
#         c = transitions[key]
#         n_k = c["n"]

#         print(
#             f"{key[0]}->{key[1]} | n={n_k:3d} | "
#             f"target={c['TARGET'] / n_k:.3f} | "
#             f"old_AT={c['OLD_AT'] / n_k:.3f} | "
#             f"other={c['OTHER_BASE_CHOICE'] / n_k:.3f} | "
#             f"outside={c['OUTSIDE_BASE_CHOICES'] / n_k:.3f}"
#         )

#     print("\n" + "=" * 70)
#     print("FIRST 20 FAILURES")
#     print("=" * 70)

#     for x in failures:
#         print(
#             f"#{x['i']:03d} AP={x['AP']} "
#             f"labels={x['labels']} "
#             f"old_AT={x['old_AT']} "
#             f"target={x['target']} "
#             f"pred={x['pred']} "
#             f"[{x['category']}]"
#         )


# if __name__ == "__main__":
#     main()




import string
import unicodedata
from collections import Counter, defaultdict

import torch

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_not_c import build_all_banks
from mcqa_gradual_das import (
    answer_label_ids,
    collect_layer_states,
    output_label_ids,
    run_das_intervention,
)


CKPT = "results_joint_standard_L17/das_ap.pt"
DATA_PATH = "randomized_copycolors"


def decode_predictions(logits, tokenizer, output_space):
    pred = logits.argmax(dim=-1).detach().cpu().tolist()

    if output_space == "az":
        return [string.ascii_uppercase[i] for i in pred]

    return [
        unicodedata.normalize("NFKC", tokenizer.decode([int(i)]))
        .strip()
        .upper()
        for i in pred
    ]


def main():
    model, tokenizer = load_gemma_model()

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    layer = int(ckpt["layer"])
    Q = ckpt["basis"].to(next(model.parameters()).device).float()
    token_position = ckpt["token_position"]
    output_space = ckpt.get("output_space", "full")
    config = ckpt["config"]

    print("AP layer:", layer)
    print("output_space:", output_space)

    all_banks = build_all_banks(
        model=model,
        tokenizer=tokenizer,
        data_path=DATA_PATH,
        ft_size=config["ft_size"],
        cal_size=config["cal_size"],
        te_size=config["te_size"],
        batch_size=config["eval_batch_size"],
        seed=config["seed"],
    )

    test_bank = all_banks["AT"][2]["answer_token"]

    assert all(p["pair_type"] == "AT_only" for p in test_bank["pairs"])

    source_states = collect_layer_states(
        model,
        test_bank["source_input_ids"],
        test_bank["source_attention_mask"],
        test_bank["source_position_by_id"],
        layer,
        token_position,
        config["eval_batch_size"],
    )

    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    with torch.no_grad():
        logits = run_das_intervention(
            model,
            test_bank,
            source_states,
            Q,
            layer,
            output_ids,
            token_position,
            config["eval_batch_size"],
            require_grad=False,
        )

    predictions = decode_predictions(logits, tokenizer, output_space)

    counts = Counter()
    transitions = defaultdict(Counter)
    failures = []

    for i, (pair, pred) in enumerate(zip(test_bank["pairs"], predictions)):
        base_labels = [str(x) for x in pair["base_choices"]["label"]]

        base_ap = pair["base_AP"]
        source_ap = pair["source_AP"]

        base_at = pair["base_AT"]
        source_at = pair["source_AT"]

        # Case 2:
        # AP fixed, AT changes
        assert base_ap == source_ap
        assert base_at != source_at

        # Sanity check: base answer letter should be the label
        # at the correct base position
        assert base_labels[base_ap] == base_at

        if pred == base_at:
            category = "BASE_AT"
        elif pred == source_at:
            category = "SOURCE_AT"
        elif pred in base_labels:
            category = "OTHER_BASE_CHOICE"
        else:
            category = "OUTSIDE_BASE_CHOICES"

        counts[category] += 1

        key = (base_at, source_at)
        transitions[key]["n"] += 1
        transitions[key][category] += 1

        if category != "BASE_AT" and len(failures) < 20:
            failures.append({
                "i": i,
                "AP": base_ap,
                "labels": base_labels,
                "base_AT": base_at,
                "source_AT": source_at,
                "pred": pred,
                "category": category,
            })

    n = len(predictions)

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)

    for category in (
        "BASE_AT",
        "SOURCE_AT",
        "OTHER_BASE_CHOICE",
        "OUTSIDE_BASE_CHOICES",
    ):
        c = counts[category]
        print(f"{category:22s}: {c:4d}/{n} = {c / n:.4f}")

    print("\n" + "=" * 70)
    print("BY AT TRANSITION")
    print("=" * 70)

    for key in sorted(transitions):
        c = transitions[key]
        n_k = c["n"]

        print(
            f"{key[0]}->{key[1]} | n={n_k:3d} | "
            f"base_AT={c['BASE_AT'] / n_k:.3f} | "
            f"source_AT={c['SOURCE_AT'] / n_k:.3f} | "
            f"other={c['OTHER_BASE_CHOICE'] / n_k:.3f} | "
            f"outside={c['OUTSIDE_BASE_CHOICES'] / n_k:.3f}"
        )

    print("\n" + "=" * 70)
    print("FIRST 20 NON-BASE PREDICTIONS")
    print("=" * 70)

    for x in failures:
        print(
            f"#{x['i']:03d} "
            f"AP={x['AP']} "
            f"labels={x['labels']} "
            f"base_AT={x['base_AT']} "
            f"source_AT={x['source_AT']} "
            f"pred={x['pred']} "
            f"[{x['category']}]"
        )


if __name__ == "__main__":
    main()