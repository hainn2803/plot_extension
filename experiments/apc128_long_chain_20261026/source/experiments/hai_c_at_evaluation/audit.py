"""Run selected pure functions from the pinned source, without importing its GPU loader."""
import ast
import itertools
from pathlib import Path
import random
import re
import string
import unicodedata

import torch

from .schema import file_hash

COMMIT = "a2bec277f2bab4c57585446c46ef7cd030ac09c1"
AUDIT_FILES = ("mcqa_generation.py", "mcqa_load_randomized_ap_at_c.py", "mcqa_gradual_das.py",
    "mcqa_gradual_das_c.py", "mcqa_gradual_das_at.py", "mcqa_joint_das_ap_at_c_balanced.py",
    "mcqa_eval_readout_mediator.py", "mcqa_neural_net.py", "mcqa_data_load_jchang_disjoint.py",
    "mcqa_data_load_all.py", "mcqa_gradual_das_ap.py", ".gitignore")


def selected_functions(path, names, namespace=None):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name in names]
    if {node.name for node in definitions} != set(names):
        raise ValueError("Pinned source lacks an audited function")
    scope = dict(torch=torch, re=re, string=string, unicodedata=unicodedata)
    if namespace:
        scope.update(namespace)
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


def inspect(snapshot):
    snapshot = Path(snapshot)
    loader = selected_functions(snapshot / "mcqa_load_randomized_ap_at_c.py",
        ["label_at_position", "position_of_color_in_base", "intervention_targets"])
    target_checks = 0
    colors = ("red", "blue", "green", "yellow")
    for labels in itertools.permutations("ABCD"):
        for base_color in colors:
            for source_labels in itertools.permutations("ABCD"):
                for source_color in colors:
                    pair = dict(base_choices=dict(label=labels, text=colors),
                        base_color=base_color, source_color=source_color,
                        source_AP=colors.index(source_color), source_AT=source_labels[colors.index(source_color)])
                    result = loader["intervention_targets"](pair)
                    expected = labels[colors.index(source_color)]
                    assert result["Y_after_C_intervene"] == result["AT_after_C_intervene"] == expected
                    assert result["Y_after_AT_intervene"] == pair["source_AT"]
                    target_checks += 1
    namespace = dict(LETTERS=string.ascii_uppercase,
        COLOR_PATTERN=re.compile(r"\b(?:is|are)\s+(.+?)\.\s*What color\b", re.IGNORECASE),
        CHOICE_LINE=re.compile(r"(?m)^([ \t]*)([A-Z])\.[ \t]+([^\r\n]+)$"))
    functions = ["question_color", "replace_question_color", "rewrite_choices", "find_answer", "make_state",
                 "make_randomized_base", "make_ap_only", "make_at_only", "choose_other_color", "make_c_only", "make_c_all_change"]
    generator = selected_functions(snapshot / "mcqa_generation.py", functions, namespace)
    row = dict(prompt="The sky is blue. What color is the sky?\nA. red\nB. blue\nC. green\nD. yellow\nAnswer:",
               choices=dict(label=list("ABCD"), text=list(colors)))
    observed = {name: set() for name in ("base", "C_only", "C_AP_AT", "AT_only")}
    for seed in range(128):
        rng = random.Random(seed)
        base = generator["make_randomized_base"](row, rng)
        observed["base"].update(base["choices"]["label"])
        for name, function in (("C_only", "make_c_only"), ("C_AP_AT", "make_c_all_change"), ("AT_only", "make_at_only")):
            source = generator[function](base, rng)
            observed[name].update(source["choices"]["label"])
    scoring = selected_functions(snapshot / "mcqa_gradual_das.py", ["full_vocab_iia"],
                                 dict(ANSWER_LETTERS=tuple(string.ascii_uppercase)))
    class Decoder:
        def decode(self, ids):
            return ("a", "A")[ids[0]]
    lowercase_score = scoring["full_vocab_iia"](torch.tensor([[2., 0.]]), torch.tensor([0]), Decoder())
    keys = selected_functions(snapshot / "mcqa_data_load_jchang_disjoint.py", ["question_key"])
    blue = keys["question_key"](dict(prompt=row["prompt"]))
    green = keys["question_key"](dict(prompt=row["prompt"].replace("is blue.", "is green.")))
    return dict(commit=COMMIT, files={name: file_hash(snapshot / name) for name in AUDIT_FILES},
        targets=dict(cases=target_checks, failures=0, target="base mapping[source color]"),
        generator=dict(seeds=128, observed_labels={key: sorted(value) for key, value in observed.items()},
                       base_is_restricted_to_ABCD=observed["base"] == set("ABCD")),
        scoring=dict(lowercase_a_counts_as_A=lowercase_score == 1.0),
        grouping=dict(question_keys_include_stated_color=blue != green, blue=blue, green=green),
        interpretation="These checks describe this commit, not Hai's unshared datasets or saved handles")
