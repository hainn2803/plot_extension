"""Exact task renderers: formatting is varied, logical answers are not."""
from functools import lru_cache
from itertools import permutations
import json

from experiments.mcqa_c_at_format_search.runner import core, groups, key
from experiments.mcqa_c_at_localization.data import Prompt
from experiments.mcqa_c_at_localization.protocol import COLORS, LABELS

FORMS = ("equals_colon", "query_line", "arrow_line", "json_quote", "suffix_colon",
         "original_line", "label_question", "code_assert", "compact",
         "one_shot_original", "balanced8_original", "balanced8_compact", "balanced16_compact")
BARE = {"query_line", "arrow_line", "json_quote", "original_line", "code_assert"}


def base_render(p, name):
    if name in ("equals_colon", "suffix_colon"):
        return core(p, "equals_key" if name == "equals_colon" else "label_suffix")+":"
    if name in ("query_line", "arrow_line", "original_line"):
        return core(p, {"query_line": "query_last", "arrow_line": "arrow_key", "original_line": "original"}[name])+"\n"
    if name == "json_quote":
        return core(p, "json_key")+' "'
    if name == "label_question":
        old = f"What color is the {p.objects[p.query]}?"
        new = f"Which option label matches the color of the {p.objects[p.query]}?"
        return p.text.replace(old, new)
    if name == "code_assert":
        facts = json.dumps({p.objects[i]: p.facts[i] for i in p.fact_order})
        mapping = json.dumps(dict(zip(p.contents, p.labels)))
        obj = json.dumps(p.objects[p.query])
        return f'colors = {facts}\nlabels = {mapping}\nassert labels[colors[{obj}]] == "'
    if name == "compact":
        facts = "; ".join(f"{p.objects[i]} is {p.facts[i]}" for i in p.fact_order)
        options = "; ".join(f"{label}={color}" for label, color in zip(p.labels, p.contents))
        return f"Facts: {facts}.\nOptions: {options}.\nWhich option matches the color of {p.objects[p.query]}?\nAnswer:"
    if name == "original":
        return p.text
    raise ValueError(name)


@lru_cache(maxsize=None)
def demonstrations(n):
    # Each additional block changes the C-to-AT and C-to-position relationships.
    # Sixteen examples cover all 16 (requested color, answer letter) states.
    result = []
    for i in range(n):
        ci, block = i % 4, i // 4
        query = (ci+block) % 2
        letter_index = block if n == 16 else (ci+block) % 4
        ap = ci ^ letter_index if n == 16 else (ci+2*block+1) % 4
        target = LABELS[letter_index]
        contents = min((p for p in permutations(COLORS) if p[ap] == COLORS[ci]), key=lambda p: key([i, p]))
        labels = min((p for p in permutations(LABELS) if p[ap] == target), key=lambda p: key([i, p]))
        other = "here" if (ci+block) % 2 == 0 else COLORS[(ci+1+block % 3) % 4]
        facts = [other, other]
        facts[query] = COLORS[ci]
        result.append(Prompt(tuple(groups()[-2+i % 2]), tuple(facts), query,
                             (0, 1) if i % 2 == 0 else (1, 0), contents, labels))
    return tuple(sorted(result, key=lambda p: key(p.id)))


def render(p, form):
    if form in ("one_shot_original", "balanced8_original", "balanced8_compact", "balanced16_compact"):
        count = 1 if form.startswith("one_shot") else 16 if form.startswith("balanced16") else 8
        style = "compact" if form.endswith("compact") else "original"
        prefix = "\n\n".join(base_render(demo, style)+" "+demo.AT for demo in demonstrations(count))
        return prefix+"\n\n"+base_render(p, style)
    return base_render(p, form)


def continuation(form, label):
    return ("" if form in BARE else " ")+label
