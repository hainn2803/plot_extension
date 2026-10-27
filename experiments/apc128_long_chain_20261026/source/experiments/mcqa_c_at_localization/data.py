"""Explicit facts, query, display order, labels, and counterfactual targets."""
from dataclasses import asdict, dataclass, replace

from .protocol import COLORS, LABELS
from .storage import digest


@dataclass(frozen=True)
class Prompt:
    objects: tuple[str, str]
    facts: tuple[str, str]
    query: int
    fact_order: tuple[int, int]
    contents: tuple[str, str, str, str]
    labels: tuple[str, str, str, str]
    wording: str = "What"

    def __post_init__(self):
        if len(set(self.objects)) != 2 or self.query not in (0, 1):
            raise ValueError("Invalid object group or query")
        if sorted(self.fact_order) != [0, 1]:
            raise ValueError("Invalid fact order")
        if set(self.contents) != set(COLORS) or set(self.labels) != set(LABELS):
            raise ValueError("Each color and label must appear exactly once")
        if self.facts[self.query] not in COLORS:
            raise ValueError("The queried object must have a stated color")
        if any(c not in (*COLORS, "here") for c in self.facts):
            raise ValueError("Unexpected fact value")
        if "here" not in self.facts and self.facts[0] == self.facts[1]:
            raise ValueError("Two-color prompts must distinguish the queried colors")
        if self.wording not in ("What", "Which"):
            raise ValueError("Unregistered wording")

    @property
    def C(self):
        return self.facts[self.query]

    @property
    def AP(self):
        return self.contents.index(self.C)

    @property
    def AT(self):
        return self.labels[self.AP]

    @property
    def condition(self):
        return "one_color" if "here" in self.facts else "two_colors"

    def answer_for_color(self, color):
        return self.labels[self.contents.index(color)]

    @property
    def text(self):
        facts = " ".join(f"The {self.objects[i]} is {self.facts[i]}." for i in self.fact_order)
        options = "\n".join(f"{label}. {color}" for label, color in zip(self.labels, self.contents))
        return f"Question: {facts} {self.wording} color is the {self.objects[self.query]}?\n{options}\nAnswer:"

    @property
    def id(self):
        return digest(self.text)

    def record(self):
        return {**asdict(self), "id": self.id, "text": self.text,
                "C": self.C, "AP": self.AP, "AT": self.AT, "Y": self.AT,
                "condition": self.condition}


def from_record(row):
    return Prompt(tuple(row["objects"]), tuple(row["facts"]), row["query"],
                  tuple(row["fact_order"]), tuple(row["contents"]), tuple(row["labels"]),
                  row.get("wording", "What"))


def targets(base, source):
    return {"C": base.answer_for_color(source.C), "AT": source.AT,
            "position": base.labels[source.AP], "base": base.AT}


def pair_record(base, source, family, slot=None):
    if base.id == source.id or base.condition != source.condition:
        raise ValueError("Pairs must be nonidentical and condition-matched")
    if base.objects != source.objects:
        raise ValueError("This experiment pairs within fixed two-object groups")
    return {"id": digest([base.id, source.id]), "base": base.record(), "source": source.record(),
            "family": family, "condition": base.condition, "group": list(base.objects),
            "slot": slot, "targets": targets(base, source),
            "C_changed": base.C != source.C, "AT_changed": base.AT != source.AT}


def effect(base, source, variable):
    target = targets(base, source)[variable]
    return [int(letter == target) - int(letter == base.AT) for letter in LABELS]


def examples():
    base = Prompt(("cup", "plate"), ("blue", "red"), 0, (0, 1),
                  ("red", "blue", "yellow", "green"), LABELS)
    sources = [
        replace(base, facts=("green", "red"), contents=("red", "green", "yellow", "blue")),
        replace(base, labels=("B", "A", "C", "D")),
        replace(base, contents=("blue", "red", "yellow", "green")),
        replace(base, facts=("green", "red")),
        replace(base, facts=("green", "red"), contents=("yellow", "red", "green", "blue"),
                labels=("B", "C", "A", "D")),
        replace(base, contents=("blue", "red", "yellow", "green"), labels=("B", "A", "C", "D")),
        replace(base, facts=("blue", "yellow")),
        replace(base, query=1),
    ]
    from .protocol import FAMILIES
    return [pair_record(base, source, family) for source, family in zip(sources, FAMILIES)]


def family_check(base, source, family, one_color_query_mode="preserve"):
    c, at = base.C != source.C, base.AT != source.AT
    if base.condition != source.condition or base.objects != source.objects or base.id == source.id:
        return False
    same_facts = base.facts == source.facts and base.query == source.query and base.fact_order == source.fact_order
    same_options = base.contents == source.contents and base.labels == source.labels
    if family == "C_change_AT_fixed":
        return c and not at and base.query == source.query and base.fact_order == source.fact_order
    if family == "labels_only":
        return not c and at and same_facts and base.contents == source.contents
    if family == "contents_only":
        return not c and at and same_facts and base.labels == source.labels
    if family == "both_options_fixed":
        return c and at and same_options and base.query == source.query and base.fact_order == source.fact_order
    if family == "four_way":
        return c and at and len(set(targets(base, source).values())) == 4 and len(set(targets(source, base).values())) == 4
    if family == "row_order_only":
        return not c and not at and same_facts and base.AP != source.AP and set(zip(base.labels, base.contents)) == set(zip(source.labels, source.contents))
    if family == "irrelevant_fact":
        if base.condition == "one_color":
            return not c and not at and same_options and base.facts == source.facts and base.query == source.query and base.fact_order != source.fact_order
        return not c and not at and same_options and base.query == source.query and base.fact_order == source.fact_order and base.facts[1-base.query] != source.facts[1-source.query]
    if family == "requested_object":
        if base.condition == "two_colors":
            return c and at and same_options and base.facts == source.facts and base.query != source.query and base.fact_order == source.fact_order
        desired_change = one_color_query_mode == "change"
        return c == desired_change and at == desired_change and same_options and base.query != source.query and base.fact_order == source.fact_order and source.facts[base.query] == "here"
    raise ValueError(f"Unknown family {family}")
