"""Validate the exact exported pairs without generating or selecting examples."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHOICE = re.compile(r"(?m)^[ \t]*([A-Z])[.:][ \t]+([^\r\n]+?)[ \t]*$")
FACT = re.compile(r"\b(?:is|are)\s+(.+?)\.\s*What color\b", re.IGNORECASE)


def normalized(value):
    return str(value).strip().casefold()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Context:
    id: str
    prompt: str
    color: str
    labels: tuple
    contents: tuple
    question: str
    color_span: tuple

    @property
    def answer(self):
        return self.label_for(self.color)

    def label_for(self, color):
        matches = [i for i, value in enumerate(self.contents) if value == normalized(color)]
        if len(matches) != 1:
            raise ValueError(f"Color {color!r} must occur once in {self.id}'s options")
        return self.labels[matches[0]]


@dataclass(frozen=True)
class Pair:
    id: str
    base: int
    source: int
    family: str
    cluster: str
    original_question: str = ""


def parse_context(row, side):
    prompt = row[f"{side}_prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("An exact nonempty prompt string is required")
    choices = list(CHOICE.finditer(prompt))
    if len(choices) != 4:
        raise ValueError("Expected exactly four explicit A-Z option rows")
    labels = tuple(match.group(1) for match in choices)
    contents = tuple(normalized(match.group(2)) for match in choices)
    if len(set(labels)) != 4 or len(set(contents)) != 4:
        raise ValueError("Repeated option labels or contents")
    first_line = prompt.splitlines()[0]
    facts = list(FACT.finditer(first_line))
    if len(facts) != 1:
        raise ValueError("This importer requires Hai's single-fact prompt; no template substitution")
    fact = facts[0]
    color = normalized(fact.group(1))
    if f"{side}_color" in row and normalized(row[f"{side}_color"]) != color:
        raise ValueError("Exported color disagrees with the prompt")
    supplied = row.get(f"{side}_choices")
    if supplied is not None and (tuple(supplied["label"]) != labels or
            tuple(map(normalized, supplied["text"])) != contents):
        raise ValueError("Exported option mapping disagrees with the prompt")
    question = row.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("An explicit original question/group key is required")
    start, end = fact.span(1)
    template = " ".join((first_line[:start] + "<COLOR>" + first_line[end:]).split()).casefold()
    context = Context(digest(prompt), prompt, color, labels, contents, template, fact.span(1))
    answer = context.answer
    for key in (f"{side}_AT", f"{side}_Y"):
        if key in row and row[key] != answer:
            raise ValueError(f"{key} disagrees with the prompt")
    if f"{side}_AP" in row and int(row[f"{side}_AP"]) != contents.index(color):
        raise ValueError("Exported answer position disagrees with the prompt")
    return context


def import_pairs(rows):
    contexts, pairs, index, seen = [], [], {}, set()
    for row in rows:
        base, source = (parse_context(row, side) for side in ("base", "source"))
        key = (base.id, source.id)
        if base.id == source.id or key in seen:
            raise ValueError("Identical prompts or repeated directed pair; preserve and inspect the export")
        seen.add(key)
        target = base.label_for(source.color)
        for field in ("AT_after_C_intervene", "Y_after_C_intervene"):
            if field in row and row[field] != target:
                raise ValueError(f"{field} must use the source color in the BASE options")
        family = row.get("pair_type", "unspecified")
        c_change, at_change = base.color != source.color, base.answer != source.answer
        ap_change = base.contents.index(base.color) != source.contents.index(source.color)
        required = {"C_only": (True, False, False), "C_AP_AT": (True, True, True),
                    "AT_only": (False, True, False), "AP_only": (False, False, True)}
        if family in required and (c_change, at_change, ap_change) != required[family]:
            raise ValueError(f"Pair does not satisfy its declared {family} natural-state changes")
        for context in (base, source):
            if context.id in index:
                if contexts[index[context.id]] != context:
                    raise ValueError("A repeated prompt has conflicting natural metadata")
            else:
                index[context.id] = len(contexts)
                contexts.append(context)
        pair_id = str(row.get("id", digest(key)))
        if any(pair.id == pair_id for pair in pairs):
            raise ValueError("Repeated pair ID")
        pairs.append(Pair(pair_id, index[base.id], index[source.id], family, base.question, row["question"]))
    if not pairs:
        raise ValueError("Empty exact pair export")
    return contexts, pairs


def read_pairs(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value["pairs"] if isinstance(value, dict) else value
    return import_pairs(rows)


def coverage(contexts, pairs):
    families, directions, labels = {}, {}, {}
    for pair in pairs:
        base, source = contexts[pair.base], contexts[pair.source]
        families[pair.family] = families.get(pair.family, 0) + 1
        key = f"{base.color}->{source.color}"
        directions[key] = directions.get(key, 0) + 1
        key = "".join(base.labels)
        labels[key] = labels.get(key, 0) + 1
    return dict(contexts=len(contexts), pairs=len(pairs), families=families,
                c_transitions=directions, base_label_orders=labels,
                natural_letters=sorted({c.answer for c in contexts}),
                clusters=len({p.cluster for p in pairs}),
                provided_question_keys=len({p.original_question or p.cluster for p in pairs}),
                cluster_definition="Exact first-line template with stated color masked; original keys retained",
                preservation_pairs=sum(contexts[p.base].color == contexts[p.source].color for p in pairs))
