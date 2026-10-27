"""Outcome-blind finite donor schedules and immutable parent imports."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from diagnostics.hai_frozen_eval_20261020 import clean_only, descriptive as desc, run as old
from experiments.hai_c_at_evaluation import schema

ROOT = Path("eval/hai_c_at_full_chain_20261020_v1")
SALT = "hai_full_chain_20261020_v1"
METHODS = ("standard", "gradual")
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def ranked(contexts, indices, tag):
    return sorted(indices, key=lambda i: (schema.digest([SALT, tag, contexts[i].id]), i))


def donors(contexts, ordered, excluded, count=2, distinct_colors=True):
    chosen, templates, colors = [], set(excluded), set()
    for i in ordered:
        c = contexts[i]
        if c.question in templates or (distinct_colors and c.color in colors):
            continue
        chosen.append(i)
        templates.add(c.question)
        colors.add(c.color)
        if len(chosen) == count:
            return chosen
    raise ValueError("Finite donor/recipient pool cannot satisfy registered independence")


def schedule(contexts, pairs, retained):
    valid = sorted({i for p in pairs for i in (p.base, p.source)})
    by_letter = {x: ranked(contexts, [i for i in valid if contexts[i].answer == x], ["AT", x]) for x in LETTERS}
    by_color = {x: ranked(contexts, [i for i in valid if contexts[i].color == x], ["C", x])
                for x in sorted({contexts[i].color for i in valid})}
    first = {}
    for j, p in enumerate(pairs):
        first.setdefault(p.base, j)
    bases = sorted(first)
    ci, ai = [], []
    for b in bases:
        c = contexts[b]
        cd = []
        for color in c.contents:
            eligible = [i for i in by_color[color] if contexts[i].answer != c.label_for(color)]
            cd.append(donors(contexts, eligible, {c.question}, count=1)[0])
        ci.append(cd)
        excluded = {c.question, *(contexts[i].question for i in cd)}
        ai.append([donors(contexts, by_letter[x], excluded) for x in LETTERS])
    recipients, cross_donors, pair_donors = [], [], []
    for p in pairs:
        b, s = contexts[p.base], contexts[p.source]
        target = b.label_for(s.color)
        excluded = {b.question, s.question}
        pair_donors.append(donors(contexts, by_letter[target], excluded))
        same = donors(contexts, by_letter[target], excluded, distinct_colors=False)
        changing_order = ranked(contexts, [i for i in valid if target in contexts[i].labels
            and contexts[i].answer != target], ["recipient_change", target])
        other = donors(contexts, changing_order, excluded | {contexts[i].question for i in same},
                       distinct_colors=False)
        selected = same + other
        recipients.append(selected)
        cross_donors.append([donors(contexts, by_letter[target], excluded | {contexts[r].question}) for r in selected])
    directions = defaultdict(set)
    for b in bases:
        for color in contexts[b].contents:
            if color != contexts[b].color:
                directions[contexts[b].color + "->" + color].add(contexts[b].question)
    return dict(salt=SALT, pairs=[asdict(p) for p in pairs], original_pair_indices=retained,
        valid_contexts=valid, bases=bases, base_pair_occurrences=[first[b] for b in bases],
        c_donors=ci, at_donors=ai, recipients=recipients, cross_donors=cross_donors,
        pair_donors=pair_donors, letters=list(LETTERS),
        coverage=dict(pairs=len(pairs), contexts=len(valid), bases=len(bases),
            templates=len({p.cluster for p in pairs}), families=dict(Counter(p.family for p in pairs)),
            colors=sorted(by_color), letters=list(LETTERS),
            factorial_direction_template_counts={k: len(v) for k, v in sorted(directions.items())},
            factorial_cases_per_method=len(bases)*4*26*2, control_cases=len(bases)*26*2,
            cross_cases_per_method=len(pairs)*4))


def readout_pool(contexts, valid):
    excluded = {contexts[i].question for i in valid}
    rows, seen = [], set()
    path = old.DATA / "generated/train.jsonl"
    for line in path.read_text().splitlines():
        r = json.loads(line)
        c = schema.parse_context(dict(base_prompt=r["prompt"], base_color=r["color"],
            base_choices=r["choices"], question=r["question_key"]), "base")
        if c.question not in excluded and c.id not in seen:
            rows.append(c)
            seen.add(c.id)
    templates = sorted({c.question for c in rows}, key=lambda t: schema.digest([SALT, "readout", t]))
    if len(templates) != 23:
        raise ValueError("Independent readout template pool changed")
    fitting = set(templates[:16])
    return dict(contexts=[asdict(c) for c in rows], split=["fit" if c.question in fitting else "validation" for c in rows],
        fit_templates=templates[:16], validation_templates=templates[16:],
        colors=sorted({contexts[i].color for i in valid}), letters=list(LETTERS),
        penalties=[1e-4, 1e-3, 1e-2], lr=.01, epochs=200, seed=1,
        selection="Natural validation accuracy, then CE; no intervention outcomes or handle selection")


def verify_originals():
    hashes = {}
    for root in (desc.ROOT, clean_only.ROOT):
        receipt = old.read(root / "completion.json")
        for name, expected in receipt["artifacts"].items():
            p = root / name
            if schema.file_hash(p) != expected:
                raise ValueError("Historical result changed: " + str(p))
            hashes[str(p)] = expected
        hashes[str(root / "completion.json")] = schema.file_hash(root / "completion.json")
    return hashes


def prepare(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    originals = verify_originals()
    old.establish_identity(old.ROOT)
    contexts, all_pairs = schema.import_pairs(old.read(desc.ROOT / "selected_pairs.json"))
    retained = old.read(clean_only.ROOT / "selection.json")["selected_indices"]
    pairs = [all_pairs[i] for i in retained]
    if len(pairs) != 1476:
        raise ValueError("Approved common bank changed")
    plan = schedule(contexts, pairs, retained)
    ro = readout_pool(contexts, plan["valid_contexts"])
    old.write(root / "plan.json", plan)
    old.write(root / "readout_plan.json", ro)
    old.write(root / "contexts.json", [asdict(c) for c in contexts])
    sources = [*Path(__file__).parent.glob("*.py"), Path(desc.__file__), Path(old.__file__),
        Path(clean_only.__file__), *Path("experiments/hai_c_at_evaluation").glob("*.py"),
        Path("experiments/mcqa_c_at_localization/runtime.py"),
        Path("experiments/mcqa_c_at_localization/profiling.py"),
        Path("experiments/mcqa_c_at_localization/readouts.py")]
    imports = {}
    for folder in [old.ROOT / "operations/natural", *[desc.ROOT / "operations" / n for n in
        ("matched_base", "standard_direct", "gradual_direct")]]:
        for path in [folder / "identity.json", *sorted(folder.glob("chunk_*.json"))]:
            imports[str(path)] = schema.file_hash(path)
    identity = dict(sources={str(p): schema.file_hash(p) for p in sources}, originals=originals,
        imports=imports, parent_identity=schema.file_hash(old.ROOT / "identity.json"),
        plan_sha256=schema.file_hash(root / "plan.json"), readout_plan_sha256=schema.file_hash(root / "readout_plan.json"),
        contexts_sha256=schema.file_hash(root / "contexts.json"),
        settings=dict(model="gemma-2-2b", revision="c5ebcd40d208330abc697524c919956e692655cf",
            backbone="BF16 frozen", projections="FP32", physical_batch=32, attention="sdpa", tf32=False,
            atol=2e-4, rtol=2e-4, budget_seconds=5400, safety="unchanged runtime; no automatic reset/resume",
            changes_only="new diagnostic operations and independent linear measurements; no handle training"),
        selection="Shared baseline-only 1476-pair mask, then metadata/hash donors. No intervention-based selection.")
    old.write(root / "identity.json", identity)
    return contexts, pairs, plan, ro


def load(root=ROOT):
    root = Path(root)
    ident = old.read(root / "identity.json")
    for group in ("sources", "originals", "imports"):
        for name, expected in ident[group].items():
            if schema.file_hash(name) != expected:
                raise ValueError("Immutable input/source changed: " + name)
    for name, key in (("plan.json", "plan_sha256"), ("readout_plan.json", "readout_plan_sha256"),
                      ("contexts.json", "contexts_sha256")):
        if schema.file_hash(root / name) != ident[key]:
            raise ValueError("Immutable diagnostic plan changed: " + name)
    contexts = [schema.Context(**{**r, "labels": tuple(r["labels"]), "contents": tuple(r["contents"]),
        "color_span": tuple(r["color_span"])}) for r in old.read(root / "contexts.json")]
    plan, ro = old.read(root / "plan.json"), old.read(root / "readout_plan.json")
    pairs = [schema.Pair(**r) for r in plan["pairs"]]
    return contexts, pairs, plan, ro


def caches(plan):
    natural = desc.load_observed(old.ROOT / "operations/natural")
    selected = plan["original_pair_indices"]
    base = desc.load_observed(desc.ROOT / "operations/matched_base")
    direct = {m: desc.load_observed(desc.ROOT / "operations" / (m + "_direct")) for m in METHODS}
    if not natural.predictions["correct"][plan["valid_contexts"]].all() or not base.predictions["correct"][selected].all():
        raise ValueError("Approved strict baseline-only mask no longer holds")
    return natural, base, direct
