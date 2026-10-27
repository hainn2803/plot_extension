"""Execute Jonathan's preserved class, rather than transcribing its equation."""
import ast
from pathlib import Path
import types
import torch

PATH = Path("eval/mcqa_internal_faithfulness_20260930_half_width/jonathan_source/experiments/mcqa/mcqa_experiment/intervention.py")


def jonathan_patch(base, source, basis):
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DASSubspaceIntervention")
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(PATH), "exec"), namespace)
    fixed = types.SimpleNamespace(project=types.SimpleNamespace(weight=basis.T))
    return namespace["DASSubspaceIntervention"].forward(fixed, base, source)
