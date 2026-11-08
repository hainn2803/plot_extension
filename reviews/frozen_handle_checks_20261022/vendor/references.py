import ast
from pathlib import Path
import re
import string
import unicodedata
import torch

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
