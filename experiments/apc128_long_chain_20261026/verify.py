"""Standard-library-only integrity and syntax checks; no model/GPU execution."""
import ast
import hashlib
import json
from pathlib import Path


def verify():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    files = manifest['source_files']
    expected = {v['path'] for v in files}
    actual = {p.relative_to(root).as_posix() for p in (root/'source').rglob('*') if p.is_file()}
    if len(expected) != len(files) or actual != expected:
        raise ValueError('Source inventory changed')
    parsed = 0
    for row in files:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root/'source') or path.suffix not in ('.py', '.ps1', '.md'):
            raise ValueError('Unexpected path or file type')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Source bytes changed: '+row['path'])
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=row['path'])
            compile(data, row['path'], 'exec')
            parsed += 1
    result = dict(passed=True, files=len(files), python_sources_compiled=parsed,
                  sealed_sources=manifest['source_registered_files'],
                  model_or_GPU_code_executed=False, original_source_bytes_preserved=True)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify()
