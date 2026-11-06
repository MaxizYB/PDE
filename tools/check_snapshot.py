"""Check the handoff using only Python's standard library.

No torch imports, package installs, networking, or source-tree changes.
Optional --source also checks the original files and original secret values.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'handoff/SNAPSHOT_MANIFEST.json').read_text())
    errors = []
    for entry in manifest['files']:
        target = root / entry['path']
        if not target.is_file() or digest(target) != entry['snapshot_sha256']:
            errors.append('Snapshot mismatch: ' + entry['path'])
        if args.source:
            original = args.source.resolve() / entry['path']
            if not original.is_file() or digest(original) != entry['source_sha256']:
                errors.append('Source mismatch: ' + entry['path'])

    syntax_count = 0
    for path in root.rglob('*.py'):
        if {'.git', '__pycache__', '.venv', 'runs', 'local_checks'}.intersection(path.relative_to(root).parts):
            continue
        try:
            ast.parse(path.read_bytes(), filename=str(path.relative_to(root)))
            syntax_count += 1
        except (SyntaxError, UnicodeError) as exc:
            errors.append('Python syntax: ' + str(path.relative_to(root)) + ': ' + str(exc))

    if args.source:
        original_agents = (args.source / 'AGENTS.md').read_text()
        secrets = re.findall(r'密码[：:]\s*([^\s，。]+)', original_agents)
        secrets += re.findall(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])', original_agents)
        if not secrets:
            errors.append('No source secrets identified for verification')
        for path in root.rglob('*'):
            if not path.is_file() or '.git' in path.relative_to(root).parts:
                continue
            data = path.read_bytes()
            if any(value.encode() in data for value in secrets):
                errors.append('Original credential value present: ' + path.relative_to(root).as_posix())
    agents = (root / 'AGENTS.md').read_text()
    if '<REDACTED_SERVER_IP>' not in agents or '<REDACTED_PASSWORD>' not in agents:
        errors.append('Redacted AGENTS placeholders missing')

    exp = Path('CDMC/experiments/multiscale_elliptic_v1')
    key_assets = [
        ('smooth FNO', exp / 'results/refinement_20261103/smooth/fno_h1/no.pt'),
        ('rough FNO', exp / 'results/refinement_20261103/rough/fno_h1/no.pt'),
        ('smooth HINTS/TB', exp / 'results/hints_20261104/smooth/model.pt'),
        ('rough HINTS/TB', exp / 'results/hints_20261104/rough/model.pt'),
        ('CDMC128', exp / 'results/refinement_20261103/balanced128_train/model.pt'),
        ('smooth frame', exp / 'results/stable_space_20261104/smooth/normal_full/model.pt'),
        ('rough frame', exp / 'results/stable_space_20261104/rough/normal_full/model.pt'),
    ]
    assets = [dict(name=name, path=path.as_posix(), included=(root / path).is_file(),
                   use_server_original_if_missing=True) for name, path in key_assets]
    report = dict(
        date='2026-11-06', passed=not errors,
        copied_files_checked=manifest['file_count'],
        copied_bytes=manifest['total_bytes'],
        source_hashes_checked=bool(args.source),
        credential_values_checked=bool(args.source),
        redacted_files=[v['path'] for v in manifest['files'] if v['credential_redacted']],
        python_files_syntax_checked=syntax_count,
        local_checkpoints=sorted(p.relative_to(root).as_posix() for p in root.rglob('*.pt')),
        key_assets=assets, errors=errors,
        scope='Integrity and syntax only; not a GPU training or performance verification',
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {'local_checkpoints', 'key_assets'}}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
