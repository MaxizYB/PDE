"""Copy the scoped research snapshot without touching source files.

This is a handoff utility, not a training entry point. Network credentials
are discovered from the source AGENTS.md and redacted in copied text only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


TREES = [
    'CDMC/experiments/multiscale_elliptic_v1',
    'CDMC/experiments/v2',
    'baselines/data/multiscale_elliptic_v1',
    'CDMC/reports/assets/multiscale_elliptic_20261104',
    'CDMC/reports/talk_20261105/assets',
]
FILES = [
    'AGENTS.md',
    'baselines/data/darcy_v2/generate.py',
    'baselines/data/darcy_v2/SPEC.md',
    'baselines/data/elasticity_v1/generate.py',
    'CDMC/reports/CDMC_NATURE_MANUSCRIPT_EXTENDED_20261027.md',
    'CDMC/reports/CDMC_PAPER_ARGUMENT_BLUEPRINT.md',
    'CDMC/reports/CDMC_REPORT_TALK_20261105.md',
    'CDMC/reports/METHOD_WALKTHROUGH_20261105.pdf',
    'CDMC/reports/METHOD_WALKTHROUGH_20261105.html',
    'CDMC/reports/CDMC_NATURE_MANUSCRIPT.md',
    'CDMC/methods/v1/EXPERIMENTS_V1.md',
    'CDMC/methods/v1/NOVELTY_AND_BOUNDARIES.md',
    'CDMC/methods/v1/POSITIONING_AND_EVALUATION_V1.md',
    'CDMC/methods/v1/METHOD_V1_1_DETAILED.md',
    'CDMC/experiments/main_benchmark_v1/results/PRELIMINARY_VALIDATION_RESULTS_V1.md',
    'CDMC/experiments/main_benchmark_v1/results/HPM_HEAT_IRREGULAR_CURRENT_REPORT.md',
    'Nature/RESEARCH_PROTOCOL.md',
    'Nature/RESEARCH_TEMPLATES.md',
]
EXCLUDED_PARTS = {'.git', '__pycache__', '.venv', '.cache'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect(source):
    files = set(FILES)
    for tree in TREES:
        root = source / tree
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.update(p.relative_to(source).as_posix() for p in root.rglob('*')
                     if p.is_file() and not p.is_symlink())
    return sorted(rel for rel in files
                  if not EXCLUDED_PARTS.intersection(Path(rel).parts)
                  and Path(rel).suffix not in EXCLUDED_SUFFIXES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    target = Path(__file__).resolve().parents[1]
    manifest_path = target / 'handoff/SNAPSHOT_MANIFEST.json'
    if manifest_path.exists():
        raise FileExistsError('Snapshot already exists; do not overwrite it.')
    original_instructions = (source / 'AGENTS.md').read_text()
    passwords = re.findall(r'密码[：:]\s*([^\s，。]+)', original_instructions)
    addresses = re.findall(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])', original_instructions)
    replacements = {value: '<REDACTED_SERVER_IP>' for value in addresses}
    replacements.update({value: '<REDACTED_PASSWORD>' for value in passwords})
    if not passwords or not addresses:
        raise RuntimeError('Could not identify expected secrets; review source manually.')
    records = []
    missing = []
    for rel in collect(source):
        src, dst = source / rel, target / rel
        if not src.is_file():
            missing.append(rel)
            continue
        if dst.exists():
            raise FileExistsError(dst)
        raw = src.read_bytes()
        copied = raw
        redacted = False
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            content = None
        if content is not None:
            for secret, replacement in replacements.items():
                if secret in content:
                    content = content.replace(secret, replacement)
                    redacted = True
            if redacted:
                copied = content.encode('utf-8')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if redacted:
            dst.write_bytes(copied)
            shutil.copymode(src, dst)
        else:
            shutil.copy2(src, dst)
        if src.read_bytes() != raw:
            raise RuntimeError('Source changed during snapshot: ' + rel)
        records.append(dict(path=rel, source_sha256=sha(raw), snapshot_sha256=sha(copied),
                            bytes=len(copied), credential_redacted=redacted))
    leaks = []
    for path in target.rglob('*'):
        if path.is_file() and '.git' not in path.relative_to(target).parts:
            raw = path.read_bytes()
            if any(secret.encode() in raw for secret in replacements):
                leaks.append(path.relative_to(target).as_posix())
    if leaks:
        raise RuntimeError('Credential audit failed in files: ' + ', '.join(leaks))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(dict(
        snapshot_date='2026-11-06', layout='Original relative paths preserved',
        source_policy='Read-only source; only copied credentials redacted',
        files=records, missing_selected_files=missing,
        file_count=len(records), total_bytes=sum(r['bytes'] for r in records),
        credential_audit='No original server IP or password detected in snapshot files',
    ), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(copied=len(records), missing=missing,
                          redacted_files=sum(r['credential_redacted'] for r in records),
                          bytes=sum(r['bytes'] for r in records)), ensure_ascii=False))


if __name__ == '__main__':
    main()
