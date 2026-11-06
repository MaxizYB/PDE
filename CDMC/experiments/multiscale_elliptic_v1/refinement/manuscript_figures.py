"""Render manuscript figures from completed JSON results; run no experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[1]
OLD = EXP / 'results/refinement_20261103'
NEW = EXP / 'results/extension_20261103'
OUT = EXP.parents[1] / 'reports/assets/multiscale_elliptic_20261104'
SOURCES = {}
KEYS = ['L2_percent', 'gradient_percent', 'flux_L2_percent']
LABELS = ['Field error (%)', 'Gradient error (%)', 'Flux error (%)']
COLORS = ['#245c87', '#c06c35', '#508462']


def read(path):
    SOURCES[str(path.relative_to(EXP))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def mean(doc, method, key, count=None):
    return statistics.mean(row['methods'][method][key] for row in doc['samples'][:count])


def save(fig, name):
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), constrained_layout=True)
    models = ['FNO', 'MgNO', 'DCNO', 'M2NO']
    for row, family in enumerate(['smooth', 'rough']):
        for i, model in enumerate(models):
            if model in ('FNO', 'MgNO'):
                suffix = 'fno_h1' if model == 'FNO' else 'mgno'
                path = OLD / family / f'frozen128_{suffix}/evaluation.json'
            else:
                path = NEW / family / f'{model}_corrected/evaluation.json'
            doc = read(path)
            for col, key in enumerate(KEYS):
                ax = axes[row, col]
                before, after = [mean(doc, m, key) for m in (model, 'dictionary_K2')]
                ax.plot([i, i], [after, before], color='#b8bec5', linewidth=2)
                ax.scatter(i, before, color='#777777', marker='o', s=34,
                           label='NO' if i == 0 else None, zorder=3)
                ax.scatter(i, after, color=COLORS[col], marker='s', s=34,
                           label='NO + CDMC K2' if i == 0 else None, zorder=3)
                ax.set_yscale('log')
                ax.set_xticks(range(4), models)
                ax.set_ylabel(LABELS[col])
                ax.set_title(f'{family.capitalize()} two-phase material')
                ax.grid(axis='y', alpha=.15)
        axes[row, 0].legend(frameon=False, fontsize=8)
    save(fig, 'cross_predictor')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    configs = [
        ('Trigonometric', OLD / 'balanced128_test32/evaluation.json',
         OLD / 'trigonometric_gpu_amg/evaluation.json'),
        ('Smooth two-phase', OLD / 'smooth/frozen128_fno_h1/evaluation.json',
         OLD / 'smooth_gpu_amg/evaluation.json'),
        ('Rough two-phase', OLD / 'rough/frozen128_fno_h1/evaluation.json',
         OLD / 'rough/gpu_amg/evaluation.json')]
    for ax, (label, ours_path, amg_path) in zip(axes, configs):
        ours, amg = read(ours_path), read(amg_path)
        assert [s['index'] for s in ours['samples'][:16]] == [s['index'] for s in amg['samples']]
        for key, color, metric in zip(KEYS, COLORS, ['Field', 'Gradient', 'Flux']):
            for doc, prefix, style, marker in [(ours, 'dictionary_K', '-', 'o'),
                                                (amg, 'NO_GPU_AMG_K', '--', 's')]:
                names = [prefix + str(k) for k in (2, 4)]
                x = [mean(doc, m, 'total_ms', 16) for m in names]
                y = [mean(doc, m, key, 16) for m in names]
                ax.plot(x, y, color=color, linestyle=style, marker=marker,
                        label=metric + (' / CDMC' if style == '-' else ' / AMG'))
        ax.set_title(label + ' (n=16)')
        ax.set_xlabel('Complete online time (ms)')
        ax.set_yscale('log')
        ax.set_ylabel('Relative error (%)')
        ax.grid(alpha=.15)
    axes[1].legend(frameon=False, fontsize=7, ncols=2)
    save(fig, 'accuracy_cost')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    docs = [read(NEW / f'matched_{v}_test/evaluation.json') for v in ['linear', 'nonlinear']]
    x = np.arange(3)
    for i, (doc, label, color) in enumerate(zip(docs, ['Linear readout', 'Nonlinear readout'], ['#999999', '#245c87'])):
        axes[0].bar(x + (i-.5)*.33, [mean(doc, 'dictionary_K2', k) for k in KEYS],
                    width=.33, color=color, label=label)
    axes[0].set_xticks(x, ['Field', 'Gradient', 'Flux'])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Relative error (%)')
    axes[0].set_title('A  Matched readout ablation (K2)')
    axes[0].legend(frameon=False, fontsize=8)
    mesh = read(NEW / 'mesh_transfer/evaluation.json')
    names = ['interpolated_FNO', 'filter_only_K2', 'dictionary_K2', 'dictionary_K4']
    for key, label, color in zip(KEYS, ['Field', 'Gradient', 'Flux'], COLORS):
        axes[1].plot(range(4), [mean(mesh, n, key) for n in names], color=color,
                     marker='o', label=label)
    axes[1].set_xticks(range(4), ['NO', 'Filter', 'K2', 'K4'])
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Relative error (%)')
    axes[1].set_title('B  Retriangulated mesh (n=16)')
    axes[1].legend(frameon=False, fontsize=8)
    diagnosis = read(NEW / 'contrast_diagnosis.json')
    for i, (sweeps, label) in enumerate([('1', 'Original calibration'), ('4', 'Four tiny solves')]):
        values = [100*statistics.mean(s['captured_energy_fraction'][sweeps] for s in case['samples'])
                  for case in diagnosis['cases']]
        axes[2].bar(np.arange(2) + (i-.5)*.33, values, width=.33,
                    color=['#245c87', '#8baec5'][i], label=label)
    axes[2].set_xticks(range(2), ['Contrast 6', 'Contrast 60'])
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel('Captured remaining A-energy (%)')
    axes[2].set_title('C  Response coverage (n=4)')
    axes[2].legend(frameon=False, fontsize=8)
    save(fig, 'mechanism_transfer')
    (OUT / 'SOURCES.json').write_text(json.dumps(SOURCES, indent=2) + '\n')
    print(json.dumps({'figures': 3, 'source_files': len(SOURCES), 'output': str(OUT)}))


if __name__ == '__main__':
    main()
