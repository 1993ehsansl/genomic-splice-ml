"""Biologically interpretable sequence motif summaries (train split only)."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def plot_motif_frequencies(training_motifs: dict, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    classes = [c for c in ('EI', 'IE', 'N') if c in training_motifs]
    fig, axes = plt.subplots(len(classes), 1, figsize=(11, 2.7 * len(classes)), squeeze=False)
    for idx, label in enumerate(classes):
        data = training_motifs[label]
        mat = np.array([[rec[base] for rec in data['frequencies']] for base in 'ACGT'])
        ax = axes[idx, 0]
        image = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_yticks(range(4), list('ACGT'))
        ax.set_xticks(range(len(data['positions_zero_based'])), [str(x - 30) for x in data['positions_zero_based']])
        ax.set_title(f'{label} — nucleotide frequency around the centered junction (training examples only)')
        ax.set_xlabel('Relative nucleotide position (0 = 31st base)')
        fig.colorbar(image, ax=ax, label='frequency', fraction=0.02)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'train_motif_heatmaps.png', dpi=160)
    plt.close(fig)


def render_report(report: dict, output_dir):
    out = Path(output_dir) / 'RESULTS.md'
    rows = []
    for name, scores in sorted(report['validation_metrics'].items(), key=lambda x: -x[1]['macro_f1']):
        rows.append(f"| {name} | {scores['macro_f1']:.3f} | {scores['accuracy']:.3f} | {scores['ece_10bins']:.3f} |")
    test = report['final_test_metrics']
    base = report['baseline_test_metrics']
    ci = report['macro_f1_bootstrap']['95pct_ci_macro_f1']
    caution = '⚠ SYNTHETIC demonstration: never quote as real-world evidence.\n\n' if not report['cv_claims_allowed'] else ''
    lines = [f'# GenomicSpliceML evaluation — {report["source_type"]}', '', caution,
             f'**Data samples:** {report["quality_audit"]["n_samples"]}, SHA256 `{report["data_sha256"]}`',
             f'**Train/validation/test sizes:** {report["split"]["train"]}/{report["split"]["validation"]}/{report["split"]["test"]}',
             f'**Seed:** {report["seed"]} — {report["split"]["strategy"]}', '',
             '## Candidate models (VALIDATION only)', '', '| Model | macro-F1 | Accuracy | ECE |', '|---|---:|---:|---:|', *rows, '',
             f'## Final HELD-OUT test — winner: {report["model_selected_on_validation_only"]}', '',
             f'- **Test macro-F1:** {test["macro_f1"]:.3f} (bootstrap 95% CI {ci[0]:.3f}–{ci[1]:.3f})',
             f'- **Test accuracy:** {test["accuracy"]:.3f}',
             f'- **Test macro ROC-AUC (OvR):** {test["macro_auc_ovr"]}',
             f'- **Test balanced accuracy:** {test["balanced_accuracy"]:.3f}',
             f'- **Test calibration ECE (10-bin):** {test["ece_10bins"]:.3f}',
             f'- **Test macro-F1 prior baseline:** {base["macro_f1"]:.3f}',
             '', '## Interpretation and limitations', '',
             'The validation scores select a model. The final test was not used in model selection. This is **NOT** a clinical benchmark.',
             '', *[f'- {item}' for item in report['limitations']], '',
             '## Outputs', '', '- `model_comparison.png` — validation scores',
             '- `test_confusion_matrix.png` — held-out errors',
             '- `train_motif_heatmaps.png` — class-conditional genomic motifs',
             '- `metrics.json` — all structured values and provenance', '']
    out.write_text('\n'.join(lines), encoding='utf-8')
