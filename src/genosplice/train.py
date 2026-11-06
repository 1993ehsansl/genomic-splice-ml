"""Train reproducible benchmarks: python -m genosplice.train --download --out artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import joblib
from .data import load, download, data_report, LABELS
from .models import model_candidates
from .evaluation import metrics, bootstrap_macro_f1, save_plots
from .features import motif_frequencies
from .interpret import plot_motif_frequencies, render_report


def grouped_split(sequences, labels, seed=42):
    """80/20 then ~75/25 of development (60/20/20 overall) group-disjoint."""
    groups = np.asarray(sequences)
    y = np.asarray(labels)
    all_ids = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(all_ids, y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(dev_idx, y[dev_idx], groups[dev_idx]))
    tr, va = dev_idx[train_rel], dev_idx[val_rel]
    for left, right in [(tr, va), (tr, test_idx), (va, test_idx)]:
        if set(groups[left]).intersection(groups[right]):
            raise AssertionError('Sequence leakage in grouped split')
    for part in (tr, va, test_idx):
        if set(y[part]) != set(LABELS):
            raise ValueError('A split lacks classes: try another random seed')
    return tr, va, test_idx


def run(path, output, seed=42, cnn=False, cnn_epochs=20, source_type='UCI_public'):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    ds = load(path)
    audit = data_report(ds)
    if source_type == 'UCI_public' and audit['n_samples'] != 3190:
        raise ValueError(f'Expected 3190 public UCI samples, got {audit["n_samples"]}.')
    tr, va, te = grouped_split(ds.sequences, ds.labels, seed)
    Xtr, ytr = ds.sequences[tr], ds.labels[tr]
    Xva, yva = ds.sequences[va], ds.labels[va]
    Xte, yte = ds.sequences[te], ds.labels[te]
    validation, models = {}, {}
    for name, model in model_candidates(seed).items():
        model.fit(Xtr, ytr)
        val = metrics(yva, model.predict_proba(Xva), model.classes_)
        validation[name] = val
        models[name] = model
        print(f'{name:20s} validation macro-F1: {val["macro_f1"]:.3f}')
    if cnn:
        from .cnn import DNAConvClassifier
        model = DNAConvClassifier(seed=seed, epochs=cnn_epochs).fit(Xtr, ytr, Xva, yva)
        validation['cnn_1d'] = metrics(yva, model.predict_proba(Xva), model.classes_)
        models['cnn_1d'] = model
        (out / 'cnn_training_history.json').write_text(json.dumps(model.history_, indent=2))
    # Select winner strictly by validation, then evaluate test once.
    name = max(validation, key=lambda k: validation[k]['macro_f1'])
    winner = models[name]
    probabilities = winner.predict_proba(Xte)
    test = metrics(yte, probabilities, winner.classes_)
    ci = bootstrap_macro_f1(yte, probabilities, winner.classes_, seed=seed)
    # Baseline evaluated on the same holdout for transparent comparison.
    base = models['prior_baseline']
    baseline_test = metrics(yte, base.predict_proba(Xte), base.classes_)
    joblib.dump(winner, out / 'model.joblib')
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {
        'project': 'GenomicSpliceML', 'dataset': 'UCI Molecular Biology (Splice-junction Gene Sequences)' if source_type == 'UCI_public' else 'SYNTHETIC DEMO DATA',
        'source_type': source_type, 'cv_claims_allowed': source_type == 'UCI_public',
        'data_sha256': digest, 'seed': seed, 'quality_audit': audit,
        'split': {'strategy': 'two-stage StratifiedGroupKFold, grouped by exact DNA sequence',
                  'train': len(tr), 'validation': len(va), 'test': len(te),
                  'note': 'No patient, gene or genomic locus IDs provided. Exact-sequence grouping is NOT a biological independence guarantee.'},
        'model_selected_on_validation_only': name,
        'validation_metrics': validation,
        'final_test_metrics': test,
        'baseline_test_metrics': baseline_test,
        'macro_f1_bootstrap': ci,
        'limitations': [
            'Small, historic benchmark; not a clinical or diagnostic model.',
            'No patient/gene IDs or external cohort; homologous sequences and gene leakage may remain.',
            'One seeded train/val/test split; repeat seeds or external validation before strong claims.',
            'Observed metrics must not be presented as drug-discovery, patient, or real-world clinical outcomes.',
        ],
    }
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    motifs = motif_frequencies(Xtr, ytr)
    (out / 'training_motifs.json').write_text(json.dumps(motifs, indent=2))
    save_plots(validation, test, out)
    plot_motif_frequencies(motifs, out)
    render_report(report, out)
    print(f'Winner: {name} | test macro-F1={test["macro_f1"]:.3f} | test accuracy={test["accuracy"]:.3f}')
    print(f'Results saved in {out.resolve()}')
    if source_type != 'UCI_public':
        print('WARNING: SYNTHETIC DEMO ONLY — metrics may NOT be quoted on a CV.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Genomic splice-junction prediction benchmark')
    parser.add_argument('--data', default='data/raw/splice.data')
    parser.add_argument('--download', action='store_true', help='Fetch the UCI splice dataset (internet required)')
    parser.add_argument('--demo', action='store_true', help='Use synthetic sanity-check data (not for CV results)')
    parser.add_argument('--out', default='artifacts')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cnn', action='store_true', help='Enable optional PyTorch CNN')
    parser.add_argument('--cnn-epochs', type=int, default=20)
    args = parser.parse_args(argv)
    path = Path(args.data)
    source = 'UCI_public'
    if args.demo:
        from .synthetic import write_demo
        path = Path(args.out) / '_demo' / 'splice_synthetic.data'
        write_demo(path, seed=args.seed)
        source = 'synthetic'
    elif args.download:
        url, sha = download(path)
        print(f'Fetched {url}\nSHA256: {sha}')
    if not path.exists():
        parser.error(f'Missing dataset {path}. Use --download for UCI, or --demo for a synthetic smoke test.')
    run(path, args.out, seed=args.seed, cnn=args.cnn, cnn_epochs=args.cnn_epochs, source_type=source)


if __name__ == '__main__':
    main()
