"""Emit honest CV lines only when data source is the verified public dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def build_bullets(metrics_path: str) -> str:
    report = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    if report.get('source_type') != 'UCI_public' or report.get('cv_claims_allowed') is not True:
        raise ValueError('Refusing CV claims: benchmark is synthetic or dataset provenance is unverified.')
    if report['quality_audit']['n_samples'] != 3190:
        raise ValueError('Refusing CV claims: expected public UCI 3190-sample dataset.')
    test = report['final_test_metrics']
    n = report['quality_audit']['n_samples']
    model = report['model_selected_on_validation_only'].replace('_', ' ')
    baseline = report['baseline_test_metrics']
    return '\n'.join([
        'Genomic Splice-Site Prediction | Bioinformatics & Machine Learning',
        f'- Developed Python DNA-sequence analysis workflows for {n:,} public primate samples, classifying donor, acceptor and non-splice regions with probabilistic models and interpretable ML.',
        f'- Benchmarked sequence models using group-disjoint validation; {model} achieved {test["macro_f1"]:.1%} macro-F1 and {test["accuracy"]:.1%} accuracy on {test["n"]} held-out sequences (prior-baseline macro-F1 {baseline["macro_f1"]:.1%}).',
        '- Implemented reproducible model comparison, motif analysis, bootstrap uncertainty estimates, FastAPI inference, Docker and automated GitHub Actions tests.',
        '',
        'Note: historical splice-junction benchmark, not external clinical validation. Confirm models/services you personally ran before using on a CV.',
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='artifacts/metrics.json')
    args = parser.parse_args()
    print(build_bullets(args.metrics))
