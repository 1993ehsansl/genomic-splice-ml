# GenomicSpliceML evaluation — UCI_public


**Data samples:** 3190, SHA256 `ebbae10c85d3c285e2a2489a37e42bc2ee8d91d15005e391a9962fa3389a114f`
**Train/validation/test sizes:** 1914/638/638
**Seed:** 42 — two-stage StratifiedGroupKFold, grouped by exact DNA sequence

## Candidate models (VALIDATION only)

| Model | macro-F1 | Accuracy | ECE |
|---|---:|---:|---:|
| positional_markov | 0.934 | 0.937 | 0.023 |
| position_logreg | 0.929 | 0.937 | 0.026 |
| kmer_logreg | 0.618 | 0.638 | 0.225 |
| prior_baseline | 0.228 | 0.519 | 0.000 |

## Final HELD-OUT test — winner: positional_markov

- **Test macro-F1:** 0.941 (bootstrap 95% CI 0.919–0.959)
- **Test accuracy:** 0.944
- **Test macro ROC-AUC (OvR):** 0.99108
- **Test balanced accuracy:** 0.936
- **Test calibration ECE (10-bin):** 0.011
- **Test macro-F1 prior baseline:** 0.228

## Interpretation and limitations

The validation scores select a model. The final test was not used in model selection. This is **NOT** a clinical benchmark.

- Small, historic benchmark; not a clinical or diagnostic model.
- No patient/gene IDs or external cohort; homologous sequences and gene leakage may remain.
- One seeded train/val/test split; repeat seeds or external validation before strong claims.
- Observed metrics must not be presented as drug-discovery, patient, or real-world clinical outcomes.

## Outputs

- `model_comparison.png` — validation scores
- `test_confusion_matrix.png` — held-out errors
- `train_motif_heatmaps.png` — class-conditional genomic motifs
- `metrics.json` — all structured values and provenance
