# GenomicSpliceML: Interpretable Genomic Sequence Learning

**Course-inspired, open-data bioinformatics portfolio project** — compare probabilistic sequence models, interpretable machine learning and an optional 1D CNN to classify DNA splice junctions. Designed as a rigorous research-software exercise, not as a clinical diagnostic or drug-discovery tool.

> **Status:** Trained and evaluated on the public UCI splice-junction dataset (3,190 sequences). The validation-selected positional Markov model achieved 94.1% macro-F1, 94.4% accuracy, and 0.991 ROC-AUC on 638 held-out test sequences (seed 42). All 7 automated tests passed.

## Why this project?

Splicing is part of gene expression: models predict whether a 60-base primate DNA window contains an exon→intron **donor (EI)** junction, an intron→exon **acceptor (IE)** junction, or **neither (N)**. This is a controlled educational benchmark relevant to genomic sequence analysis. It is **not** a patient-level, variant pathogenicity, RNA-seq, or drug-response model.

**Public dataset:** UCI [Molecular Biology (Splice-junction Gene Sequences)](https://archive.ics.uci.edu/dataset/69/molecular-biology-splice-junction-gene-sequences), DOI [10.24432/C5M888](https://doi.org/10.24432/C5M888), CC BY 4.0. It contains **3,190** labelled, **60-nucleotide** windows. The original benchmark is historical (1991). We exclude the source data from Git so it can be retrieved at build time and credited appropriately.

### Core methods and what they demonstrate

| Method | Learning / bioinformatics concept |
|---|---|
| Class-prior-only classifier | Honest trivial baseline |
| **Position-dependent Markov model** with Laplace smoothing | DNA motif and generative sequence modeling; this is **not** a hidden Markov model |
| **Position-specific logistic regression** | Interpretable local splice-site sequence signals |
| Character 3–5-mer logistic regression | Motif discovery and feature engineering |
| **Optional PyTorch 1D CNN** | Local sequence motif learning with validation early stopping |
| Group-disjoint training/validation/test | Split leakage control on exact sequence duplicates |
| Accuracy, macro-F1, balanced accuracy, multiclass ROC-AUC, log loss, Brier, ECE | Fair model comparison, uncertainty and probability-quality diagnostics |
| Held-out confusion matrix and bootstrap 95% macro-F1 interval | Reproducible error analysis and uncertainty |
| FastAPI + Docker + GitHub Actions | Versioned, tested, reproducible Python ML application |

The project directly applies the University of Basel course chapters on DNA sequence analysis, Markov/probabilistic models (Ch. 2), and neural nets (Ch. 6). Pairwise/multiple alignments, phylogenetics, and NGS quasispecies (Chs. 1, 3–5) are valuable related domains, but are **not** claimed as methods used in this narrow classifier. Resist including unrelated algorithms solely to fill keywords.

## Requirements

- Python **3.10+** (recommended 3.11), Git
- Internet access once to download the public UCI data
- Docker only if running the API in a container

## Run it — macOS, Linux, Windows (PowerShell)

```bash
# From this repository root
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell alternative:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[api,dev]'

# 1. Download UCI's REAL 3,190-sequence dataset (requires internet)
python scripts/download_data.py
# 2. Benchmark class-prior, position-specific Markov, positional logistic and k-mer logistic
python -m genosplice.train --data data/raw/splice.data --out artifacts --seed 42
# 3. Optional: train and compare a PyTorch 1D CNN as well
pip install -e '.[deep]'
python -m genosplice.train --data data/raw/splice.data --out artifacts --seed 42 --cnn --cnn-epochs 20
# 4. Run tests
pytest -q
# 5. After real training, generate honest numbers for your CV
python scripts/cv_bullets.py --metrics artifacts/metrics.json
```

If the dataset is not directly downloadable, obtain **splice.data** manually from the [UCI dataset page](https://archive.ics.uci.edu/dataset/69/molecular-biology-splice-junction-gene-sequences), save it at `data/raw/splice.data`, and rerun without `--download`.

**No network / smoke test** (creates **synthetic data only**, not scientifically credible metrics):

```bash
python -m genosplice.train --demo --out artifacts/demo
```

## Outputs after running real training

- `artifacts/metrics.json`: dataset checksum, quality audit, sizes, class distribution, source provenance, leakage limitations, candidate **validation** metrics, independently held-out **test** metrics for only the validation-selected model, and a bootstrap macro-F1 CI.
- `artifacts/model.joblib`: selected fitted sklearn / PyTorch model (trusted local artifact only).
- `artifacts/model_comparison.png`: validation macro-F1 by model.
- `artifacts/test_confusion_matrix.png`: held-out test confusion matrix.
- `artifacts/training_motifs.json` and `train_motif_heatmaps.png`: training-set nucleotide frequencies near the splice junction; never calculated using test labels.
- `artifacts/RESULTS.md`: human-readable evaluation and limitations ready for scientific review.
- `artifacts/cnn_training_history.json`: only for optional CNN run.

To interpret the results: first compare held-out **macro-F1** against the prior baseline, then inspect class-specific errors, calibration, CI width, and how model choice changed when adding the CNN. The bootstrap CI is a non-clustered uncertainty approximation, **not** a confidence interval for generalization across genes, organisms, or patients.

### Evaluation integrity

1. Sequence duplicates are held together using `StratifiedGroupKFold` on the **exact DNA sequence**. Training and model selection use training/validation only; the final test is evaluated once after validation chooses a winner.
2. The historic benchmark lacks reliable **gene, patient, chromosome or sampling-group labels**. Duplicates can be controlled, but related sequences can remain across splits. No clinical robustness is implied. In particular, no external-cohort evaluation is present.
3. A group fold with a random seed is not a guarantee of population independence; results from a single seed can be variable. Repeat seeds and add genuinely independent genomics data before claiming robust real-world performance.
4. Any ambiguous IUPAC symbols are handled with a dedicated fifth category. No target leakage from sample identifiers (identifiers are never features).
5. All output metrics are **data-dependent**. The repository contains *no invented accuracy or ROC-AUC numbers*.

## Serve predictions using FastAPI

```bash
# after running REAL training once
uvicorn genosplice.api:app --reload
# Docs: http://127.0.0.1:8000/docs
# Example POST, 60 bases exactly
python - <<'PY'
import json, urllib.request
from pathlib import Path
sequence = Path('data/raw/splice.data').read_text().splitlines()[0].split(',', 2)[2]
sequence = ''.join(sequence.upper().split())
request = urllib.request.Request(
  'http://127.0.0.1:8000/predict',
  data=json.dumps({'sequence': sequence}).encode(),
  headers={'Content-Type': 'application/json'})
print(urllib.request.urlopen(request).read().decode())
PY
```

Build Docker **after training** (requires `artifacts/model.joblib`):

```bash
docker build -t genomic-splice-ml .
docker run --rm -p 8000:8000 genomic-splice-ml
```

Do not load models received from untrusted sources via `joblib.load`.

## GitHub publishing

```bash
# Create an EMPTY repository called genomic-splice-ml at https://github.com/new
cd genomic-splice-ml
git init
git branch -M main
git add .
git commit -m "Build tested genomics ML benchmark and inference API"
git remote add origin https://github.com/YOUR_USERNAME/genomic-splice-ml.git
git push -u origin main
```

Or with GitHub CLI, after `gh auth login`: `gh repo create genomic-splice-ml --public --source=. --remote=origin --push`.

Commit the **implementation** and README; the `.gitignore` intentionally prevents committing UCI raw data and trained model artifacts. If you later want to publish verified test results, add a non-sensitive *manually copied* `reports/metrics.json` and a carefully labelled evaluation summary with dataset source and random seed. Never upload private company/patient/clinical data or secrets.



**Genomic Splice-Site Prediction | Bioinformatics & Machine Learning**

###  Benchmark Results

Evaluated on the public UCI Splice-Junction dataset containing 3,190 DNA sequences.

- **Best model:** Position-dependent Markov classifier
- **Test accuracy:** 94.4%
- **Macro-F1:** 0.941 (95% bootstrap CI: 0.919–0.959)
- **ROC-AUC:** 0.991
- **Held-out test samples:** 638
- **Baseline macro-F1:** 0.228

Model selection used validation data, with exact duplicate sequences grouped across train, validation, and test splits.


See [Detailed Results](reports/RESULTS.md) and [Evaluation Metrics](reports/metrics.json).


## License and credits

Code: MIT (see LICENSE). Dataset: separate third-party UCI data under CC BY 4.0, citation above. Course materials: University of Basel Bioinformatics lectures — concepts used, not redistributed. No clinical usage approved.
