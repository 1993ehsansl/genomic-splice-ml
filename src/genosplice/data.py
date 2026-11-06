"""Data retrieval, parsing and quality checks for the UCI splice benchmark."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import urllib.request
import zipfile
import io
from collections import Counter
import numpy as np

LABELS = ('EI', 'IE', 'N')
ALLOWED = set('ACGTRYSWKMBDHVN')  # ambiguous IUPAC symbols map to unknown feature
DATA_URLS = (
    'https://archive.ics.uci.edu/static/public/69/molecular+biology+splice+junction+gene+sequences.zip',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data',
    'https://huggingface.co/datasets/mstz/splice/resolve/main/splice.data',
)

@dataclass(frozen=True)
class Dataset:
    sequences: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray


def download(path: str | Path, timeout: int = 40) -> tuple[str, str]:
    """Download the public UCI dataset; return (source URL, SHA256)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'GenomicSpliceML/0.1 (educational project)'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read()
            if body[:2] == b'PK':
                with zipfile.ZipFile(io.BytesIO(body)) as z:
                    matches = [n for n in z.namelist() if n.endswith('splice.data')]
                    if not matches:
                        raise ValueError('UCI zip did not contain splice.data')
                    body = z.read(matches[0])
            if b'EI' not in body[:2000] or len(body) < 100_000:
                raise ValueError('Unexpected dataset content or incomplete download')
            path.write_bytes(body)
            return url, hashlib.sha256(body).hexdigest()
        except Exception as exc:
            errors.append(f'{url} ({type(exc).__name__}: {exc})')
    raise RuntimeError('All public download sources failed:\n' + '\n'.join(errors))


def load(path: str | Path) -> Dataset:
    sequences, labels, ids = [], [], []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(',', 2)
        if len(parts) != 3:
            raise ValueError(f'Line {line_no}: expected label, id, sequence')
        label, sid, sequence = (p.strip() for p in parts)
        sequence = ''.join(sequence.upper().split())
        if label not in LABELS or not sid or len(sequence) != 60 or set(sequence) - ALLOWED:
            raise ValueError(f'Line {line_no}: invalid record (label={label!r}, length={len(sequence)})')
        labels.append(label)
        ids.append(sid)
        sequences.append(sequence)
    if not sequences:
        raise ValueError('Dataset empty')
    return Dataset(np.array(sequences), np.array(labels), np.array(ids))


def data_report(data: Dataset) -> dict:
    counts = Counter(data.labels.tolist())
    sequences = data.sequences.tolist()
    conflicts = {}
    for seq, label in zip(sequences, data.labels.tolist()):
        conflicts.setdefault(seq, set()).add(label)
    return {
        'n_samples': len(sequences), 'length': len(sequences[0]),
        'class_counts': {k: int(counts.get(k, 0)) for k in LABELS},
        'unique_sequences': len(set(sequences)),
        'duplicate_sequences': len(sequences) - len(set(sequences)),
        'conflicting_labels_for_identical_sequence': sum(len(x) > 1 for x in conflicts.values()),
        'ambiguous_base_fraction': round(sum(c not in 'ACGT' for s in sequences for c in s) / (len(sequences)*60), 6),
    }
