"""Explicitly synthetic data for offline software smoke tests; NOT for CV claims."""
from pathlib import Path
import numpy as np


def write_demo(path, seed=42, n_per_class=100):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lines = []
    motifs = {'EI': 'GT', 'IE': 'AG', 'N': 'CC'}
    for label, motif in motifs.items():
        for i in range(n_per_class):
            bases = rng.choice(list('ACGT'), size=60)
            if label != 'N' or rng.random() < 0.3:
                bases[30:32] = list(motif)
            lines.append(f'{label},DEMO-{label}-{i},{"".join(bases)}')
    rng.shuffle(lines)
    path.write_text('\n'.join(lines) + '\n')
    return path
