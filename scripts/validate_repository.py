#!/usr/bin/env python3
"""Check file integrity and scientific table contracts, without external services."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root=ROOT):
    entries = json.loads((root / 'file_manifest.json').read_text())
    for relative, expected in entries.items():
        path = root / relative
        require(path.is_file(), f'Missing file: {relative}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Hash mismatch: {relative}')
    base = root / 'data/paper_sources/figure_ready'
    masks = pd.read_csv(base / 'main32/seed_metrics_64.csv')
    summary = pd.read_csv(base / 'main32/ensemble_summary_64.csv')
    require(masks.seed.nunique() == 32, 'Expected 32 primary masks')
    for r in summary.itertuples():
        v = masks[(masks.scale == r.scale) & (masks.metric == r.metric) &
                  (masks.window_frames == r.window_frames) & (masks.statistic_type == r.statistic_type)].value
        require(len(v) == 32 and np.isfinite(v).all(), 'Invalid primary group')
        require(np.isclose(v.mean(), r.surrogate_mean, atol=1e-12, rtol=1e-12), 'Primary mean mismatch')
        require(np.isclose(v.std(ddof=1), r.surrogate_std, atol=1e-12, rtol=1e-12), 'Primary SD mismatch')
    cases = pd.read_csv(base / 'isotropic_rep2/isotropic_rep2_all_case_metrics.csv')
    six = pd.read_csv(base / 'isotropic_rep2/isotropic_rep2_primary_six.csv')
    for r in six.itertuples():
        v = cases[(cases.scale == r.scale) & (cases.case_type == 'surrogate')][r.metric]
        ref = cases[(cases.scale == r.scale) & (cases.case_type == 'reference_star32')][r.metric]
        require(len(v) == 24 and len(ref) == 1, 'Invalid independent group')
        require(np.isclose(float(ref.iloc[0]), r.comparator_value), 'Independent comparator mismatch')
        require(int((v >= ref.iloc[0]).sum()) == r.n_surrogate_ge_comparator, 'Independent rank mismatch')
    fail = six[(six.scale == 8) & (six.metric == 'mean_Pi')]
    require(len(fail) == 1 and int(fail.iloc[0].n_surrogate_ge_comparator) == 8, 'Raw-mean failure was lost')
    for step in [1750, 2250, 2750]:
        with np.load(root / f'data/checkpoints/checkpoint_step_{step:07d}.npz', allow_pickle=False) as z:
            require(z['uhat'].shape == (3, 64, 64, 64), 'Checkpoint shape mismatch')
            require(np.isfinite(z['uhat']).all() and int(z['step']) == step, 'Checkpoint content mismatch')
    report = {'status': 'PASS', 'files_checked': len(entries), 'primary_masks': 32, 'independent_masks': 24,
              'scope': 'File integrity, table consistency, and checkpoint structure; not a complete DNS rerun.'}
    return report


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
