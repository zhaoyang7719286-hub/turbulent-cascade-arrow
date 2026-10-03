#!/usr/bin/env python3
"""Compare a generated full dynamic case against the corresponding archived tables."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compare(case_path, checkpoint, seed, tolerance=2e-9):
    data = ROOT / 'data/paper_sources/figure_ready/dynamic_recovery'
    original = pd.read_csv(data / 'heldout_3x3_raw_long.csv')
    original = original[(original.checkpoint == checkpoint) & (original.phase_seed == seed)]
    generated = pd.read_csv(case_path / 'nl_lin/recovery_diagnostics_all_branches.csv')
    original_frz = pd.read_csv(data / 'amplitude_matched_3x3_long.csv')
    original_frz = original_frz[(original_frz.checkpoint == checkpoint) & (original_frz.phase_seed == seed)]
    generated_frz = pd.read_csv(case_path / 'frz/amplitude_matched_counterfactual.csv')
    rows = []
    for kind, observed, reference, columns in [
        ('NL_LIN_natural', generated, original,
         ['Pi_normalized_mean','M_normalized_mean','corr_Pi_M','energy','dissipation_rate']),
        ('FRZ_replay', generated_frz, original_frz,
         ['Pi_normalized_mean','R_Pi_normalized_mean','corr_Pi_M','energy','dissipation_rate'])]:
        observed = observed.assign(time_key=np.rint(observed.relative_time * 1e10).astype(np.int64))
        reference = reference.assign(time_key=np.rint(reference.relative_time * 1e10).astype(np.int64))
        merged = observed.merge(reference, on=['branch','time_key'], suffixes=('_new','_recorded'), validate='one_to_one')
        if len(merged) != len(observed) or len(merged) != len(reference):
            raise ValueError('Incomplete or mismatched recorded time grid')
        if not np.allclose(merged.relative_time_new, merged.relative_time_recorded, rtol=0, atol=1e-10):
            raise ValueError('Time grids differ beyond serialization tolerance')
        for column in columns:
            diff = float(np.max(np.abs(merged[column+'_new'] - merged[column+'_recorded'])))
            rows.append(dict(group=kind, metric=column, max_absolute_difference=diff, pass_check=diff <= tolerance))
    if not all(r['pass_check'] for r in rows):
        raise ValueError(json.dumps(rows, indent=2))
    return dict(status='PASS', checkpoint=checkpoint, phase_seed=seed, tolerance=tolerance,
                scope='This one matched case, full recorded grid 0 to 3; not nine independent simulations', comparisons=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.case, args.checkpoint, args.seed), indent=2))
