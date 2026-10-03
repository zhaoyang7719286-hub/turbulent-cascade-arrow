#!/usr/bin/env python3
"""Run matched NL/LIN/FRZ cases from supplied checkpoints. Explicit opt-in simulation."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=int, choices=[1750, 2250, 2750], default=2250)
    parser.add_argument('--seed', type=int, choices=[2026074101, 2026074102, 2026074103], default=2026074101)
    parser.add_argument('--all-cases', action='store_true', help='Run all nine crossed cases')
    parser.add_argument('--smoke', action='store_true', help='Two steps only; not a paper-result rerun')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/dynamic_runs')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    config = json.loads((ROOT / 'config/reproduction.json').read_text())
    cases = [(c, s) for c in config['checkpoints'] for s in config['phase_seeds']] if args.all_cases else [(args.checkpoint, args.seed)]
    target = args.out.resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {target}')
    target.mkdir(parents=True, exist_ok=True)
    parameters = dict(config['dynamic'])
    if args.smoke:
        parameters.update(n_steps=2, output_every=1)
    for checkpoint, seed in cases:
        case = target / f'checkpoint_{checkpoint}' / f'seed_{seed}'
        nl = case / 'nl_lin'
        frz = case / 'frz'
        checkpoint_file = ROOT / f'data/checkpoints/checkpoint_step_{checkpoint:07d}.npz'
        common = ['--checkpoint', str(checkpoint_file), '--phase_seed', str(seed), '--workers', str(args.workers)]
        common += ['--n_steps', str(parameters['n_steps']), '--output_every', str(parameters['output_every'])]
        for key in ['viscosity', 'epsilon_input', 'dt', 'scale']:
            common += ['--' + key, str(parameters[key])]
        subprocess.run([sys.executable, str(ROOT / 'code/numerics/run_dynamic_recovery_smoke.py'),
                        *common, '--outdir', str(nl)], check=True)
        subprocess.run([sys.executable, str(ROOT / 'code/numerics/run_amplitude_matched_frozen_counterfactual.py'),
                        *common, '--reference_csv', str(nl / 'recovery_diagnostics_all_branches.csv'),
                        '--outdir', str(frz)], check=True)
    (target / 'run_config.json').write_text(json.dumps({'cases': cases, 'smoke': args.smoke,
                                                      'parameters': parameters, 'workers': args.workers}, indent=2) + '\n')


if __name__ == '__main__':
    main()
