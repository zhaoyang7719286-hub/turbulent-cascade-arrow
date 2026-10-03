from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/numerics'))
import spectral_dns_core as dns
import df_phase_randomization_gate_64cube as gate
import population_amplitude as pop
import run_dynamic_recovery_smoke as dynamic


class NumericalChecks(unittest.TestCase):
    def test_phase_reality_and_two_time_gram(self):
        operators = dns.make_operators(12)
        fields = [dns.random_initial_condition(operators, seed=s, target_energy=0.5, spectral_peak=3, workers=1) for s in [13, 17]]
        phase = gate.make_hermitian_phase((12, 12, 12), seed=123)
        transformed = [u * phase[None, ...] for u in fields]
        for u in transformed:
            self.assertLess(np.max(np.abs(dns.ifft_vector_complex(u).imag)), 1e-12)
        for a in range(2):
            for b in range(2):
                original = np.einsum('ixyz,jxyz->ijxyz', fields[a], fields[b].conj())
                after = np.einsum('ixyz,jxyz->ijxyz', transformed[a], transformed[b].conj())
                np.testing.assert_allclose(original, after, atol=1e-10, rtol=2e-12)

    def test_population_amplitude_identity(self):
        result = pop.pop_amp_statistics(np.array([3.0, 1.0, -2.0, -4.0]))
        self.assertEqual(result['n_zero'], 0)
        self.assertLess(result['identity_abs_error'], 1e-14)

    def test_saved_checkpoint_initial_observables(self):
        # Compare actual supplied field diagnostics to the paper's recorded t=0.
        source = pd.read_csv(ROOT / 'data/paper_sources/figure_ready/dynamic_recovery/heldout_3x3_raw_long.csv')
        operators = dns.make_operators(64)
        for checkpoint in [1750, 2250, 2750]:
            u, _, _ = dynamic.load_checkpoint(ROOT / f'data/checkpoints/checkpoint_step_{checkpoint:07d}.npz')
            for seed in [2026074101, 2026074102, 2026074103]:
                randomized, audit = dynamic.phase_randomize_spectral(u, operators, seed)
                self.assertLess(audit['energy_relative_error'], 1e-12)
                for branch, field in [('natural', u), ('nonlinear_recovery', randomized)]:
                    with self.subTest(checkpoint=checkpoint, seed=seed, branch=branch):
                        selected = source[(source.checkpoint == checkpoint) & (source.phase_seed == seed) &
                                          (source.branch == branch) & np.isclose(source.relative_time, 0)]
                        self.assertEqual(len(selected), 1)
                        result = dynamic.arrow_metrics(field, operators, scale=8, viscosity=0.015, workers=1)
                        for metric in ['Pi_normalized_mean', 'M_normalized_mean', 'corr_Pi_M', 'energy', 'dissipation_rate']:
                            self.assertAlmostEqual(result[metric], float(selected.iloc[0][metric]), delta=2e-11)


if __name__ == '__main__':
    unittest.main()
