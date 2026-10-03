# Executed validation for the prepared release

Validation date: 2026-10-03 UTC. Environment: Python 3.12 with the exact versions in `requirements.txt`.

| Check | Executed result |
|---|---|
| Original archive | Complete 13,364,034-byte archive verified; recorded SHA-256 in config |
| Numeric input preservation | 59 archive-matched data tables and 682 numeric columns unchanged |
| Original spectral solver tests | All 14 checks pass, including Parseval, nonlinear energy conservation, forcing power, divergence, dealiasing, and RK3 checks |
| Added numerical tests | Phase reality/two-time Gram preservation, population-amplitude identity, and initial field observables pass |
| Initial dynamic data | All nine crossed checkpoint × mask cases checked; natural and perturbed Pi, M, correlation, energy, and dissipation agree with recorded t=0 values |
| Main figure reconstruction | 338 hash/table/mark checks pass |
| Supplemental reconstruction | 143 hash/table/mark checks pass |
| Current paper artwork | All ten reconstructed PDFs are pixel-identical to the corresponding current reference figures at 144 dpi in this environment |
| Short branch execution | Two-step NL/LIN/FRZ numerical smoke run passes; no t=1 endpoint is inferred from that run |
| Full dynamic execution | checkpoint 2250 × phase seed 2026074101, all branches, 300 steps, dt=0.01, 31 saved times through t=3; numerical checks pass |
| Full-case record comparison | Ten metric/group comparisons pass; largest absolute difference about 1.60e-15 |
| Clean ZIP extraction | Input verification, numerical tests, original solver tests, all ten figures, statistics, and short branch execution pass using the pinned installed environment |
| Optional statistical audit | Whole-mask interval and rank/deletion calculations execute successfully |
| Source and credential scan | Shipped Python files parse; no private host paths or known credential-pattern matches detected |

The full dynamic check covers one of the nine crossed cases. The other eight full trajectories were not rerun during packaging; their initial-field diagnostics and all supplied trajectory tables were checked. The executable `--all-cases` command remains available for a complete nine-case rerun.

The offline workflow does not retrieve original JHTDB cutouts or execute the incomplete upstream static pipeline. This limitation is separate from the successful figure reconstruction and selected full dynamic rerun.

`full_dynamic_comparison.json`, `numeric_input_comparison.json`, and `security_scan.json` give machine-readable preparation checks. Clean-extraction results are recorded in `clean_checkout_validation.json`.

Pixel identity describes this environment. Other platforms may render equivalent vector figures differently. GitHub Actions has been configured, but its hosted execution awaits the actual repository upload; the checks reported here are local executions.
