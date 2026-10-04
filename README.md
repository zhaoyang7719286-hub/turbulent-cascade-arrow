# Turbulent cascade arrow

Reproduction code and numerical records for **Erasing and Regenerating the Turbulent Cascade Arrow**.

The repository provides the recorded phase-mask results, initial dynamic fields, numerical kernels, and scripts that rebuild Main Figures 1–4 and Supplemental Figures S1–S6. The figure workflow runs offline. A separate entry point evolves matched NL/LIN/FRZ branches from the supplied checkpoints.

## Quick start

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/validate_repository.py
python -m unittest discover -s tests -v
python scripts/reproduce_figures.py
```

The final ten PDF/SVG figures are written to `outputs/paper_figures/figures/`. Exported panel tables, source-row references, checks, and normalization diagnostics are under `outputs/paper_figures/main/` and `outputs/paper_figures/supplement/`. The scripts refuse a nonempty figure output directory; pass `--out NEW_DIRECTORY` for another run.

The reference artwork in `figures/reference/` is the current paper figure set. The rebuilt figures preserve its data and layout, including removal of small footer audit notes. Rendering bytes can depend on the plotting, font, and PDF libraries; compare numerical values and plot regions rather than assuming PDF byte identity across systems.

## Run numerical checks and dynamic branches

```bash
python code/numerics/test_spectral_dns_core.py
# Short execution check on a real supplied checkpoint; two steps only:
python scripts/run_dynamic_cases.py --smoke
# Full recorded protocol for one checkpoint × mask:
python scripts/run_dynamic_cases.py --checkpoint 2250 --seed 2026074101 --out outputs/full_one_case
# All nine crossed cases, 300 steps each, output every 10 steps:
python scripts/run_dynamic_cases.py --all-cases --out outputs/full_nine_cases
```

Full dynamic runs can take substantially longer than table/figure reconstruction. They use the original viscosity, forcing power, time step, box filter, and phase seeds in `config/reproduction.json`. The default worker count is one; `--workers` changes FFT parallelism. New outputs are written separately from all recorded inputs. The nine checkpoint-by-mask cases share a simulated flow and are not nine independent DNS realizations.

## Reproduction coverage

| Workflow | Supplied inputs and executable scope |
|---|---|
| Main/Supplement figures | Rebuild all ten figures and panel tables from recorded results; verify sample means, SDs, recovery definitions, and original certified marks |
| Phase and solver checks | Run numerical preservation, solver, and initial-checkpoint observable checks |
| NL/LIN/FRZ evolution | Run the matched protocol from three supplied Fourier checkpoints; full nine-case validation status is recorded in `docs/VALIDATION.md` |
| Optional statistical audit | Recalculate whole-mask conditional intervals, six-comparison rank audit, and dynamic deletion checks; these are a later audit, not additional experiments in the current paper |
| Static raw-field recomputation | Not complete: original static NPZ frames and a legacy flux module were not in the handoff; see `docs/UPSTREAM_GAPS.md` |

Run the optional statistics with `python code/statistics/analyze_existing_tables.py`. Its 95% intervals describe mask Monte Carlo uncertainty conditional on a fixed recorded field and comparator. They are not confidence intervals across flows. Dynamic ranges and deletion checks are descriptive.

## Repository map

```text
code/numerics/         runnable original kernels and adapted imports
code/figures/          main/supplement builders
code/statistics/       optional existing-data audit
code/legacy/           sanitized upstream source inventory; some dependencies missing
data/checkpoints/      three original 64³ Fourier initial fields
data/paper_sources/    recorded figure tables, audits, and provenance records
data/statistics_inputs/ inputs for the optional audit
scripts/              reproducible entry points and integrity check
tests/                numerical and recorded-field checks
config/               original protocol and mask/case identifiers
docs/                 definitions, provenance, validation, publication notes
figures/reference/    current paper artwork
```

`source_lineage.csv` records upstream and prepared-file hashes and transformations. `file_manifest.json` freezes the prepared release files. The input copy changes host path strings where needed; numerical values remain unchanged. After an intentional source edit, regenerate the manifest with `python scripts/update_manifest.py` and review the diff before committing.

## Data and citation

The static flow originates from the Johns Hopkins Turbulence Database, https://turbulence.pha.jhu.edu/ . Dataset credit and acquisition coverage are described in `docs/DATA.md`. The repository does not contain an access token or redistribute the full JHTDB dataset. Numerical checkpoints are from the generated periodic experiment.

Repository URL, final authorship, and reuse license have not been assigned in this prepared version. 
