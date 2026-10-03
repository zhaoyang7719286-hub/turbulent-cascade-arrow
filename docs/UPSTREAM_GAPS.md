# Upstream dependencies and coverage

The verified handoff archive supplied 35 Python/shell source files, figure-ready results, numerical audit records, and three Fourier checkpoints. It did not supply the full static NPZ frame sequences or recovery-time Fourier sequences.

The legacy wrappers reference several absent files, notably:

- `eulerian_finite_time_arrow_axisfixed_64cube.py`: required by the original finite-window V3, population/amplitude CLI, and Clark-mechanism CLI.
- `make_projected_reference_dataset.py`: invoked by the independent static replication shell script.
- `write_phase_hash_64cube.py`: invoked by the same replication script.

Additional unresolved references are listed by path in `docs/legacy_dependency_inventory.json`. `code/legacy/` keeps the supplied source inventory, with private host paths sanitized. It is not advertised as a working monolithic launcher. Acquisition scripts also depend on historical `giverny`/`givernylocal` clients and a user-provided JHTDB token; no online acquisition was performed in validating this release.

The dynamic branch used only the pure `pop_amp_statistics` function from the missing-module import chain. That function was extracted verbatim to `code/numerics/population_amplitude.py`. The dynamic module now imports it without importing the incomplete static CLI. Spectral kernels, phase construction, evolution, observables, and recovery definitions were preserved. Initial-field diagnostics are tested directly against the paper's recorded table.

Restoring the static pipeline requires obtaining the actual missing upstream files and source cutouts, then verifying original frame/seed/projection identities and reproducing the recorded observables. No approximate replacement is passed off as the original module here.
