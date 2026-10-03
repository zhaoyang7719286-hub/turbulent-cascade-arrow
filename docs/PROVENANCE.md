# Prepared-release provenance

Source archive: `JHTDB_LOCAL_HANDOFF_V1.tar(1).gz`.

SHA-256: `e0695c277d91b322a829a7f9f5c895eb4db77e0d06b7bda430b4dfef8a9925c5`.

The complete archive was verified and extracted before this preparation. Original numerical records and all 35 upstream Python/shell files were available. `source_lineage.csv` provides member-relative provenance for copied artifacts; derived figure builders and later statistical scripts are identified separately. Source identities were matched by hash where the earlier figure package used renamed or duplicated table copies.

Prepared-code changes:

1. Collocate runnable numerical modules to satisfy their relative imports.
2. Resolve the phase dataset generator's gate module relative to its own file.
3. Extract the original `pop_amp_statistics` function verbatim, so dynamic code no longer imports the absent static flux module.
4. Write original solver-test output under `outputs/` and use one FFT worker by default.
5. Let a short FRZ execution finish its numerical audit without reporting nonexistent t=1/AUC(0,1) results. Full protocol definitions and full-horizon reporting remain the same.
6. Remove low-level figure footer notes before rendering, preserve the S3 common axis definition, and apply the current paper's bottom crops.
7. Add explicit figure, dynamic, and validation entry points. Original numerical inputs are not overwritten.

Private machine roots were replaced with relative legacy-location strings in metadata, path-valued table fields, and retained legacy scripts. Metadata hashes were updated for those prepared copies while retaining upstream hashes. Numerical CSV columns and original checkpoint bytes were preserved; the preparation's numeric-column comparison is recorded in the validation report.

The project contains no manuscript draft or reviewer/audit conversation, no Git history, no author email, and no committed access token. The pattern-based security check is recorded with validation; it does not claim to detect arbitrary encoded credentials.

The repository was prepared locally. A public remote, release tag, archive DOI, final citation authors, and software reuse license are to be assigned by the research team.
