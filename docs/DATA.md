# Data provenance and acquisition

The prepared repository is based on `JHTDB_LOCAL_HANDOFF_V1.tar(1).gz`, whose SHA-256 is recorded in `config/reproduction.json`. The data lineage manifest records member-relative paths, upstream hashes, and prepared-file hashes; no private machine root is required to resolve a shipped input.

The current figures use the main32, independent isotropic replication, Clark association, spectral/identity audit, and dynamic recovery records. Channel pilot records are not presented as evidence for the current paper and are not included in the figure workflow.

The main static input is the historical `isotropic1024coarse` 64³/100-frame cutout. The independent isotropic cutout in the supplied acquisition script requests x/y/z indices 385–448 and time indices 2501–2600. Full acquisition conventions and the main range are in the supplied original acquisition scripts, retained under `code/legacy/acquisition/`. The original main client imports `givernylocal`, while the independent client imports `giverny`. Those historical clients are separate from the pinned offline requirements and have not been validated online in this release.

Access credentials belong in the user's environment (`JHTDB_TOKEN`), never in a committed file. A future acquisition adapter should accept explicit destination paths, record the returned coordinate/time conventions and source hashes, and validate the recovered raw frames against the historical manifest before claiming exact static recomputation.

The three `data/checkpoints/` files are from the generated N64 periodic forced DNS, not JHTDB cutouts. Each supplies `uhat`, simulation time, and step. They remain byte-for-byte identical to the handoff. The active dynamic code uses the recorded constant-power forcing law and parameters; branch force vectors depend on state and are not assumed identical.

JHTDB states that its database data are available under the Open Data Commons Attribution License (ODC-By). Retain attribution to the database and its dataset publications: https://turbulence.pha.jhu.edu/citing.aspx . This source-data attribution is separate from the repository's future software license. This prepared release includes derived result tables and generated checkpoints, rather than the full source database.
