# Provenance and contents

The scientific modules in `hbag3/` were copied from the final revised-manuscript implementation. The SHA-256 map in `frozen/scientific_source_sha256.json` covers all 56 non-test Python modules. `reproduce.py verify` checks these identities. No controller, physical model, solver or baseline behavior was changed to package this release.

The public wrapper redirects output paths into an isolated workspace, uses the already-frozen parameters, replaces the author-workspace stage locks with scoped completed-study seed checks, and makes the disk threshold configurable. Original full/node storage policies are retained; corrected RF and delay runs use the final v2.4 node tier. The tests include the original scientific regression fixtures. One test specific to the author's email/cache-deletion automation was omitted.

## Evidence versions

| Evidence | Source revision | Included form |
| --- | --- | --- |
| Optical E2/E5/E6/E7, convergence, scaling, MPC/PPO | v2.3 | Per-seed summaries, original tables, convergence traces |
| Paired recovery ratio and orbit differences | v2.3.1 | Per-seed and aggregate CSVs |
| RF power/acquisition correction; RF selection; delayed actions | v2.4 | Per-seed summaries and corrected `tables/v24/` CSVs |
| Lyapunov RF fallback | Explicit final follow-up to v2.4 | Frozen fallback selection and all 60 fallback test runs |
| Sleep-value diagnostic | v2.4 | Link/actual/joint CSV evidence; diagnostic implementation |
| Optical preservation check | v2.4 versus v2.3 | Archived full-scale validation summaries and original check record |

`reference/per_seed.jsonl.gz` contains 4,465 records. Each record has the experiment, cell, method label, relative original source path, SHA-256 of the original summary file, and the numerical summary. Only the original host-specific `manifest` object is omitted. The source hash identifies the original file before this omission; it is not a hash of the transformed JSONL row. Counts include common E2 cells deliberately reused for E6, the 200-iteration comparison and E13; they are not 4,465 independently generated environments.

`reference/catalogue.json` records counts and the superseded tables. Reports under `reference/reports/` are historical stage reports. Their original author-workspace links may not resolve in this compact release. The current `reference/tables/v24/RF_summary.csv` includes the final disclosed fallback and supersedes the earlier report text that listed its RF-selected results as unavailable.

The final optical study uses seeds 4000–4019. RF-specific selection uses validation seeds 1000–1009; its independent tests use 5000–5019. The discarded v2.2 formal results on seeds 3000–3019 are not included as current results. Solver failure fixtures are numerical regression inputs, not selected method-performance evidence.

## Data and third-party software

`base_rev3/sources/natural_earth_cities.geojson` is the same bundled populated-places input used in the experiments, derived from [Natural Earth](https://www.naturalearthdata.com/). Natural Earth data are public domain; see its [terms of use](https://www.naturalearthdata.com/about/terms-of-use/).

Python libraries and solvers remain subject to their respective upstream licenses. This update does not change the repository's existing code-licensing terms or grant a new license.

The model checkpoint is the authors' small CPU PPO actor-critic, trained in the validation environment described by `frozen/ppo/training.json`. The release includes the checkpoint for deterministic evaluation, and the original training code in `hbag3/optional_r2.py` for inspection.

No manuscript source/PDF, multi-gigabyte raw result tree, local environment cache, notification credential or private machine configuration is needed by the release.
