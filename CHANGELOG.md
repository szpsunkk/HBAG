# Change log

## v2.0.0 — 2026-11-06

Second public reproduction release for the revised manuscript.

- Add the current `hbag3` scientific implementation with SHA-256 identities for the unchanged modules.
- Add frozen optical/RF selections, experiment matrices, the validation gate record, the PPO checkpoint, and numerical solver regression fixtures.
- Provide an isolated, portable experiment runner, source/result verification, a small smoke run, full-scale validation regression, and RF grid replay.
- Include 4,465 per-method run summaries, convergence traces, corrected RF/delay tables, paired bootstrap results, and the original vector figures.
- Preserve unsafe/ineligible baseline outcomes and both optical-selected and RF-selected RF comparisons.
- Preserve the entire original `Battery/` implementation. No version 1 files are changed.

Internal experiment revisions included here: v2.3 (optical), v2.3.1 (paired statistics and RF protocol), and v2.4 (RF acquisition/power correction and short-slot acquisition state). These are experiment revisions, not earlier public release tags.

## Original version

The original repository snapshot is commit `44eec9f` (`final version`). Its `Battery/` directory is retained for historical comparison and is not the reproduction target for the revised manuscript.
