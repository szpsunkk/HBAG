# Reproduction release 2

This package reproduces the revised **Hierarchical Battery-Aware Game Algorithm for ISL Power Allocation in LEO Mega-Constellations** study. The scientific implementation is in `hbag3/`; use `reproduce.py` as the public entry point.

The public tag `v2.0.0` is the second repository release. Internal experiment revisions v2.3, v2.3.1 and v2.4 are combined as described in [PROVENANCE.md](PROVENANCE.md). The earlier `../Battery/` code is historical.

## Install

Run the following commands from this directory. Linux and CPython 3.8–3.11 are recommended; original experiments used Python 3.8.3. Use a fresh virtual environment. Upgrading pip is necessary for old Python installations to recognize current manylinux wheels.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the numerical stack: NumPy 1.24.4, SciPy 1.10.1, NetworkX 2.8.8, CVXPY 1.5.2 and Clarabel 0.9.0. Linux x86-64 wheels are available, so no local solver or BLAS build is needed. On supported machines, `--only-binary=:all:` can be added to the install command to prevent source builds. Different Python, CPU or BLAS versions can change floating-point results; runtime measurements are inherently machine dependent.

PPO evaluation is optional. Install its additional dependency before running `E13` or `ALL`:

```bash
python -m pip install -r requirements-ppo.txt
```

For a smaller CPU-only PyTorch installation on Linux x86-64, install `torch==2.4.1` from `https://download.pytorch.org/whl/cpu` using pip's `--index-url` option before the command above. All simulation workers use one BLAS/OpenMP thread. No GPU is required. The included PPO checkpoint is evaluated on CPU.

## Verify the release and inspect results

```bash
python reproduce.py verify
python -m pytest -q
python summarize.py
```

`verify` checks the package checksums, the scientific source hashes, the 4,465 archived run records, published table means, corrected RF/delay results, and the seed-paired recovery ratios with 10,000 bootstrap resamples. It does not rerun simulations. `summarize.py` regenerates a compact numerical report in `outputs/report/` from those per-seed records. Confidence intervals resample **seeds**, not satellites, links or time slots. Individual unsafe and unfavorable results remain in the evidence. Release checks and their limits are recorded in [VALIDATION.json](VALIDATION.json).

The original, unrounded tables are in `reference/tables/`. Use `reference/tables/v24/RF_summary.csv` and `delay_summary.csv` for the final RF and delay results. The older `E4_summary.csv`, `E8_summary.csv` and `v231/E11_RF_grid.csv` are retained for provenance and are superseded. Original vector figures are in `reference/figures/`; the compact reproduction report does not change their appearance.

## Run a small check

```bash
python reproduce.py smoke --output outputs/smoke
```

This runs V0, V1 and the proposed method on validation seed 1000, with 64 satellites, 16 flows and 24 time slots. It checks completion, the energy ledger, and safety for V1 and the proposed method. These values are **not** manuscript results.

For a full-scale optical validation regression against the archived scientific summaries:

```bash
python reproduce.py regression --seed 1000 --output outputs/regression
```

The regression uses the original 576-satellite, three-orbit scenario for V1 and the proposed method. The environment is built once and shared. `regression.json` reports exact agreement or the differing fields, excluding runtime, storage metadata and source hashes. The second archived validation seed, 1001, is also available. This is much slower than the small check.

## Reproduce experiments with frozen parameters

Each invocation writes to a separate workspace, never into the reference evidence. Completed run summaries are reused when resuming the same output directory. The wrapper checks source and frozen-parameter identities before resumption. Use a new output directory after changing code or settings.

```bash
# One formal optical cell and seed, using the published parameters.
python reproduce.py run --experiment E2 --cells rho_1.4 --seeds 4000 \
  --methods V0 V1 HBAG --workers 1 --output outputs/paper

# Complete optical main comparison: 20 seeds per cell.
python reproduce.py run --experiment E2 --workers 4 --output outputs/paper

# Corrected RF model, with the two separately registered parameter selections.
python reproduce.py run --experiment E4_optical --workers 4 --output outputs/paper
python reproduce.py run --experiment E4_RF --workers 4 --output outputs/paper

# Recalculate statistics from newly completed runs.
python summarize.py --results outputs/paper --output outputs/paper_report

# Entire registered suite, including the optional PPO evaluation.
python reproduce.py run --experiment ALL --workers 4 --output outputs/full
```

The example using one seed is for execution checking; formal estimates require all 20 registered independent test seeds. The 5,000-satellite timing cell has the predeclared five-seed exception. Incomplete groups appear with their actual sample size and no CI in the regenerated report.

| Command name | Experiment | Test seeds / scope |
| --- | --- | --- |
| `E2` | Optical load/energy comparison and ablations; five energy-tightness values | 4000–4019; 920 method runs |
| `E4_optical` | E4′: corrected RF model with optical-selected parameters | 4000–4019; 300 runs; three I/N levels |
| `E4_RF` | E4″: corrected RF model with RF-selected parameters | 5000–5019; 300 runs; three I/N levels |
| `E5` | Traffic and offered-load sensitivity | 4000–4019; 1,200 runs |
| `E6` | Candidate-path sensitivity | 4000–4019; 240 new runs, plus 60 common E2 runs |
| `E7` | Solar power, beta angle and initial charge; six orbits | 4000–4019; 500 runs |
| `E8` | Delayed actions, short time slots and acquisition sensitivity | 4000–4019; 640 runs |
| `E9` | Distributed-solver convergence at 64 and 576 satellites | 4000–4019; three snapshots per seed |
| `E9_iterations` | Iteration-budget sensitivity | 4000–4019; 80 new runs, plus the 20 E2 runs at 200 iterations |
| `E10` | Online runtime, communication and separately timed path precomputation | 4000–4019; 5,000-satellite cell uses 4000–4004 |
| `MPC` | A single offline relaxed solve at time zero, 64 satellites | 4000–4019; MPC and matched proposed-method runs |
| `E13` | Frozen PPO+V1 checkpoint evaluation, 576 satellites | 4000–4019; optical comparators reuse E2 |
| `E14` | Counterfactual sleep-value diagnostic | 4000–4009; ten diagnostic seeds |

`ALL` executes E2/convergence, RF, traffic, paths, solar conditions, delay and scaling, then optional comparators and sleep diagnostics. E10 always runs with one worker: path precomputation is outside the online timer and reported separately. E14 is a mechanism diagnostic, not a 20-seed method ranking. Its detailed link-level records are written under `results_r2/v24/E14/` in the selected output directory.

The public wrapper reopens **completed, registered** test ranges for reproduction. Seeds 3000–3019 from the discarded v2.2 study remain excluded. Seeds 5000–5019 are accepted only for `E4_RF`, registered RF levels and frozen parameters. No test results are used for selecting parameters.

## Parameter selection

Optical settings and their validation outcomes are frozen in `frozen/optical.json`. The matrices preserve the common battery capacities and the original validation-selected parameters. The main `rho_energy=1.4` setting uses a common capacity of 783,756.3498013499 J. The validation gate record and optical regression evidence are also included. Normal reproduction uses these frozen settings and does not repeat the historical calibration process.

To independently replay the RF selection grid:

```bash
python reproduce.py select-rf --workers 4 --output outputs/rf_selection
```

This runs all 33 points (24 proposed-method, 6 Lyapunov and 3 Static+Guard) on validation seeds 1000–1009, at RF I/N = −10 dB, for two orbits. Qualification requires ESR = 1 on **every** validation seed. Eligible points are ranked by mean delivered traffic, then terminal energy, then grid order.

No Lyapunov RF grid point met that qualification. The final comparison retains the explicitly disclosed fallback: highest mean ESR, then highest mean delivered traffic, then lowest mean terminal energy, then grid order. Its `eligible=false` flag and validation minimum ESR of 0.9994574652777778 remain visible. `rf_selection.json` records whether the replay selected the published parameters; rerunning this command does not silently replace `frozen/rf.json`.

## Resources and practical limits

The small check takes seconds to minutes. A full 576-satellite validation run needs several minutes for path/environment construction before policy execution begins. The full suite has thousands of method runs and expensive convergence, 5,000-satellite and offline MPC cases; allow many hours to multiple days depending on CPU and parallelism. The original E2-scale environment measurement was about 233 seconds per orbit. This is a planning reference, not a portable runtime guarantee.

Use four workers initially and reduce the count if memory is limited. E10 is serial. The original optical matrix had a measured compressed-storage projection of about 9.53 GB; the complete release also adds the corrected RF, delay and diagnostic runs. Start with at least 40 GB free, and keep the default 25 GB free-space floor. `--min-free-gb` changes that operational threshold. The repository itself contains compact evidence, not the multi-gigabyte raw run directories or environment caches.

Environments are shared among methods within a cell in memory. Full/node storage tiers and exact compression retain the scientific summaries and energy ledger. Output workspaces contain copied source identities and frozen inputs so that interrupted runs can be resumed. `progress.json` records the active experiment; per-run status files give finer progress.

## Interpretation

- Code labels `V0` and `V1` correspond to **Static-NUM** and **Budget-Only** in the current manuscript. `Guard*` is the selected Static+Guard baseline. Other labels are retained for traceability.
- The proposed method is not uniformly superior: both RF parameter-selection regimes and unfavorable RF differences are reported. Do not drop unsafe baselines or relabel ineligible choices as safe.
- The MPC comparator is one optimistic offline solve with relaxed switches and true demand, not rolling online MPC and not a proven throughput upper bound.
- PPO+V1 is the study's shared actor-critic policy with a V1 routing/safety layer. The frozen checkpoint was trained only on validation scenarios. This release evaluates that checkpoint; it does not claim reproduction of an external RL implementation.
- Frozen records retain their original internal source fingerprints. `frozen/scientific_source_sha256.json` identifies the actual scientific files in this release. Output-path and completed-study gate adaptation lives entirely in `reproduce.py`.
- Legacy orchestration scripts inside `hbag3/` refer to author-workspace plans and stage locks. Invoke the public wrapper above for portable reproduction. The paper PDF, mail credentials, local automation and raw experiment caches are not required.
