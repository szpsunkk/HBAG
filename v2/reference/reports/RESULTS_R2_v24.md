# HBAG R2 v2.4 results

Offline execution of BRIEF_v24.md. Existing manuscript and archived evidence are unchanged. All new scientific outputs are under results_r2/v24 and tables_r2/v24.

## Model changes and regression

RF links use slot-average incremental power kappa q(R/kappa) in the distributed and central objectives, power constraints, recovery, marginal costs, and energy ledger. Optical arithmetic is unchanged. Each on ISL retains its remaining acquisition time across slots, resets on a new wake, and consumes acquisition idle energy in every affected slot. Config flags rf_capture_power and acquisition_carry default to true and can independently restore the prior behavior.

E2 optical regression: PASS. Exact float64 JSON values for all scientific summary and per-slot ledger fields. Only source/config/manifest/storage metadata and wall-clock timings excluded; no numerical tolerance.

- Seed 1000, HBAG: bit-identical scientific summaries and ledgers, 288 slots, dt = 59.6888238472355 s.
- Seed 1000, V1: bit-identical scientific summaries and ledgers, 288 slots, dt = 59.6888238472355 s.
- Seed 1001, HBAG: bit-identical scientific summaries and ledgers, 288 slots, dt = 59.6888238472355 s.
- Seed 1001, V1: bit-identical scientific summaries and ledgers, 288 slots, dt = 59.6888238472355 s.

## RF re-selection

The exact 33-point v2.3.1 grid uses validation seeds 1000–1009, two orbits, I/N = −10 dB, and the unchanged optical battery. Eligibility requires ESR = 1 for every seed; ties use maximal mean delivered traffic, minimal mean terminal energy, then original grid order. Fresh seeds 5000–5019 are locked until E11_RF/frozen.json exists.

- HBAG: `{"dual_eta": 0.03, "dual_warm": "zero", "kappa_off": 2.0}`.
- Static+Guard: `{"guard_soc": 0.9}`.

No eligible RF point for: Lyapunov. The author-approved fallback selects these methods by highest mean ESR, then highest mean delivered traffic, lowest mean terminal energy, and original grid order. Their completed E4″ rows are marked † below.

## RF fallback follow-up

† selected by the fallback rule (no grid point with ESR = 100% on every validation seed). Selection uses only the corrected E11-RF′ validation seeds 1000–1009. The fallback freeze precedes all fresh-seed runs of these methods.

- Lyapunov: `Lyapunov_38`; parameters `{"kappa_off": 1.0, "lyapunov_lambda": 2.0, "lyapunov_theta": 0.6}`; minimum validation ESR = 0.999457465278 (99.945746528%); mean validation ESR = 0.999630172164 (99.963017216%); eligible = false.

These E4″ cells use seeds 5000–5019, I/N = −300, −10 and 0 dB, three orbits, and the main pipeline’s exact environment and node storage tier. The planned Holm family remains the same six HBAG-versus-V1 comparisons. Selection provenance and combined completion are recorded in results_r2/v24/fallback/frozen.json and DONE.json.

## RF table rows

Delivered is Tbit, with a 95% percentile bootstrap CI (10,000 seed resamples, bootstrap seed 1729). ESR is the mean of the 20 seed values. E4′ uses optical-selected parameters on seeds 4000–4019; E4″ uses RF-selected parameters on seeds 5000–5019. Each run spans three orbits.

| Experiment | I/N dB | Method | Delivered Tbit [95% CI] | ESR |
|---|---:|---|---|---:|
| E4_prime | -300.0 | V0 | 839.226 [814.6637, 862.7201] | 0.8144654827 |
| E4_prime | -300.0 | Static+Guard | 562.2908 [541.3776, 583.6503] | 1 |
| E4_prime | -300.0 | Lyapunov | 595.6209 [574.9697, 616.7043] | 0.9987639251 |
| E4_prime | -300.0 | V1 | 657.1345 [635.3133, 679.1202] | 1 |
| E4_prime | -300.0 | HBAG | 638.0126 [616.4293, 659.9007] | 1 |
| E4_prime | -10.0 | V0 | 801.1519 [778.3808, 822.9792] | 0.8132357374 |
| E4_prime | -10.0 | Static+Guard | 537.8915 [518.2265, 557.9881] | 1 |
| E4_prime | -10.0 | Lyapunov | 562.6038 [543.2886, 582.3314] | 0.999100598 |
| E4_prime | -10.0 | V1 | 627.4566 [607.1083, 648.0213] | 1 |
| E4_prime | -10.0 | HBAG | 607.564 [587.5797, 627.787] | 1 |
| E4_prime | 0.0 | V0 | 573.0851 [559.6285, 585.8765] | 0.8065640673 |
| E4_prime | 0.0 | Static+Guard | 390.3615 [377.958, 403.0877] | 1 |
| E4_prime | 0.0 | Lyapunov | 380.9116 [368.93, 393.2971] | 0.9999150029 |
| E4_prime | 0.0 | V1 | 450.2823 [437.59, 463.3637] | 1 |
| E4_prime | 0.0 | HBAG | 427.7835 [415.32, 440.6257] | 1 |
| E4_double_prime | -300.0 | V0 | 816.9447 [793.9306, 838.8993] | 0.8156376591 |
| E4_double_prime | -300.0 | Static+Guard | 545.8141 [528.5512, 562.0168] | 1 |
| E4_double_prime | -300.0 | Lyapunov † | 579.0278 [561.9698, 594.9152] | 0.9987479504 |
| E4_double_prime | -300.0 | V1 | 639.8864 [620.8507, 657.9448] | 1 |
| E4_double_prime | -300.0 | HBAG | 653.1561 [634.5811, 670.9009] | 1 |
| E4_double_prime | -10.0 | V0 | 780.3009 [758.8821, 800.5933] | 0.814343111 |
| E4_double_prime | -10.0 | Static+Guard | 522.2749 [506.1169, 537.3972] | 1 |
| E4_double_prime | -10.0 | Lyapunov † | 547.3341 [531.2716, 562.2495] | 0.9991289304 |
| E4_double_prime | -10.0 | V1 | 611.0791 [593.2535, 627.9428] | 1 |
| E4_double_prime | -10.0 | HBAG | 623.3192 [605.9197, 639.7732] | 1 |
| E4_double_prime | 0.0 | V0 | 560.3385 [547.915, 571.9426] | 0.8071846668 |
| E4_double_prime | 0.0 | Static+Guard | 380.3023 [370.5852, 389.4282] | 1 |
| E4_double_prime | 0.0 | Lyapunov † | 371.0193 [361.3734, 380.1138] | 0.9998954114 |
| E4_double_prime | 0.0 | V1 | 439.7897 [428.9784, 449.9152] | 1 |
| E4_double_prime | 0.0 | HBAG | 446.7295 [436.4339, 456.3627] | 1 |

Paired difference is the mean of 100(HBAG − V1)/V1 across matched seeds. The CI resamples matched seed pairs. Wilcoxon tests use delivered-bit differences, two-sided, with Holm correction across the six planned RF comparisons. Ratio-of-means sensitivity estimates are also retained in RF_paired.csv.

| Experiment | I/N dB | Paired difference % [95% CI] | Holm p |
|---|---:|---|---:|
| E4_prime | -300 | -2.916 [-3.264, -2.552] | 1.144e-05 |
| E4_prime | -10 | -3.172 [-3.502, -2.827] | 1.144e-05 |
| E4_prime | 0 | -5.008 [-5.39, -4.609] | 1.144e-05 |
| E4_double_prime | -300 | 2.094 [1.895, 2.301] | 1.144e-05 |
| E4_double_prime | -10 | 2.022 [1.813, 2.234] | 1.144e-05 |
| E4_double_prime | 0 | 1.598 [1.379, 1.823] | 1.144e-05 |

## Delay table (dt = 15 s)

Change is relative to zero delay of the same method and seed. The measured energy change is the mean across seeds of each run’s maximum measured absolute energy change during the delay; the maximum across seeds is also shown. The 64-satellite shell, 32 flows, 1 s integration, and published seeds are unchanged.

| Delay s | Method | Delivered change % [95% CI] | ESR | Measured change, mean / max J |
|---:|---|---|---:|---:|
| 0 | HBAG | 0 [0, 0] | 1 | 0 / 0 |
| 0 | HBAG+margin | 0 [0, 0] | 1 | 0 / 0 |
| 0 | V0 | 0 [0, 0] | 0.8929176186 | 0 / 0 |
| 1 | HBAG | -1.459 [-1.553, -1.351] | 0.999999864 | 415.15 / 415.15 |
| 1 | HBAG+margin | -0.08998 [-0.1609, -0.02898] | 1 | 415.15 / 415.15 |
| 1 | V0 | -1.46 [-1.489, -1.431] | 0.8930667972 | 327.75 / 327.75 |
| 15 | HBAG | -22.64 [-23.09, -22.2] | 0.9999468288 | 6227.25 / 6227.25 |
| 15 | HBAG+margin | -0.3088 [-0.5055, -0.1158] | 1 | 6096.15 / 6227.25 |
| 15 | V0 | -21.9 [-22.34, -21.47] | 0.8951858953 | 4916.25 / 4916.25 |
| 5 | HBAG | -7.221 [-7.4, -7.042] | 0.9999944245 | 2075.75 / 2075.75 |
| 5 | HBAG+margin | -0.1155 [-0.2396, 0.009351] | 1 | 2075.75 / 2075.75 |
| 5 | V0 | -7.301 [-7.446, -7.157] | 0.8936696312 | 1638.75 / 1638.75 |

Published dt = 60 s regression: 400/400 completed comparisons are bit-identical in all scientific summary fields. This includes the acquisition-duration sensitivity cells. No published dt = 5 s cell exists in the frozen matrix or current E8 results; no new 5 s cell is substituted.
The corrected 5 s acquisition sequence is covered by the physical regression tests.

## Sleep-value diagnostic (E14)

Ten seeds (4000–4009), one orbit, 60 s slots, 64 satellites, 32 OD pairs, optical-selected HBAG parameters at rho_E = 1.4. Every on ISL with positive executed flow is sampled whenever any endpoint in the shell is eclipsed; the subset with an eclipsed endpoint on that ISL is also saved. Controller values use the actual executed allocation and recent-peak scaling. Both exact optima use the same prices, node budgets, candidate catalogue and peak-scaled demand. Removal zeros both directed capacities; it does not release fixed-power budgets or add new routes. Phi includes utility, incremental power price, and quadratic regularization. Utility-only losses are saved separately.

Clarabel uses tolerance 1e-9 and must meet its original-problem primal/dual certificate. Ratios exclude losses no larger than the summed certified numerical error bounds; counts and the literal under-half share over all samples are saved. Rank correlations are descriptive pooled correlations; dependent links/slots are not treated as independent replicates.

| Value | Samples | Spearman | Kendall | Ratio median [5%, 95%] | Share below half true loss |
|---|---:|---:|---:|---|---:|
| predicted | 70432 | 0.7112 | 0.529 | 0.04315 [0, 6.547] | 0.5839 |
| greedy | 70432 | 0.7679 | 0.5806 | 0.4215 [0, 7.534] | 0.5105 |

A2 uses the preceding slot’s observed value. Actual sleeps are matched by physical edge to that value-generating slot and its unchanged reference problem (value_slot and decision_slot are saved). Thus predicted and true losses refer to the same prices and budgets. Zero-flow actual sleeps are included in this separate cohort.

- predicted: 3056 actual A2 sleeps; mean predicted loss 0.004011167; mean true loss 0.05199258.
- greedy: 3056 actual A2 sleeps; mean predicted loss 0.02495455; mean true loss 0.05199258.

For 816 slots with at least two A2 sleeps, mean sum of individual losses = 0.1921957; mean joint-removal loss = 0.1943198; mean joint minus individual sum = 0.002124071.
The greedy diagnostic reserves alternative directed residual capacity across displaced flows, in ascending flow/path order, independently for each hypothetical removed edge. It is never passed to the controller. Full ratio distributions, utility losses, per-seed correlations, actual-sleep statistics and joint-removal records for both predictors are in tables_r2/v24.

## Fallback-path timing

scaling_cell starts tick before decide; usable_catalogue runs before elapsed; mean_slot_s averages algorithm_wall_s. control.usable_catalogue contains lazy active graph construction and nx.shortest_path.
usable_catalogue_s includes masking/incidence assembly as well as fallback; it is not a fallback-only estimate. No additional measurement is required by step 7.

| Satellites | Snapshots | Online mean s | Usable catalogue mean s (included) |
|---:|---:|---:|---:|
| 1024 | 80 | 0.2596131 | 0.02442689 |
| 1584 | 80 | 0.4925357 | 0.03913384 |
| 256 | 80 | 0.07255095 | 0.004117202 |
| 5000 | 10 | 2.439758 | 0.156965 |
| 576 | 80 | 0.1588081 | 0.01164042 |

## Reproducibility

See results_r2/v24/REPORT.md, code.patch, launch.json, progress.json and DONE.json. DONE.json is written only after report/table generation. A failed pipeline writes FAILED.json and a partial report. The source snapshot under pristine/hbag3 is the unmodified baseline. No network or external notification is used.
