已根据你最新 `@interact/improve.md` 完成修改并重跑，结果已重新写入 `@interact/diagnostics.json`。

这轮完成的核心改动：

- `exp7` 场景与 `exp1` 完全对齐（P0）
  - 文件：`experiments/exp7_ablation.py`
  - 改为使用与 `exp1` 一致的关键参数：
    - `constellation = starlink_a`
    - `traffic_level = high`
    - `eclipse_fraction = 0.38`
    - `n_steps = 360`
    - `seeds = CFG.SEEDS`（10个）
  - 新增 `--verify-config` 参数用于快速验证场景一致性。

已验证：

- 运行 `python experiments/exp7_ablation.py --verify-config`
- 输出确认：
  - `{'constellation': 'starlink_a', 'traffic_level': 'high', 'eclipse_fraction': 0.38, 'n_steps': 360, 'n_seeds': 10}`

已全量重跑并写回：

- `python run_all_experiments.py`
- 成功输出：
  - `Saved: results/all_experiment_results.json`
  - `Saved: interact/diagnostics.json`

当前关键结果（本轮）：
- `Exp1 HBAG`: `ESR 100%`, `FVR 7.62%`, `EE 3.9569`
- 全部实验状态 `[✓ DONE]`。