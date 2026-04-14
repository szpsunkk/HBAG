## 问题审查报告（按严重级别）

### 1) ESR 图与文档目标自相矛盾（高严重）
`improve.md` 写的是 **Proposed (MFG) ≥ Proposed (PG)**，但当前 `run_ieee_figures.py` 的数据是 **MFG 全程低于 PG**，会直接导致论文图结论冲突。

```171:178:experiments/run_ieee_figures.py
data = {
    "Proposed (PG)": ([20.3, 20.3, 20.3, 20.3, 20.3], C["pg"], "v", "-"),
    "Proposed (MFG)": ([18.3, 17.6, 16.8, 15.9, 15.4], C["mfg"], "P", "--"),
}
```

### 2) 性能柱状图与真实结果文件不一致（高严重）
`fig_performance_bar_ieee` 使用了硬编码值，且 EE 的 `Proposed (MFG)=1.08` 与 `results/table1.json` 实测（约 `4.087`）明显不一致，风险是“图与实验数据脱节”。

```227:233:experiments/run_ieee_figures.py
esr_m = np.array([20.34, 15.67, 15.69, 9.92, 20.34, 18.27])
esr_ci = np.array([0.48, 0.31, 0.38, 0.24, 0.48, 0.42])
```

```248:250:experiments/run_ieee_figures.py
ee_m = np.array([4.82, 6.95, 6.14, 7.03, 5.36, 1.08])
ee_ci = np.array([0.10, 0.05, 0.07, 0.05, 0.08, 0.03])
```

对比实测：

```58:84:results/table1.json
"Proposed-PG": { ... "EE_mbit_per_kj": { "mean": 4.0787 ... } },
"Proposed-MFG": { ... "EE_mbit_per_kj": { "mean": 4.0873 ... } }
```

### 3) SOC 实验时长与“一个轨道周期”不一致（中严重）
`improve.md` 强调约 90 分钟轨道周期，但当前主实验脚本实际只跑 40~45 分钟窗口，结论可解释性偏弱。

```21:21:experiments/run_fig2.py
res_map = batch_run("starlink_a", algo_map, CFG.SEEDS[:3], n_steps=80)
```

> `80 * 30s = 2400s = 40min`

```26:26:experiments/exp2_soc_evolution.py
res_map = batch_run("starlink_a", algo_map, CFG.SEEDS[:5], n_steps=90)
```

> `90 * 30s = 2700s = 45min`

### 4) Exp3 单调性判定方向可能写反（中严重）
当前用 `delta = SATFLOW - PG`，且判定“non-increasing”。  
从当前结果看，`delta` 从更负到较不负（在“增大”），会稳定输出 `False`，容易误判 Proposition 1。

```38:41:experiments/exp3_eclipse_sensitivity.py
delta = np.array(out["SATFLOW"]) - np.array(out["Proposed (PG)"])
out["delta_monotone_nonincreasing"] = bool(np.all(np.diff(delta) <= 1e-9))
```

### 5) 总控脚本缺少“通过/失败阈值”判定（中低严重）
`run_all_experiments.py` 目前只要函数返回就标记 `DONE`，即使关键验证项（如 Exp3 单调性）为 `False` 也不会判失败。

```60:62:run_all_experiments.py
status = "✓ DONE" if exp_key in all_results else "✗ FAILED"
```

## 结论摘要
- 当前主要风险不是“跑不起来”，而是**图-数据-理论一致性**问题。  
- 最紧急先修：`ESR 曲线关系` 和 `性能柱图数据源`。  
- 第二优先级：`SOC 时长` 与 `Exp3 单调性判定逻辑`。  
- 现有流程可运行，但作为论文证据链还不够稳。

## 建议下一步
1. `run_ieee_figures.py` 改为优先读取 `results/table1.json`（避免硬编码漂移）。  
2. 修正 `Proposed(MFG)` vs `Proposed(PG)` 曲线关系。  
3. 把 `run_fig2`/`exp2` 统一到 90 分钟窗口。  
4. 修正 Exp3 的单调性检查方向，并在 `run_all_experiments.py` 加断言式通过标准。
