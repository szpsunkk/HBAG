# 修改后实验结果分析

## 核心问题诊断

你的最新修改已经取得**显著进展**，但仍存在**一个关键问题**需要立即解决。

---

## ✅ 已成功解决的问题

### 1. Runtime 图 — 完美 [4]
- **数据点**：5 个点（172, 348, 784, 1584, 3168）
- **线性关系**：斜率 $1.50 \times 10^{-5}$，完美验证 $\mathcal{O}(n_t)$ 复杂度
- **状态**：✓ 通过，可直接用于论文

### 2. 收敛图 — 优秀 [4]
- **改进**：$\mathcal{O}(1/k)$ 参考线与实际残差曲线吻合度高
- **MSE 指标**：0.00171（优秀）
- **平台线**：清晰显示收敛到 ~0.15 的平台
- **状态**：✓ 通过

### 3. ESR-θ 曲线 — 优秀 [4]
- **HBAG 表现**：在所有 θ 值上 ESR ≥ 96.5%（θ=0.6 时仍 93.4%）
- **Gap 曲线**：单调递增，从 0 增至 84.5%，完美验证 Proposition 1
- **状态**：✓ 通过

### 4. SOC 演化 — 清晰展示 [4]
- **HBAG 最终 SOC**：~27%（vs SATFLOW=0%）
- **标注完整**：Eclipse 起点、$C_{min}$ 阈值、维持说明
- **状态**：✓ 通过

---

## 🔴 **致命问题：HBAG FVR 仍然过高**

### 问题描述

即使参数自动搜索后（`lambda=0.200, margin=0.200`），**HBAG 的 FVR 仍为 7.62%**，**仍然高于 SMFG-adapted（0.15%）和 MAAC-IILP（2.29%）**：

| 方法 | ESR | FVR | EE |
|------|-----|-----|-----|
| SMFG-adapted | 100% | **0.15%** | 4.18 |
| MAAC-IILP | 100% | **2.29%** | 3.69 |
| **HBAG** | 100% | **7.62%** ⚠️ | 3.96 |

**论文声称** [1]：
> "HBAG keeps FVR low across reported settings while preserving battery sustainability."

这**直接矛盾**。

### 根本原因分析

从 exp7（消融实验）[4] 可以看出：

| 版本 | ESR | FVR | EE |
|------|-----|-----|-----|
| V0-SATFLOW | 100% | 0% | 2.41 |
| V1-dynamic-only | 100% | 0.97% | 2.87 |
| V2-penalty-only | 100% | 1.67% | 2.91 |
| **V3-HBAG-full** | 100% | **1.71%** | 3.43 |

**在消融实验中，HBAG 的 FVR 仅 1.71%**，但在完整 exp1 中跳到 7.62%。这说明：

**问题不在参数调整，而在 exp1 的场景设置与 exp7 的差异**。

---

## 🔍 **问题根源：场景差异分析**

对比两个实验的设置：

| 指标 | Exp1 | Exp7 |
|------|------|------|
| 卫星数 | 172 | 172 |
| Eclipse 比例 θ | 0.38 | 0.38 |
| 需求/容量比 | 0.65 | ? |
| **关键差异** | **完整网络流** | **可能简化场景** |

**假设**：exp7 可能使用了更宽松的流量需求或更高的初始 SOC，导致 FVR 自然较低。

---

## 🔧 **必须修复的代码改动**

### 最小改动方案：统一 exp1 和 exp7 的场景

在 `experiments/exp7_ablation.py` 中，确保与 exp1 使用**完全相同的**：

```python
# exp7_ablation.py - 修改前
# 可能存在的问题：
# 1. 初始 SOC 设置不同
# 2. 流量需求不同
# 3. Eclipse 时间不同

# 修改后：从 exp1 复用场景参数
from experiments.run_table1 import (
    CONSTELLATION_CONFIG,
    TRAFFIC_CONFIG,
    ENERGY_CONFIG
)

# 确保 exp7 使用完全相同的参数
```

### 代码修改（最小化）

```python
# experiments/exp7_ablation.py

def run_ablation_study():
    """
    关键修改：使用与 exp1 完全相同的场景参数
    """
    # 从 exp1 复用配置
    config = {
        'n_sats': 172,
        'theta': 0.38,  # 与 exp1 一致
        'demand_capacity_ratio': 0.65,  # 与 exp1 一致
        'cb_init_ratio': 0.8,  # 与 exp1 一致
        'n_runs': 10,
        'seed_base': 42
    }
    
    # 运行消融实验
    results = {}
    for version in ['V0-SATFLOW', 'V1-dynamic-only', 'V2-penalty-only', 'V3-HBAG-full']:
        results[version] = run_single_version(version, config)
    
    return results
```

### 验证步骤

执行以下命令确认修复：

```bash
# 1. 重新运行 exp7，确保与 exp1 场景一致
python experiments/exp7_ablation.py --verify-config

# 2. 对比 exp1 和 exp7 的 HBAG FVR
python -c "
import json
with open('results/all_experiment_results.json') as f:
    data = json.load(f)
    exp1_fvr = data['exp1']['Proposed-HBAG']['FVR']['mean']
    exp7_fvr = data['exp7']['V3-HBAG-full']['FVR']
    print(f'Exp1 HBAG FVR: {exp1_fvr:.2f}%')
    print(f'Exp7 HBAG FVR: {exp7_fvr:.2f}%')
    if abs(exp1_fvr - exp7_fvr) > 2:
        print('⚠️ 场景不一致！需要修复')
    else:
        print('✓ 场景一致')
"

# 3. 全量重跑
python run_all_experiments.py
```

---

## 📊 **修复后的预期结果**

修复后，exp7 的 HBAG FVR 应该**接近 exp1 的 7.62%**（而不是当前的 1.71%），这样才能：

1. **验证消融的有效性**：V0→V3 的 FVR 增长反映真实的权衡
2. **解释 FVR 高的原因**：不是参数问题，而是场景本身的约束
3. **为论文提供合理解释**：在高流量需求下，HBAG 必须在 ESR 和 FVR 之间权衡

---

## 📋 **修改优先级**

| 优先级 | 任务 | 代码行数 |
|--------|------|----------|
| 🔴 P0 | 统一 exp1 和 exp7 的场景参数 | ~10 行 |
| 🟢 P1 | 验证修复后的一致性 | ~5 行 |

**建议立即执行**：修改 `exp7_ablation.py` 以使用与 exp1 完全相同的场景参数，然后重跑全量实验。