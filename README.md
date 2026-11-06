# Hierarchical Battery-Aware Game Algorithm for ISL Power Allocation in LEO Mega-Constellations

This repository contains the reproduction code for the paper of the same name.

**Use [version 2](v2/README.md) for the revised manuscript.** It includes the current `hbag3` implementation, frozen experiment settings, a portable runner, tests, and compact numerical evidence from the completed study.

| Directory | Purpose |
| --- | --- |
| [`v2/`](v2/) | Second public reproduction release (`v2.0.0`); current manuscript experiments |
| [`Battery/`](Battery/) | Original implementation, preserved as historical version 1 |

The public release number is separate from the internal experiment revision numbers. Version 2 contains optical results from experiment revision v2.3, paired statistics from v2.3.1, and corrected RF/acquisition results from v2.4. See [provenance](v2/PROVENANCE.md) and the [change log](CHANGELOG.md).

## Quick start

Use CPython 3.8–3.11 on Linux. The original scientific environment used Python 3.8.3; the numerical dependencies are pinned.

```bash
git clone https://github.com/szpsunkk/Hierarchical-Battery-Aware-Game-Algorithm-for-ISL-Power-Allocation-in-LEO-Mega-Constellations.git
cd Hierarchical-Battery-Aware-Game-Algorithm-for-ISL-Power-Allocation-in-LEO-Mega-Constellations/v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python reproduce.py verify
python -m pytest -q
python reproduce.py smoke
```

`verify` recalculates published statistics from bundled per-seed evidence. `smoke` runs a small validation scenario; it is **not** a paper experiment. The [reproduction guide](v2/README.md) gives full experiment commands, seed ranges, storage requirements, and scientific limitations.

中文说明：第二版代码位于 [`v2/`](v2/)，第一版 `Battery/` 原样保留。先按以上命令安装依赖、核对已有结果并运行小规模检查，再按 [第二版说明](v2/README.md) 运行正式实验。正式运行使用已冻结的参数，不根据测试结果重新选参。
