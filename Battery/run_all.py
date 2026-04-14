"""Run all experiments and generate all paper artifacts."""

import os
import sys
import time


def main():
    print("=" * 70)
    print("Battery-Aware ISL Power Allocation 论文实验")
    print("生成所有表格和图")
    print("=" * 70)

    print("\n[1/6] 生成 Table I (性能对比)")
    print("-" * 40)
    start = time.time()
    from experiments.run_table1 import run_table1

    run_table1()
    print("Table I 完成，耗时 %.1fs" % (time.time() - start))

    print("\n[2/6] 生成 Fig.2 (Battery SOC Evolution)")
    print("-" * 40)
    start = time.time()
    from experiments.run_fig2 import run_fig2

    run_fig2()
    print("Fig.2 完成，耗时 %.1fs" % (time.time() - start))

    print("\n[3/6] 生成 Fig.3 (ESR vs Eclipse Fraction)")
    print("-" * 40)
    start = time.time()
    from experiments.run_fig3 import run_fig3

    run_fig3()
    print("Fig.3 完成，耗时 %.1fs" % (time.time() - start))

    print("\n[4/6] 生成 Fig.4 (收敛曲线)")
    print("-" * 40)
    start = time.time()
    from experiments.run_fig4 import run_fig4

    run_fig4()
    print("Fig.4 完成，耗时 %.1fs" % (time.time() - start))

    print("\n[5/6] 生成 Fig.5 (运行时间)")
    print("-" * 40)
    start = time.time()
    from experiments.run_fig5 import run_fig5

    run_fig5()
    print("Fig.5 完成，耗时 %.1fs" % (time.time() - start))

    print("\n[6/6] 生成 IEEE改进图组 (基于 improve.md)")
    print("-" * 40)
    start = time.time()
    from experiments.run_ieee_figures import run_ieee_figures

    run_ieee_figures()
    print("IEEE图组完成，耗时 %.1fs" % (time.time() - start))

    print("\n" + "=" * 70)
    print("所有实验完成！结果保存在 results/ 目录：")
    print("  results/table1.json  - Table I数据")
    print("  results/figures/fig2_soc_evolution.pdf")
    print("  results/figures/fig3_esr_vs_theta.pdf")
    print("  results/figures/fig4_convergence.pdf")
    print("  results/figures/fig5_runtime.pdf")
    print("  results/figures/ieee/*.pdf/.png")
    print("=" * 70)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
