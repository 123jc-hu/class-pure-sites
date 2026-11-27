"""T20-A 的重复间汇总、预定判据与曲线。"""
import numpy as np
import pandas as pd

from partial_confounding import DELTAS, METHODS, OUT, PURE_METHODS

KEYS = ["repeat", "delta", "method"]
AUC = ["test_auc", "pooled_auc", "reverse_auc"]


def paired_differences(metrics):
    paired = metrics[KEYS + AUC].copy()
    balanced = paired.loc[paired.delta == 0].drop(columns="delta")
    m0 = paired.loc[paired.method == "M0"].drop(columns="method")
    paired = paired.merge(balanced, on=["repeat", "method"], suffixes=("", "_0"), validate="many_to_one")
    paired = paired.merge(m0, on=["repeat", "delta"], suffixes=("", "_m0"), validate="many_to_one")
    for metric in AUC:
        paired[f"{metric}_vs_delta0"] = paired[metric] - paired[f"{metric}_0"]
        paired[f"{metric}_vs_m0"] = paired[metric] - paired[f"{metric}_m0"]
    m0_at_zero = balanced.loc[balanced.method == "M0"].set_index("repeat")["test_auc"]
    paired["test_auc_interaction"] = (
        paired["test_auc_vs_m0"] - (paired["test_auc_0"] - paired["repeat"].map(m0_at_zero))
    )
    paired["reverse_auc_drop"] = -paired["reverse_auc_vs_delta0"]
    return paired[KEYS + [f"{metric}_{contrast}" for metric in AUC for contrast in ("vs_delta0", "vs_m0")]
                  + ["test_auc_interaction", "reverse_auc_drop"]]


def monotone_with_one_rise(values):
    rises = np.diff(values)
    return bool((rises > 0).sum() <= 1 and np.all(rises <= 0.01))


def interval(values):
    mean = values.mean()
    lo, hi = values.quantile([0.025, 0.975])
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def markdown(headers, rows):
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def pass_text(condition):
    return "满足" if condition else "未满足"


def render_report(metrics, paired):
    indexed = metrics.set_index(["method", "delta", "repeat"]).sort_index()
    diffs = paired.set_index(["method", "delta", "repeat"]).sort_index()
    means = metrics.groupby(["method", "delta"])[AUC].mean()
    m0 = means.loc["M0"]
    drop = diffs.loc["M0", "reverse_auc_drop"]
    drop03 = drop.loc[0.3]
    g1_monotone = monotone_with_one_rise(m0.reverse_auc)
    g1 = g1_monotone and drop03.mean() >= .05 and (drop03 > 0).mean() >= .8
    g2 = drop.loc[0.2].mean() >= .03
    l3 = means.loc["L3"]
    l3_vs_m0 = (l3.test_auc - m0.test_auc).loc[:0.3]
    l3_drop04 = diffs.loc[("L3", 0.4), "reverse_auc_drop"]
    m0_drop04 = drop.loc[0.4]
    g3 = l3_drop04.mean() <= .5 * m0_drop04.mean() and (l3_vs_m0 >= -.01).all()
    sections = [
        "# T20-A 第一阶段",
        "每个统计单元包含 50 次重复。方括号给出重复间第 2.5 和 97.5 百分位数，描述重复间波动，"
        "并非置信区间。配对比例以同一重复内的差值计算。",
        "## G1–G3",
        markdown(["判据", "结果", "统计量"], [
            ["G1 场景存在", pass_text(g1),
             f"反向集曲线满足单调规则：{pass_text(g1_monotone)}。"
             f"δ = 0.3 配对下降 {interval(drop03)}，下降比例 {(drop03 > 0).mean():.1%}"],
            ["G2 真实程度下已明显", pass_text(g2), f"δ = 0.2 配对下降 {interval(drop.loc[0.2])}"],
            ["G3 站点内对比有余地", pass_text(g3),
             f"δ = 0.4 配对下降 L3 {interval(l3_drop04)}，M0 {interval(m0_drop04)}。"
             f"δ ≤ 0.3 的最小平均测试差 {l3_vs_m0.min():+.4f}"],
        ]),
        "下表为测试 AUC 交互差，即同一重复中相对 M0 的差值再减去 δ = 0 的方法差。",
        markdown(["δ", "L1", "L2", "L3", "E-blind"], [
            [delta, *(interval(diffs.loc[(method, delta), "test_auc_interaction"])
                      if delta < .5 or method in PURE_METHODS else "—"
                      for method in ("L1", "L2", "L3", "E-blind"))]
            for delta in DELTAS
        ]),
    ]

    g4_rows = []
    for method in ("RW", "L1", "CB"):
        for delta in DELTAS[:-1]:
            difference = indexed.loc[(method, delta), ["test_auc", "reverse_auc"]] - indexed.loc[
                ("L3", delta), ["test_auc", "reverse_auc"]
            ]
            g4_rows.append([method, delta, interval(difference.test_auc), f"{(difference.test_auc > 0).mean():.1%}",
                            interval(difference.reverse_auc), f"{(difference.reverse_auc > 0).mean():.1%}"])
    simple_enough = [
        method for method in ("RW", "L1")
        if ((means.loc[method, ["test_auc", "reverse_auc"]] -
             l3[["test_auc", "reverse_auc"]]) >= -.01).all().all()
    ]
    l2_rows = []
    for delta in DELTAS:
        l2 = indexed.loc[("L2", delta)]
        vs_l1 = interval(l2.test_auc - indexed.loc[("L1", delta), "test_auc"]) if delta < .5 else "—"
        l2_rows.append([delta, interval(l2.test_auc), vs_l1, interval(l2.reverse_auc)])
    sections.extend([
        "## G4 简单办法",
        markdown(["方法", "δ", "测试 AUC − L3", "测试差值为正", "反向集 AUC − L3", "反向集差值为正"], g4_rows),
        (f"简单重新加权或站点截距已足以处理部分混杂（{', '.join(simple_enough)}）。"
         if simple_enough else "RW 和 L1 均未在所有 δ 上同时达到 L3 的两项 −0.01 界限。"),
        "### L2 单列",
        "L2 用训练站点均值中心化反向集。该均值由另一类占多数，会抬高反向集 AUC。"
        "因此这列不用于 G4、G6 判读。",
        markdown(["δ", "测试 AUC", "测试 AUC − L1", "反向集 AUC（受中心化影响）"], l2_rows),
    ])

    g5_rows = []
    for delta in DELTAS:
        rows = indexed.loc[("M0", delta)]
        for estimator, label in (("subject_cv_auc", "按被试 5 折"), ("site_cv_auc", "留一训练站点")):
            error = rows[estimator] - rows.test_auc
            g5_rows.append([delta, label, interval(error) if error.notna().any() else "—",
                            interval(error.abs()) if error.notna().any() else "—"])
    sections.extend([
        "## G5 评估偏差",
        "按被试 5 折估计按诊断分层，每折先在该折训练部分内选参，再对五个留出折的 AUC 等权平均。",
        markdown(["δ", "训练内估计", "估计 − 测试 AUC", "绝对误差"], g5_rows),
        "δ = 0.5 时训练站点是纯类站点，留一训练站点 AUC 无定义。",
    ])

    g6_rows, above_balanced, verdicts, sufficient = [], [], {}, []
    for method in ("E-blind", "E-cond"):
        rows = diffs.loc[method]
        loss = rows["test_auc_interaction"]
        erasure_drop = rows["reverse_auc_drop"]
        loss04 = loss.loc[0.4]
        drop04 = erasure_drop.loc[0.4]
        monotone = monotone_with_one_rise(loss.groupby("delta").mean().loc[0.1:0.4])
        lose_signal = loss04.mean() <= -.02 and (loss04 < 0).mean() >= .8 and monotone
        drop_ratio_met = (drop04 >= .5 * m0_drop04).mean()
        retain_shortcut = drop04.mean() >= .5 * m0_drop04.mean() and drop_ratio_met >= .8
        verdicts[method] = (lose_signal, retain_shortcut)
        for delta in DELTAS:
            d = erasure_drop.loc[delta]
            g6_rows.append([method, delta, interval(loss.loc[delta]), f"{(loss.loc[delta] < 0).mean():.1%}",
                            interval(d), f"{(d >= .5 * drop.loc[delta]).mean():.1%}"])
            if delta > 0 and (d < 0).any():
                above_balanced.append([method, delta, interval(-d), f"{(d < 0).mean():.1%}"])
        own = means.loc[method].loc[:.4]
        if (own.reverse_auc >= l3.reverse_auc - .01).all() and (own.test_auc >= m0.test_auc.loc[:.4] - .01).all():
            sufficient.append(method)

    sections.extend([
        "## G6 擦除的两种失败",
        markdown(["方法", "δ", "测试 AUC 交互差", "交互差为负", "反向集 AUC 配对下降", "下降 ≥ ½ M0"],
                 g6_rows),
        markdown(["方法", "丢诊断信号", "去不掉捷径"],
                 [[method, pass_text(loss), pass_text(shortcut)] for method, (loss, shortcut) in verdicts.items()]),
    ])
    if verdicts["E-blind"][0] and verdicts["E-cond"][1]:
        sections.append("擦除的两难成立。L3 与 E-blind 的逐 δ 交互差见 G3。")
        l3_loss = diffs.loc["L3", "test_auc_interaction"]
        l3_loss04 = l3_loss.loc[.4]
        blind_loss04 = diffs.loc[("E-blind", .4), "test_auc_interaction"]
        l3_same_criterion = (l3_loss04.mean() <= -.02 and (l3_loss04 < 0).mean() >= .8
                             and monotone_with_one_rise(l3_loss.groupby("delta").mean().loc[0.1:0.4]))
        sections.append(f"δ = 0.4 时，L3 的交互差为 {interval(l3_loss04)}，E-blind 为 "
                        f"{interval(blind_loss04)}，两者的配对差为 {interval(l3_loss04 - blind_loss04)}。"
                        f"L3 对同一测试 AUC 代价判据的结果为{pass_text(l3_same_criterion)}。")
    if not verdicts["E-blind"][0]:
        sections.append("擦除的代价测不出，叙事退回‘去不掉捷径’。")
    if sufficient:
        sections.append(f"线性层面擦除已够用（{', '.join(sufficient)}）。")
    if above_balanced:
        sections.extend([
            "### 反向集 AUC 超过自身 δ = 0 参照的情况",
            "有重复超过参照的单元如下。升高只作描述。",
            markdown(["方法", "δ", "相对自身参照的配对变化", "高于参照比例"], above_balanced),
        ])
    sections.extend([
        "## AUC 汇总",
        markdown(["δ", "方法", "测试站点平均 AUC", "合并测试 AUC", "反向集 AUC"], [
            [delta, method, interval(group.test_auc), interval(group.pooled_auc),
             "见 G4 单列" if method == "L2" else interval(group.reverse_auc)]
            for (delta, method), group in metrics.groupby(["delta", "method"], sort=False)
        ]),
        "![反向集 AUC 曲线](reverse_auc.png)",
        "![测试站点平均 AUC 曲线](test_auc.png)",
    ])
    return "\n\n".join(sections) + "\n"


def plot_auc(metrics, metric, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P", "X", "h")
    for method, color, marker in zip(METHODS, colors, markers):
        if metric == "reverse_auc" and method == "L2":
            continue
        values = metrics.loc[metrics.method == method].groupby("delta")[metric].mean()
        ax.plot((2 * values.index.to_numpy())**2, values, label=method, color=color, marker=marker)
    for x, label in ((.02, "ABIDE-I"), (.25, "ABIDE-II / ADHD-200")):
        ax.axvline(x, color=".55", linewidth=1, linestyle=":")
        ax.text(x, 1.01, label, transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=8)
    ax.set(xlabel=r"$L/L_{\mathrm{pure}}$", ylabel="Reverse-set AUC" if metric == "reverse_auc" else "Mean within-site test AUC")
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / f"{metric}.png", dpi=180)
    plt.close(fig)


def main():
    metrics = pd.read_csv(OUT / "repeat_metrics.csv")
    expected = {(repeat, delta, method)
                for repeat in range(50) for delta in DELTAS
                for method in (METHODS if delta < .5 else PURE_METHODS)}
    assert len(metrics) == len(expected) and set(metrics[KEYS].itertuples(index=False, name=None)) == expected
    paired = paired_differences(metrics)
    paired.to_csv(OUT / "paired_differences.csv", index=False)
    for metric in ("reverse_auc", "test_auc"):
        plot_auc(metrics, metric, OUT)
    (OUT / "stage1_report.md").write_text(render_report(metrics, paired))


if __name__ == "__main__":
    main()
