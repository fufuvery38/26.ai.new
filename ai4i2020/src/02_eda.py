"""步驟 1：探索式資料分析、視覺化與 EDA 摘要。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from common import (
    FAILURE_COLORS,
    FEATURE_UNITS,
    FEATURE_ZH,
    FIG_DIR,
    MODE_COLORS,
    MODE_FLAGS,
    MODE_ZH,
    RESULT_DIR,
    SENSOR_FEATURES,
    TARGET,
    TYPE_ORDER,
    TYPE_ZH,
    add_derived_features,
    force_utf8_stdout,
    load_data,
    save_json,
    setup_plotting,
)

SUMMARY_JSON = RESULT_DIR / "eda_summary.json"
REPORT_MD = RESULT_DIR / "eda_summary.md"

DERIVED = ["power_w", "temp_diff_k", "wear_torque"]
ANALYSIS_FEATURES = SENSOR_FEATURES + DERIVED
SHORT_ZH = {
    "air_temp_k": "空氣溫",
    "process_temp_k": "製程溫",
    "rotational_speed_rpm": "轉速",
    "torque_nm": "扭矩",
    "tool_wear_min": "磨耗",
    "power_w": "功率",
    "temp_diff_k": "溫差",
    "wear_torque": "磨耗×扭矩",
    TARGET: "故障",
}
WEAR_BINS = [0, 50, 100, 150, 200, 240, 260]
WEAR_LABELS = ["0–50", "50–100", "100–150", "150–200", "200–240", "240–260"]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  圖：{path.name}")


def annotate_bars(ax, bars, values, labels, log: bool = False) -> None:
    for bar, value, label in zip(bars, values, labels):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y * 1.08 if log else y,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


# ---------------------------------------------------------------------------
# 圖 1–2：類別結構
# ---------------------------------------------------------------------------
def fig01_class_imbalance(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})

    counts = df[TARGET].value_counts().reindex([0, 1]).fillna(0).astype(int)
    ax = axes[0]
    bars = ax.bar(
        ["正常（0）", "故障（1）"],
        counts.values,
        color=[FAILURE_COLORS[0], FAILURE_COLORS[1]],
        width=0.6,
    )
    annotate_bars(
        ax,
        bars,
        counts.values,
        [f"{v:,}\n({v / len(df) * 100:.2f}%)" for v in counts.values],
    )
    ax.set_ylabel("筆數")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("目標類別：少數類僅 3.39%")
    ax.text(
        0.98,
        0.92,
        f"全猜多數類 accuracy = {counts.max() / len(df) * 100:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.4"},
    )

    ax = axes[1]
    mode_counts = df[MODE_FLAGS].sum().reindex(MODE_FLAGS)
    bars = ax.barh(
        [MODE_ZH[f] for f in MODE_FLAGS],
        mode_counts.values,
        color=[MODE_COLORS[f] for f in MODE_FLAGS],
    )
    for bar, value in zip(bars, mode_counts.values):
        ax.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value}（{value / len(df) * 100:.2f}%）",
            va="center",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.set_xlabel("筆數")
    ax.set_xlim(0, mode_counts.max() * 1.25)
    ax.set_title("五種故障模式（可重疊）")

    fig.suptitle("AI4I 2020：類別不平衡與故障模式組成", fontsize=14, y=1.02)
    save_fig(fig, "fig01_class_imbalance.png")


def fig02_mode_overlap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})

    per_row = df[MODE_FLAGS].sum(axis=1).value_counts().sort_index()
    ax = axes[0]
    bars = ax.bar(per_row.index.astype(str), per_row.values, color="#4c72b0", width=0.6)
    ax.set_yscale("log")
    annotate_bars(
        ax,
        bars,
        per_row.values,
        [f"{v:,}" for v in per_row.values],
        log=True,
    )
    ax.set_xlabel("同一筆資料的旗標數")
    ax.set_ylabel("筆數（對數刻度）")
    ax.set_ylim(0.7, per_row.max() * 3)
    ax.set_title("多數資料沒有任何模式旗標")

    combo = df.apply(
        lambda r: "+".join(f.upper() for f in MODE_FLAGS if r[f] == 1) or "無",
        axis=1,
    )
    combo_counts = combo.value_counts()
    combo_counts = combo_counts[combo_counts.index != "無"]
    ax = axes[1]
    colors = []
    for label in combo_counts.index:
        parts = label.split("+")
        colors.append(MODE_COLORS[parts[0].lower()] if len(parts) == 1 else "#6c5ce7")
    bars = ax.barh(combo_counts.index, combo_counts.values, color=colors)
    for bar, value in zip(bars, combo_counts.values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("筆數（不含 9,652 筆無旗標者）")
    ax.set_xlim(0, combo_counts.max() * 1.15)
    ax.set_title("模式組合：重疊案例相對少見")
    ax.text(
        0.98,
        0.05,
        f"324 筆單一模式、23 筆雙模式、1 筆三模式",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.4"},
    )

    fig.suptitle("故障模式的重疊結構", fontsize=14, y=1.02)
    save_fig(fig, "fig02_mode_overlap.png")


# ---------------------------------------------------------------------------
# 圖 3–5：特徵分布與模式剖面
# ---------------------------------------------------------------------------
def fig03_feature_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    for idx, col in enumerate(ANALYSIS_FEATURES):
        ax = axes[idx]
        normal = df.loc[df[TARGET] == 0, col]
        failure = df.loc[df[TARGET] == 1, col]
        bins = np.histogram_bin_edges(df[col], bins=45)
        ax.hist(normal, bins=bins, density=True, alpha=0.45, color=FAILURE_COLORS[0], label="正常")
        ax.hist(failure, bins=bins, density=True, alpha=0.45, color=FAILURE_COLORS[1], label="故障")
        ax.axvline(normal.median(), color=FAILURE_COLORS[0], ls="--", lw=1.2)
        ax.axvline(failure.median(), color=FAILURE_COLORS[1], ls="--", lw=1.2)
        unit = FEATURE_UNITS.get(col, "")
        ax.set_xlabel(f"{FEATURE_ZH[col]}（{unit}）" if unit else FEATURE_ZH[col])
        ax.set_ylabel("密度")
        ax.set_title(FEATURE_ZH[col])
    axes[-1].axis("off")
    handles = [
        Patch(facecolor=FAILURE_COLORS[0], alpha=0.5, label="正常（n=9,661）"),
        Patch(facecolor=FAILURE_COLORS[1], alpha=0.5, label="故障（n=339）"),
        Line2D([0], [0], color="gray", ls="--", lw=1.2, label="中位數"),
    ]
    axes[-1].legend(handles=handles, loc="center", fontsize=10, frameon=False)
    fig.suptitle("特徵分布：故障與正常的高度重疊", fontsize=14, y=0.995)
    fig.tight_layout()
    save_fig(fig, "fig03_feature_distributions.png")


def fig04_feature_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    for idx, col in enumerate(ANALYSIS_FEATURES):
        ax = axes[idx]
        groups = [df.loc[df[TARGET] == 0, col], df.loc[df[TARGET] == 1, col]]
        bp = ax.boxplot(
            groups,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.4},
        )
        for patch, color in zip(bp["boxes"], [FAILURE_COLORS[0], FAILURE_COLORS[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticklabels(["正常\n(9,661)", "故障\n(339)"])
        unit = FEATURE_UNITS.get(col, "")
        ax.set_ylabel(f"{FEATURE_ZH[col]}（{unit}）" if unit else FEATURE_ZH[col])
        ax.set_title(FEATURE_ZH[col])
    axes[-1].axis("off")
    fig.suptitle("各特徵的箱型圖（隱藏離群點以聚焦四分位結構）", fontsize=14, y=0.995)
    fig.tight_layout()
    save_fig(fig, "fig04_feature_boxplots.png")


def fig05_mode_profile(df: pd.DataFrame) -> None:
    rows = MODE_FLAGS + [TARGET]
    matrix = np.zeros((len(rows), len(ANALYSIS_FEATURES)))
    for i, flag in enumerate(rows):
        positive = df[flag] == 1
        for j, col in enumerate(ANALYSIS_FEATURES):
            matrix[i, j] = (
                df.loc[positive, col].median() - df.loc[~positive, col].median()
            ) / df[col].std()

    fig, ax = plt.subplots(figsize=(11, 5.4))
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(ANALYSIS_FEATURES)))
    ax.set_xticklabels([SHORT_ZH[c] for c in ANALYSIS_FEATURES])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["TWF", "HDF", "PWF", "OSF", "RNF", "整體故障"])
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if abs(value) > 1.6 else "#222222",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("標準化中位數差（正例 - 負例）")
    ax.set_title("各故障模式的特徵剖面：RNF 偏移較小且無一致方向")
    fig.tight_layout()
    save_fig(fig, "fig05_mode_profile.png")


# ---------------------------------------------------------------------------
# 圖 6：相關性
# ---------------------------------------------------------------------------
def fig06_correlation(df: pd.DataFrame) -> None:
    cols = ANALYSIS_FEATURES + [TARGET]
    labels = [SHORT_ZH[c] for c in cols]
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, matrix, title in zip(
        axes, [pearson, spearman], ["Pearson（線性）", "Spearman（單調秩相關）"]
    ):
        im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(value) > 0.65 else "#222222",
                )
        ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="相關係數")
    fig.suptitle("相關係數熱圖：整體相關偏弱，且掩蓋了分模式的異質性", fontsize=14, y=1.01)
    save_fig(fig, "fig06_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 圖 7–9：物理邊界
# ---------------------------------------------------------------------------
def fig07_power_boundary(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    normal = df[df[TARGET] == 0]
    other_fail = df[(df[TARGET] == 1) & (df["pwf"] == 0)]
    pwf = df[df["pwf"] == 1]

    ax.scatter(
        normal["rotational_speed_rpm"],
        normal["torque_nm"],
        s=5,
        c="#c9c9c9",
        alpha=0.25,
        rasterized=True,
        label="正常",
    )
    ax.scatter(
        other_fail["rotational_speed_rpm"],
        other_fail["torque_nm"],
        s=26,
        marker="x",
        c="#111111",
        linewidth=0.8,
        label="其他故障",
    )
    ax.scatter(
        pwf["rotational_speed_rpm"],
        pwf["torque_nm"],
        s=34,
        c=MODE_COLORS["pwf"],
        edgecolor="white",
        linewidth=0.5,
        label="PWF（95 筆）",
        zorder=4,
    )

    rpm_grid = np.linspace(1100, 2950, 400)
    t_low = 3500 * 60 / (2 * np.pi * rpm_grid)
    t_high = 9000 * 60 / (2 * np.pi * rpm_grid)
    ax.fill_between(rpm_grid, 0, t_low, color=MODE_COLORS["pwf"], alpha=0.08)
    ax.fill_between(rpm_grid, t_high, 85, color=MODE_COLORS["pwf"], alpha=0.08)
    ax.plot(rpm_grid, t_low, color=MODE_COLORS["pwf"], lw=1.6, ls="--")
    ax.plot(rpm_grid, t_high, color=MODE_COLORS["pwf"], lw=1.6, ls="--")
    ax.text(2000, 3500 * 60 / (2 * np.pi * 2000) - 3.2, "P = 3.5 kW", color=MODE_COLORS["pwf"], fontsize=9)
    ax.text(1500, 9000 * 60 / (2 * np.pi * 1500) + 2.0, "P = 9 kW", color=MODE_COLORS["pwf"], fontsize=9)
    ax.text(
        2650,
        8,
        "PWF 區\n（P < 3.5 kW）",
        color=MODE_COLORS["pwf"],
        fontsize=9,
        ha="center",
    )

    ax.set_xlim(1100, 2950)
    ax.set_ylim(0, 85)
    ax.set_xlabel("轉速（rpm）")
    ax.set_ylabel("扭矩（Nm）")
    ax.set_title("扭矩 × 轉速：PWF 正好落在功率區間之外（零誤判）")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig07_power_boundary.png")


def fig08_hdf_boundary(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    normal = df[df["hdf"] == 0]
    hdf = df[df["hdf"] == 1]

    ax.scatter(
        normal["rotational_speed_rpm"],
        normal["temp_diff_k"],
        s=5,
        c="#c9c9c9",
        alpha=0.25,
        rasterized=True,
        label="非 HDF",
    )
    ax.scatter(
        hdf["rotational_speed_rpm"],
        hdf["temp_diff_k"],
        s=34,
        c=MODE_COLORS["hdf"],
        edgecolor="white",
        linewidth=0.5,
        label="HDF（115 筆）",
        zorder=4,
    )
    ax.add_patch(
        Rectangle(
            (1100, 7.5),
            280,
            1.1,
            facecolor=MODE_COLORS["hdf"],
            alpha=0.10,
            edgecolor=MODE_COLORS["hdf"],
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax.axvline(1380, color=MODE_COLORS["hdf"], ls="--", lw=1.2)
    ax.axhline(8.6, color=MODE_COLORS["hdf"], ls="--", lw=1.2)
    ax.text(1150, 7.7, "HDF 區：ΔT < 8.6 K 且轉速 < 1,380 rpm", color=MODE_COLORS["hdf"], fontsize=9)
    ax.set_xlim(1100, 2950)
    ax.set_ylim(7.5, 12.5)
    ax.set_xlabel("轉速（rpm）")
    ax.set_ylabel("製程溫 - 空氣溫（K）")
    ax.set_title("溫差 × 轉速：HDF 邊界可由兩條門檻完全還原（零誤判）")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig08_hdf_boundary.png")


def fig09_wear_torque(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    normal = df[(df["osf"] == 0) & (df["twf"] == 0)]
    osf = df[df["osf"] == 1]
    twf = df[df["twf"] == 1]

    ax.scatter(
        normal["tool_wear_min"],
        normal["torque_nm"],
        s=5,
        c="#d0d0d0",
        alpha=0.22,
        rasterized=True,
        label="非 OSF／TWF",
    )
    ax.scatter(
        osf["tool_wear_min"],
        osf["torque_nm"],
        s=34,
        c=MODE_COLORS["osf"],
        edgecolor="white",
        linewidth=0.5,
        label="OSF（98 筆）",
        zorder=4,
    )
    ax.scatter(
        twf["tool_wear_min"],
        twf["torque_nm"],
        s=38,
        marker="^",
        c=MODE_COLORS["twf"],
        edgecolor="white",
        linewidth=0.5,
        label="TWF（46 筆）",
        zorder=5,
    )
    ax.add_patch(
        Rectangle(
            (200, 46),
            40,
            34,
            facecolor="#999999",
            alpha=0.08,
            edgecolor="#666666",
            linestyle="--",
            linewidth=1.0,
        )
    )
    ax.text(202, 77.5, "文獻簡化區間\n磨耗 200–240 且高扭矩", fontsize=8.5, color="#444444")
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 82)
    ax.set_xlabel("工具磨耗（min）")
    ax.set_ylabel("扭矩（Nm）")
    ax.set_title("磨耗 × 扭矩：OSF／TWF 集中於高磨耗區，但邊界不是決定性的")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig09_wear_torque.png")


# ---------------------------------------------------------------------------
# 圖 10–11：機型、磨耗分桶與序列
# ---------------------------------------------------------------------------
def _wear_bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    bucket = pd.cut(
        df["tool_wear_min"],
        bins=WEAR_BINS,
        labels=WEAR_LABELS,
        right=False,
        include_lowest=True,
    )
    table = (
        df.assign(bucket=bucket)
        .groupby("bucket", observed=True)[TARGET]
        .agg(["count", "sum", "mean"])
        .reindex(WEAR_LABELS)
    )
    table["ci_low"] = [wilson_ci(int(k), int(n))[0] for k, n in zip(table["sum"], table["count"])]
    table["ci_high"] = [wilson_ci(int(k), int(n))[1] for k, n in zip(table["sum"], table["count"])]
    return table


def fig10_type_wear(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.6, 1.4]})

    ax = axes[0]
    rates, lows, highs, ns = [], [], [], []
    for t in TYPE_ORDER:
        sub = df[df["type"] == t]
        k, n = int(sub[TARGET].sum()), len(sub)
        low, high = wilson_ci(k, n)
        rates.append(k / n)
        lows.append(k / n - low)
        highs.append(high - k / n)
        ns.append(n)
    type_zh = [f"{t}\n{TYPE_ZH[t]}" for t in TYPE_ORDER]
    bars = ax.bar(type_zh, rates, yerr=[lows, highs], capsize=5, color="#4c72b0", width=0.6)
    annotate_bars(ax, bars, rates, [f"{r * 100:.2f}%\n(n={n:,})" for r, n in zip(rates, ns)])
    ax.set_ylabel("故障率")
    ax.set_ylim(0, max(rates) * 1.45)
    ax.set_title("依機型")

    ax = axes[1]
    table = _wear_bucket_table(df)
    colors = ["#4c72b0"] * 4 + ["#dd8452", "#c44e52"]
    bars = ax.bar(
        table.index,
        table["mean"],
        yerr=[table["mean"] - table["ci_low"], table["ci_high"] - table["mean"]],
        capsize=4,
        color=colors,
        width=0.65,
    )
    annotate_bars(
        ax,
        bars,
        table["mean"],
        [f"{r * 100:.1f}%\n({int(n)})" for r, n in zip(table["mean"], table["count"])],
    )
    ax.set_ylabel("故障率")
    ax.set_xlabel("工具磨耗（min）")
    ax.set_ylim(0, 0.42)
    ax.set_title("依磨耗分桶（括號為筆數）")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[2]
    tmp = df.assign(
        bucket=pd.cut(
            df["tool_wear_min"], bins=WEAR_BINS, labels=WEAR_LABELS, right=False, include_lowest=True
        )
    )
    rate_df = (
        tmp.pivot_table(index="type", columns="bucket", values=TARGET, aggfunc="mean", observed=True)
        .reindex(index=TYPE_ORDER, columns=WEAR_LABELS)
    )
    count_df = (
        tmp.pivot_table(index="type", columns="bucket", values=TARGET, aggfunc="count", observed=True)
        .reindex(index=TYPE_ORDER, columns=WEAR_LABELS)
    )
    rate_m = rate_df.to_numpy(dtype=float)
    count_m = count_df.to_numpy(dtype=float)
    im = ax.imshow(rate_m, cmap="Reds", vmin=0, vmax=0.20, aspect="auto")
    ax.set_xticks(range(len(WEAR_LABELS)))
    ax.set_xticklabels(WEAR_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(TYPE_ORDER)))
    ax.set_yticklabels(TYPE_ORDER)
    ax.grid(False)
    for i in range(rate_m.shape[0]):
        for j in range(rate_m.shape[1]):
            rate = rate_m[i, j]
            count = count_m[i, j]
            if np.isnan(rate):
                continue
            ax.text(
                j,
                i,
                f"{rate * 100:.1f}%\n(n={int(count)})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if rate > 0.14 else "#222222",
            )
    ax.set_title("機型 × 磨耗：顏色上限 20%")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="故障率")

    fig.suptitle("機型與磨耗的故障率：邊際差異受操作條件混雜", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig10_type_wear.png")


def fig11_wear_sequence(df: pd.DataFrame) -> None:
    wear = df["tool_wear_min"].to_numpy()
    udi = df["udi"].to_numpy()
    resets = np.where(np.diff(wear) < 0)[0] + 1
    failures = df[TARGET].to_numpy() == 1

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharey=False)

    ax = axes[0]
    ax.plot(udi, wear, color="#8c8c8c", lw=0.6)
    ax.scatter(udi[failures], wear[failures], s=12, c=FAILURE_COLORS[1], label="故障", zorder=3)
    ax.scatter(udi[resets], np.full(len(resets), 1), s=12, c="#f0ad4e", label="磨耗重置", zorder=3)
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 265)
    ax.set_xlabel("UDI（資料生成順序）")
    ax.set_ylabel("工具磨耗（min）")
    ax.set_title(f"全程：{len(resets)} 次重置、{len(resets) + 1} 個週期")
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    ax = axes[1]
    mask = udi <= 1000
    ax.plot(udi[mask], wear[mask], color="#555555", lw=0.9)
    ax.scatter(udi[mask & failures], wear[mask & failures], s=26, c=FAILURE_COLORS[1], zorder=3)
    local_resets = resets[resets < 1000]
    ax.scatter(udi[local_resets], np.full(len(local_resets), 1), s=26, c="#f0ad4e", zorder=3)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 265)
    ax.set_xlabel("UDI（前 1,000 筆）")
    ax.set_ylabel("工具磨耗（min）")
    ax.set_title("放大：磨耗逐步累積至 200–240 min 後歸零，橘點為重置")

    fig.suptitle("工具磨耗的累積與重置型態（UDI 只是生成順序，不是真實時間）", fontsize=14, y=0.995)
    fig.tight_layout()
    save_fig(fig, "fig11_wear_sequence.png")


# ---------------------------------------------------------------------------
# 摘要
# ---------------------------------------------------------------------------
def _series_stats(series: pd.Series) -> dict:
    return {
        "n": int(series.size),
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
        "median": round(float(series.median()), 4),
        "q25": round(float(series.quantile(0.25)), 4),
        "q75": round(float(series.quantile(0.75)), 4),
    }


def build_summary(df: pd.DataFrame) -> dict:
    target_counts = df[TARGET].value_counts().reindex([0, 1]).fillna(0).astype(int)
    mode_counts = {f: int(df[f].sum()) for f in MODE_FLAGS}
    per_row = df[MODE_FLAGS].sum(axis=1).value_counts().sort_index()
    combo = df.apply(
        lambda r: "+".join(f.upper() for f in MODE_FLAGS if r[f] == 1) or "none", axis=1
    ).value_counts()

    type_stats = {}
    for t in TYPE_ORDER:
        sub = df[df["type"] == t]
        k, n = int(sub[TARGET].sum()), len(sub)
        low, high = wilson_ci(k, n)
        type_stats[t] = {
            "count": n,
            "failures": k,
            "rate_pct": round(k / n * 100, 4),
            "ci95_pct": [round(low * 100, 4), round(high * 100, 4)],
        }

    bucket_table = _wear_bucket_table(df)
    wear_buckets = []
    for label, row in bucket_table.iterrows():
        wear_buckets.append(
            {
                "bucket": label,
                "count": int(row["count"]),
                "failures": int(row["sum"]),
                "rate_pct": round(float(row["mean"]) * 100, 4),
                "ci95_pct": [round(float(row["ci_low"]) * 100, 4), round(float(row["ci_high"]) * 100, 4)],
            }
        )

    feature_stats_target = {
        col: {
            "normal": _series_stats(df.loc[df[TARGET] == 0, col]),
            "failure": _series_stats(df.loc[df[TARGET] == 1, col]),
        }
        for col in ANALYSIS_FEATURES
    }

    feature_profile = {}
    for flag in MODE_FLAGS:
        positive = df[flag] == 1
        feature_profile[flag] = {
            col: {
                "positive_median": round(float(df.loc[positive, col].median()), 4),
                "negative_median": round(float(df.loc[~positive, col].median()), 4),
                "z_median_diff": round(
                    float((df.loc[positive, col].median() - df.loc[~positive, col].median()) / df[col].std()),
                    4,
                ),
            }
            for col in ANALYSIS_FEATURES
        }

    corr_cols = ANALYSIS_FEATURES + [TARGET]
    pearson = df[corr_cols].corr(method="pearson").round(3)
    spearman = df[corr_cols].corr(method="spearman").round(3)
    target_corr = {
        col: {
            "pearson": float(pearson.loc[col, TARGET]),
            "spearman": float(spearman.loc[col, TARGET]),
        }
        for col in ANALYSIS_FEATURES
    }

    wear = df["tool_wear_min"].to_numpy()
    resets = np.where(np.diff(wear) < 0)[0] + 1
    cycles = np.diff(np.r_[0, resets, len(wear)])

    power_w = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60
    temp_diff = df["process_temp_k"] - df["air_temp_k"]
    pwf_rule = (power_w < 3500) | (power_w > 9000)
    hdf_rule = (temp_diff < 8.6) & (df["rotational_speed_rpm"] < 1380)

    def _counts(pred: pd.Series, actual: pd.Series) -> dict:
        pred, actual = pred.astype(bool), actual.astype(bool)
        return {
            "tp": int((pred & actual).sum()),
            "fp": int((pred & ~actual).sum()),
            "fn": int((~pred & actual).sum()),
        }

    anomalies = {
        "rnf_positive_target_zero": int(((df["rnf"] == 1) & (df[TARGET] == 0)).sum()),
        "target_positive_without_mode": int(
            ((df[TARGET] == 1) & (df[MODE_FLAGS].sum(axis=1) == 0)).sum()
        ),
        "total": 27,
    }

    return {
        "n": int(len(df)),
        "target": {
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "rate_pct": round(float(df[TARGET].mean() * 100), 4),
            "majority_baseline_accuracy_pct": round(float(target_counts.max() / len(df) * 100), 4),
        },
        "modes": {
            f: {"count": mode_counts[f], "rate_pct": round(mode_counts[f] / len(df) * 100, 4)}
            for f in MODE_FLAGS
        },
        "flags_per_row": {str(k): int(v) for k, v in per_row.items()},
        "combinations": {str(k): int(v) for k, v in combo.items()},
        "types": type_stats,
        "wear_buckets": wear_buckets,
        "feature_stats_by_target": feature_stats_target,
        "feature_profile_z": feature_profile,
        "correlations": {
            "target": target_corr,
            "pearson_matrix": pearson.to_dict(),
            "spearman_matrix": spearman.to_dict(),
        },
        "wear_sequence": {
            "resets": int(len(resets)),
            "cycles": int(len(resets) + 1),
            "cycle_length_mean": round(float(cycles.mean()), 3),
            "cycle_length_median": round(float(np.median(cycles)), 3),
            "cycle_length_min": int(cycles.min()),
            "cycle_length_max": int(cycles.max()),
            "max_wear_min": int(wear.max()),
            "wear_before_reset_max": int(wear[resets - 1].max()) if len(resets) else 0,
        },
        "rules": {
            "pwf": _counts(pwf_rule, df["pwf"]),
            "hdf": _counts(hdf_rule, df["hdf"]),
        },
        "label_anomalies": anomalies,
    }


def write_report(df: pd.DataFrame, summary: dict) -> None:
    t = summary["target"]
    modes = summary["modes"]
    types = summary["types"]
    buckets = summary["wear_buckets"]
    corr = summary["correlations"]["target"]
    wear_seq = summary["wear_sequence"]

    def pct(value: float) -> str:
        return f"{value:.2f}%"

    lines = [
        "# AI4I 2020 探索式資料分析摘要",
        "",
        "由 `src/02_eda.py` 自動產生；圖檔位於 `figures/`。",
        "",
        "## 1. 類別結構",
        "",
        f"- 總筆數 **{summary['n']:,}**，目標正例 **{t['counts']['1']:,} 筆（{pct(t['rate_pct'])}）**。",
        f"- 全猜多數類即可取得 **{pct(t['majority_baseline_accuracy_pct'])} accuracy**，因此 accuracy 不適合作為主要指標。",
        "- 五種模式可重疊：324 筆單一模式、23 筆雙模式、1 筆三模式。",
        "",
        "| 模式 | 筆數 | 占比 |",
        "| --- | ---: | ---: |",
    ]
    for flag in MODE_FLAGS:
        lines.append(f"| {flag.upper()} | {modes[flag]['count']} | {pct(modes[flag]['rate_pct'])} |")

    lines += [
        "",
        "![類別不平衡](../figures/fig01_class_imbalance.png)",
        "",
        "![模式重疊](../figures/fig02_mode_overlap.png)",
        "",
        "## 2. 特徵分布與模式剖面",
        "",
        "- 五個感測特徵的邊際分布高度重疊，單一特徵不足以分離故障。",
        "- 功率與溫差是少數具備清楚物理意義的衍生特徵，分別對應 PWF 與 HDF。",
        "- RNF 的特徵偏移較小且方向不一致（|z| ≤ 0.8），符合隨機失效難以學習的預期；仍需在階段 5 以模型驗證是否僅為雜訊。",
        "",
        "![特徵分布](../figures/fig03_feature_distributions.png)",
        "",
        "![箱型圖](../figures/fig04_feature_boxplots.png)",
        "",
        "![模式剖面](../figures/fig05_mode_profile.png)",
        "",
        "## 3. 相關性",
        "",
        "| 特徵 | Pearson | Spearman |",
        "| --- | ---: | ---: |",
    ]
    for col in ANALYSIS_FEATURES:
        lines.append(
            f"| {SHORT_ZH[col]} | {corr[col]['pearson']:+.3f} | {corr[col]['spearman']:+.3f} |"
        )

    lines += [
        "",
        "Pearson 假設線性關係、對極端值敏感，且二元目標在 3.39% 正例下會衰減係數；"
        "Spearman 只看單調性，較適合此處的初步篩選。兩者都只描述邊際關聯，"
        "會掩蓋「不同故障模式由不同特徵驅動」的異質性，因此不能取代分模式分析。",
        "",
        "![相關性](../figures/fig06_correlation_heatmap.png)",
        "",
        "## 4. 物理邊界",
        "",
        f"- **PWF**：功率 < 3,500 W 或 > 9,000 W，規則命中 TP={summary['rules']['pwf']['tp']}、"
        f"FP={summary['rules']['pwf']['fp']}、FN={summary['rules']['pwf']['fn']}。",
        f"- **HDF**：溫差 < 8.6 K 且轉速 < 1,380 rpm，規則命中 TP={summary['rules']['hdf']['tp']}、"
        f"FP={summary['rules']['hdf']['fp']}、FN={summary['rules']['hdf']['fn']}。",
        "- **OSF／TWF**：集中於高磨耗區，但 OSF 正例最低磨耗 172 min、TWF 最低 198 min，"
        "且有 TWF 超過 240 min；簡化磨耗窗不是必要條件。",
        "",
        "![PWF 邊界](../figures/fig07_power_boundary.png)",
        "",
        "![HDF 邊界](../figures/fig08_hdf_boundary.png)",
        "",
        "![磨耗與扭矩](../figures/fig09_wear_torque.png)",
        "",
        "## 5. 機型與磨耗",
        "",
        "| 機型 | 筆數 | 故障數 | 故障率 | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for t_key in TYPE_ORDER:
        row = types[t_key]
        lines.append(
            f"| {t_key} | {row['count']:,} | {row['failures']} | {pct(row['rate_pct'])} | "
            f"{row['ci95_pct'][0]:.2f}–{row['ci95_pct'][1]:.2f}% |"
        )

    lines += [
        "",
        "| 磨耗區間（min） | 筆數 | 故障數 | 故障率 | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in buckets:
        lines.append(
            f"| {row['bucket']} | {row['count']:,} | {row['failures']} | {pct(row['rate_pct'])} | "
            f"{row['ci95_pct'][0]:.2f}–{row['ci95_pct'][1]:.2f}% |"
        )

    lines += [
        "",
        "邊際上 L 的故障率高於 H，但這不是「低階機型體質較差」的證據："
        "機型的扭矩與轉速分布不同，且 200–240 min 分桶內 L 的故障率仍高於 H，"
        "顯示磨耗與操作條件共同作用。H 在 240 min 以上僅 3 筆，估計極不穩定。",
        "",
        "![機型與磨耗](../figures/fig10_type_wear.png)",
        "",
        "## 6. 磨耗序列",
        "",
        f"- 共 **{wear_seq['resets']} 次重置**、**{wear_seq['cycles']} 個週期**，磨耗每次歸零。",
        f"- 週期長度平均 {wear_seq['cycle_length_mean']:.1f} 筆、中位數 {wear_seq['cycle_length_median']:.0f} 筆"
        f"（{wear_seq['cycle_length_min']}–{wear_seq['cycle_length_max']}）。",
        f"- 最大磨耗 {wear_seq['max_wear_min']} min；重置前的磨耗最高 {wear_seq['wear_before_reset_max']} min。",
        "- UDI 只是生成順序，不是真實時間；因此後續不以 UDI 作為特徵，但可用來做時間切分的穩健性對照。",
        "",
        "![磨耗序列](../figures/fig11_wear_sequence.png)",
        "",
        "## 7. 重點觀察",
        "",
        "1. 這是一份**合成資料**：PWF 與 HDF 可由規則完全還原，模型若在此拿下高分，"
        "代表它學到了生成規則，而非發現了新的物理現象。",
        "2. 整體 PR-AUC 會被 HDF／PWF 這兩個「送分題」拉高，必須拆模式呈現。",
        "3. OSF／TWF 的邊界重疊且含隨機成分，是較誠實的機器學習問題。",
        "4. RNF 提供了一個天然的下限對照：任何模型都不該顯著優於隨機。",
        "5. 類別不平衡使 accuracy 無意義；後續以 PR-AUC、recall@precision 與成本矩陣為主。",
        "",
        "## 圖表清單",
        "",
        "| 圖 | 檔案 |",
        "| --- | --- |",
        "| 類別不平衡 | `fig01_class_imbalance.png` |",
        "| 模式重疊 | `fig02_mode_overlap.png` |",
        "| 特徵分布 | `fig03_feature_distributions.png` |",
        "| 箱型圖 | `fig04_feature_boxplots.png` |",
        "| 模式剖面 | `fig05_mode_profile.png` |",
        "| 相關性 | `fig06_correlation_heatmap.png` |",
        "| PWF 邊界 | `fig07_power_boundary.png` |",
        "| HDF 邊界 | `fig08_hdf_boundary.png` |",
        "| 磨耗與扭矩 | `fig09_wear_torque.png` |",
        "| 機型與磨耗 | `fig10_type_wear.png` |",
        "| 磨耗序列 | `fig11_wear_sequence.png` |",
        "",
        "## 已知資料問題",
        "",
        f"- RNF=1 但 target=0：{summary['label_anomalies']['rnf_positive_target_zero']} 筆。",
        f"- target=1 但沒有任何模式旗標：{summary['label_anomalies']['target_positive_without_mode']} 筆。",
        "- 完整 UDI 清單見 `results/data_audit.md`。",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    force_utf8_stdout()
    setup_plotting()
    df = add_derived_features(load_data())
    print(f"載入 {len(df):,} 筆，開始產生 EDA 圖表")

    fig01_class_imbalance(df)
    fig02_mode_overlap(df)
    fig03_feature_distributions(df)
    fig04_feature_boxplots(df)
    fig05_mode_profile(df)
    fig06_correlation(df)
    fig07_power_boundary(df)
    fig08_hdf_boundary(df)
    fig09_wear_torque(df)
    fig10_type_wear(df)
    fig11_wear_sequence(df)

    print("彙整 EDA 摘要")
    summary = build_summary(df)
    save_json(summary, SUMMARY_JSON)
    write_report(df, summary)

    print(f"完成：{SUMMARY_JSON.name}、{REPORT_MD.name}，共 11 張圖")
    print(
        f"目標 {summary['target']['counts']['1']} 筆（{summary['target']['rate_pct']:.2f}%）；"
        f"多數類基準 accuracy {summary['target']['majority_baseline_accuracy_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
