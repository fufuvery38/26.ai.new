"""02 - 資料基本探索與檢查。

產出：
    results/eda_summary.json / .md
    figures/fig01_class_distribution.png
    figures/fig02_feature_histograms.png
    figures/fig03_feature_boxplots.png
    figures/fig04_correlation_heatmap.png
    figures/fig05_pairplot.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif

import matplotlib.pyplot as plt
from common import (
    FEATURES,
    FEATURES_ZH,
    FIG_DIR,
    PALETTE,
    RESULT_DIR,
    SPECIES,
    SPECIES_ZH,
    force_utf8_stdout,
    load_iris_df,
    save_json,
    setup_plotting,
)

sns.set_theme(style="whitegrid", rc={"grid.alpha": 0.25})
setup_plotting()


def main() -> None:
    df = load_iris_df()
    print("=" * 70)
    print("步驟 2/4：資料探索與檢查")
    print("=" * 70)

    counts = df["species"].value_counts().reindex(SPECIES)
    missing = df.isna().sum()
    duplicated = int(df.duplicated().sum())
    desc = df[FEATURES].describe().T
    desc["cv"] = desc["std"] / desc["mean"]

    # 以 IQR 1.5 倍規則標記極端值（僅標記，不刪除）
    outlier_rows = []
    outlier_flags = pd.Series(False, index=df.index)
    for c in FEATURES:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag = (df[c] < lo) | (df[c] > hi)
        outlier_flags |= flag
        outlier_rows.append(
            {
                "feature": c,
                "q1": q1,
                "q3": q3,
                "lower": lo,
                "upper": hi,
                "n_outliers": int(flag.sum()),
                "outliers": df.loc[flag, ["species", c]].to_dict("records"),
            }
        )

    # 特徵與類別的關聯強度
    X = df[FEATURES].to_numpy()
    y = df["species"].cat.codes.to_numpy()
    f_vals, p_vals = f_classif(X, y)
    mi = mutual_info_classif(X, y, random_state=42)

    corr = df[FEATURES].corr()
    print("\n[類別分布]")
    for s in SPECIES:
        print(f"  {s:<11} {counts[s]:>3} 筆（{counts[s] / len(df):.1%}）")
    print(f"\n[缺失值] 總計 {int(missing.sum())} 個；[完全重複列] {duplicated} 列")
    print(f"[極端值] 依 1.5×IQR 規則共 {int(outlier_flags.sum())} 筆（保留，因屬真實生物變異）")

    print("\n[敘述統計]")
    print(desc[["mean", "std", "min", "25%", "50%", "75%", "max", "cv"]].round(3).to_string())

    print("\n[特徵與品種關聯]")
    for i, c in enumerate(FEATURES):
        print(f"  {c:<14} F={f_vals[i]:7.2f}  p={p_vals[i]:.2e}  MI={mi[i]:.3f}")

    print("\n[相關係數矩陣]")
    print(corr.round(3).to_string())

    # ------------------------------------------------------------------
    # 圖 1：類別分布
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    bars = axes[0].bar(
        [SPECIES_ZH[s] for s in SPECIES],
        counts.to_numpy(),
        color=[PALETTE[s] for s in SPECIES],
        width=0.6,
    )
    axes[0].bar_label(bars, padding=3)
    axes[0].set_title("各品種樣本數（類別平衡）")
    axes[0].set_ylabel("樣本數")
    axes[0].set_ylim(0, 60)

    sns.stripplot(
        data=df,
        x="species",
        y="petal_length",
        hue="species",
        palette=PALETTE,
        order=SPECIES,
        size=5,
        alpha=0.8,
        jitter=0.18,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xticks(range(3), [SPECIES_ZH[s] for s in SPECIES])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("花瓣長度 (cm)")
    axes[1].set_title("花瓣長度已可大致分開三個品種")
    fig.suptitle("圖 1　Iris 資料類別分布", fontsize=14, y=1.02)
    fig.savefig(FIG_DIR / "fig01_class_distribution.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 2：各特徵的類別直方圖
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, c in zip(axes.ravel(), FEATURES):
        sns.histplot(
            data=df,
            x=c,
            hue="species",
            hue_order=SPECIES,
            palette=PALETTE,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            alpha=0.28,
            ax=ax,
            legend=(c == "sepal_length"),
        )
        ax.set_title(FEATURES_ZH[c])
        ax.set_xlabel(f"{FEATURES_ZH[c]} (cm)")
        ax.set_ylabel("密度")
    axes[0, 0].legend(
        handles=[plt.Line2D([], [], color=PALETTE[s], lw=6, label=SPECIES_ZH[s]) for s in SPECIES],
        title="品種",
        loc="upper right",
    )
    fig.suptitle("圖 2　四項特徵依品種的分布", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_feature_histograms.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 3：箱型圖
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for ax, c in zip(axes, FEATURES):
        sns.boxplot(
            data=df,
            x="species",
            y=c,
            hue="species",
            order=SPECIES,
            palette=PALETTE,
            width=0.55,
            legend=False,
            ax=ax,
            fliersize=3,
        )
        ax.set_title(FEATURES_ZH[c])
        ax.set_xlabel("")
        ax.set_ylabel("cm")
        ax.set_xticks(range(3), ["setosa", "versicolor", "virginica"], rotation=25, ha="right")
    fig.suptitle("圖 3　特徵箱型圖：花瓣特徵具較佳的類別分離度", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_feature_boxplots.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 4：相關係數熱圖
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        cbar_kws={"label": "Pearson 相關係數", "shrink": 0.8},
        ax=ax,
    )
    ax.set_xticklabels([FEATURES_ZH[c] for c in FEATURES], rotation=20, ha="right")
    ax.set_yticklabels([FEATURES_ZH[c] for c in FEATURES], rotation=0)
    ax.set_title("圖 4　特徵間相關係數")
    fig.savefig(FIG_DIR / "fig04_correlation_heatmap.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 5：成對散點圖
    # ------------------------------------------------------------------
    g = sns.pairplot(
        df,
        vars=FEATURES,
        hue="species",
        hue_order=SPECIES,
        palette=PALETTE,
        corner=False,
        diag_kind="kde",
        plot_kws={"s": 26, "alpha": 0.75, "edgecolor": "white", "linewidth": 0.3},
        height=2.0,
    )
    g.figure.suptitle("圖 5　特徵成對散點圖（下三角：散點；對角：核密度）", y=1.01, fontsize=13)
    g._legend.set_title("品種")
    for t, s in zip(g._legend.texts, SPECIES):
        t.set_text(SPECIES_ZH[s])
    g.savefig(FIG_DIR / "fig05_pairplot.png")
    plt.close("all")

    # ------------------------------------------------------------------
    # 輸出摘要
    # ------------------------------------------------------------------
    summary = {
        "n_rows": int(len(df)),
        "n_features": len(FEATURES),
        "class_counts": {s: int(counts[s]) for s in SPECIES},
        "missing_total": int(missing.sum()),
        "duplicate_rows": duplicated,
        "iqr_outlier_rows": int(outlier_flags.sum()),
        "iqr_outlier_detail": outlier_rows,
        "describe": desc.round(4).to_dict(orient="index"),
        "correlation": corr.round(4).to_dict(),
        "feature_association": {
            c: {"F": float(f_vals[i]), "p_value": float(p_vals[i]), "mutual_info": float(mi[i])}
            for i, c in enumerate(FEATURES)
        },
        "strongest_correlation": {
            "pair": ["petal_length", "petal_width"],
            "r": float(corr.loc["petal_length", "petal_width"]),
        },
    }
    save_json(summary, RESULT_DIR / "eda_summary.json")

    md = "# 資料探索摘要\n\n"
    md += f"- 樣本數：{len(df)}；特徵數：{len(FEATURES)}\n"
    md += f"- 類別分布：{', '.join(f'{s}={counts[s]}' for s in SPECIES)}（完全平衡）\n"
    md += f"- 缺失值：{int(missing.sum())}；完全重複列：{duplicated}\n"
    md += f"- 1.5×IQR 極端值：{int(outlier_flags.sum())} 筆（保留）\n\n"
    md += "## 敘述統計\n\n"
    md += "| 特徵 | 平均 | 標準差 | 最小 | Q1 | 中位數 | Q3 | 最大 | 變異係數 |\n"
    md += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for c in FEATURES:
        r = df[c].describe()
        md += (
            f"| {FEATURES_ZH[c]} | {r['mean']:.3f} | {r['std']:.3f} | {r['min']:.1f} | "
            f"{r['25%']:.2f} | {r['50%']:.2f} | {r['75%']:.2f} | {r['max']:.1f} | "
            f"{df[c].std() / df[c].mean():.3f} |\n"
        )
    md += "\n## 特徵與品種的關聯強度（單因子變異數分析）\n\n"
    md += "| 特徵 | F 值 | p 值 | 互資訊 |\n| --- | ---: | ---: | ---: |\n"
    for i, c in enumerate(FEATURES):
        md += f"| {FEATURES_ZH[c]} | {f_vals[i]:.2f} | {p_vals[i]:.2e} | {mi[i]:.4f} |\n"
    md += "\n## 相關係數（Pearson）\n\n"
    md += "| | " + " | ".join(FEATURES_ZH[c] for c in FEATURES) + " |\n"
    md += "| --- | " + " | ".join("---:" for _ in FEATURES) + " |\n"
    for c in FEATURES:
        md += f"| **{FEATURES_ZH[c]}** | " + " | ".join(f"{corr.loc[c, k]:.3f}" for k in FEATURES) + " |\n"
    md += "\n## 極端值（1.5×IQR）\n\n"
    for row in outlier_rows:
        md += f"- {FEATURES_ZH[row['feature']]}：{row['n_outliers']} 筆，合理範圍 [{row['lower']:.2f}, {row['upper']:.2f}]\n"
    md += "\n> 極端值多為 setosa 的花瓣或 virginica 的花萼，屬於真實生物個體差異，因此保留不刪除。\n"
    (RESULT_DIR / "eda_summary.md").write_text(md, encoding="utf-8")

    print("\n[完成] 已輸出 5 張 EDA 圖與 results/eda_summary.json / .md")


if __name__ == "__main__":
    force_utf8_stdout()
    main()
