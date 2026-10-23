"""03 - 主成分分析（PCA）與 3D 特徵分布圖。

產出：
    results/pca_summary.json
    results/pca_loadings.csv
    results/pca_scores.csv
    figures/fig06_pca_scree.png
    figures/fig07_pca_3d.png
    figures/fig08_pca_3d_interactive.html
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (註冊 3D projection)

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


def main() -> None:
    df = load_iris_df()
    X = df[FEATURES].to_numpy()

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=4, random_state=42)
    scores = pca.fit_transform(Z)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=[f"PC{i + 1}" for i in range(4)])

    print("=" * 70)
    print("步驟 3/4：PCA 與 3D 特徵分布")
    print("=" * 70)
    print("[解釋變異量]")
    for i, (r, c) in enumerate(zip(evr, cum), start=1):
        print(f"  PC{i}: {r:7.2%}  累積 {c:7.2%}")
    print(f"[維度] 前 3 個主成分保留 {cum[2]:.2%} 的總變異量")
    print("\n[主成分負荷量]")
    print(loadings.round(3).to_string())

    # ------------------------------------------------------------------
    # 圖 6：陡坡圖
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(1, 5)
    bars = ax1.bar(xs, evr * 100, color="#4c72b0", width=0.55, label="個別解釋變異量")
    ax1.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax1.set_xlabel("主成分")
    ax1.set_ylabel("解釋變異量 (%)")
    ax1.set_xticks(xs, [f"PC{i}" for i in xs])
    ax1.set_ylim(0, 85)

    ax2 = ax1.twinx()
    ax2.plot(xs, cum * 100, "o-", color="#c44e52", lw=2, ms=5, label="累積解釋變異量")
    ax2.set_ylabel("累積解釋變異量 (%)")
    ax2.set_ylim(0, 105)
    for x, c in zip(xs, cum * 100):
        ax2.annotate(f"{c:.1f}%", (x, c), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color="#c44e52")
    ax2.grid(False)
    ax1.set_title("圖 6　PCA 陡坡圖：前三個主成分已保留 99.5% 變異")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_pca_scree.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 7：3D 靜態圖（兩個視角）
    # ------------------------------------------------------------------
    species = df["species"].astype(str).to_numpy()
    fig = plt.figure(figsize=(13.5, 6.2))
    views = [(22, -60, "視角 A"), (18, 125, "視角 B")]
    for k, (elev, azim, label) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, k, projection="3d")
        for s in SPECIES:
            m = species == s
            ax.scatter(
                scores[m, 0],
                scores[m, 1],
                scores[m, 2],
                s=34,
                color=PALETTE[s],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.85,
                label=SPECIES_ZH[s],
                depthshade=True,
            )
        ax.set_xlabel(f"PC1（{evr[0]:.1%}）", labelpad=6)
        ax.set_ylabel(f"PC2（{evr[1]:.1%}）", labelpad=6)
        ax.set_zlabel(f"PC3（{evr[2]:.1%}）", labelpad=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.06)
        if k == 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.16, 1.03), title="品種")
    fig.suptitle("圖 7　Iris 資料的 PCA 三維特徵分布（PC1–PC3）", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_pca_3d.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 表：主成分負荷量
    # ------------------------------------------------------------------
    loadings.round(4).to_csv(RESULT_DIR / "pca_loadings.csv", encoding="utf-8-sig")

    pca_df = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(4)])
    pca_df.insert(0, "sample_id", np.arange(1, len(df) + 1))
    pca_df["species"] = species
    pca_df = pd.concat([pca_df, df[FEATURES].reset_index(drop=True)], axis=1)
    pca_df.to_csv(RESULT_DIR / "pca_scores.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 圖 8：互動式 3D（Plotly）
    # ------------------------------------------------------------------
    plot_df = pca_df.copy()
    plot_df["品種"] = plot_df["species"].map(SPECIES_ZH)
    plot_df["花萼長度"] = plot_df["sepal_length"]
    plot_df["花萼寬度"] = plot_df["sepal_width"]
    plot_df["花瓣長度"] = plot_df["petal_length"]
    plot_df["花瓣寬度"] = plot_df["petal_width"]
    figp = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="品種",
        symbol="品種",
        color_discrete_map={SPECIES_ZH[s]: PALETTE[s] for s in SPECIES},
        hover_data={
            "sample_id": True,
            "花萼長度": ":.1f",
            "花萼寬度": ":.1f",
            "花瓣長度": ":.1f",
            "花瓣寬度": ":.1f",
            "PC1": ":.2f",
            "PC2": ":.2f",
            "PC3": ":.2f",
            "species": False,
        },
        labels={
            "PC1": f"PC1（{evr[0]:.1%}）",
            "PC2": f"PC2（{evr[1]:.1%}）",
            "PC3": f"PC3（{evr[2]:.1%}）",
            "sample_id": "樣本編號",
        },
        title="Iris PCA 三維特徵分布（可拖曳旋轉、滾輪縮放）",
        opacity=0.88,
    )
    figp.update_traces(marker=dict(size=5, line=dict(width=0.5, color="white")))
    figp.update_layout(
        template="plotly_white",
        width=1000,
        height=760,
        legend_title_text="品種",
        font=dict(family="Microsoft JhengHei, Noto Sans TC, sans-serif", size=13),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title=f"PC1（{evr[0]:.1%}）",
            yaxis_title=f"PC2（{evr[1]:.1%}）",
            zaxis_title=f"PC3（{evr[2]:.1%}）",
            aspectmode="cube",
        ),
    )
    figp.write_html(FIG_DIR / "fig08_pca_3d_interactive.html", include_plotlyjs=True, full_html=True)

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------
    summary = {
        "standardized": True,
        "explained_variance_ratio": [float(v) for v in evr],
        "cumulative_variance": [float(v) for v in cum],
        "n_components_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
        "n_components_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "loadings": loadings.round(4).to_dict(),
        "pc1_interpretation": "花瓣長度、花瓣寬度與花萼長度同向載荷，代表整體花朵尺寸。",
        "pc2_interpretation": "花萼寬度為主，且與花瓣尺寸方向相反，對比花萼寬度與花瓣大小。",
        "pc3_interpretation": "主要區分花萼長度與花瓣寬度的細微差異，解釋量很低。",
    }
    save_json(summary, RESULT_DIR / "pca_summary.json")
    print("\n[完成] 已輸出 PCA 陡坡圖、3D 靜態圖、互動式 HTML 與 PCA 結果表")


if __name__ == "__main__":
    force_utf8_stdout()
    setup_plotting()
    main()
