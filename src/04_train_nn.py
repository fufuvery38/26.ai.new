"""04 - 以類神經網路（多層感知器, MLP）建立 Iris 分類模型。

實驗設計：
    1. 分層抽樣 80/20：120 筆開發集（訓練＋驗證）、30 筆測試集（只在最後評估一次）。
    2. 在開發集上以 5-fold 分層交叉驗證比較 4 種 MLP 架構與 3 種傳統基準模型。
    3. 以交叉驗證的 macro-F1 選出最佳 MLP，於開發集重新訓練（含 early stopping）。
    4. 對測試集評估，並以 bootstrap 計算準確率的 95% 信賴區間。

產出：
    results/metrics.json
    results/cv_results.csv
    results/classification_report.txt
    results/test_predictions.csv
    results/training_history.csv
    results/model_results.md
    results/model_mlp.joblib
    figures/fig09 ~ fig15
"""

from __future__ import annotations

import platform
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from common import (
    FEATURES,
    FEATURES_ZH,
    FIG_DIR,
    PALETTE,
    RESULT_DIR,
    SEED,
    SPECIES,
    SPECIES_ZH,
    force_utf8_stdout,
    load_iris_df,
    save_json,
    setup_plotting,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

COLORS = {
    "MLP-8": "#4c72b0",
    "MLP-16-8": "#dd8452",
    "MLP-32-16": "#55a868",
    "MLP-16-α0.01": "#8172b3",
    "邏輯迴歸": "#937860",
    "SVM (RBF)": "#da8bc3",
    "kNN (k=5)": "#8c8c8c",
}


def build_candidates() -> dict[str, Pipeline]:
    """回傳候選模型（皆含標準化）。"""
    def mlp(hidden, alpha, lr=0.01):
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=lr,
            batch_size=min(32, 120),
            max_iter=5000,
            early_stopping=False,
            random_state=SEED,
        )

    return {
        "MLP-8": Pipeline([("scaler", StandardScaler()), ("clf", mlp((8,), 1e-4))]),
        "MLP-16-8": Pipeline([("scaler", StandardScaler()), ("clf", mlp((16, 8), 1e-4))]),
        "MLP-32-16": Pipeline([("scaler", StandardScaler()), ("clf", mlp((32, 16), 1e-3, lr=0.005))]),
        "MLP-16-α0.01": Pipeline([("scaler", StandardScaler()), ("clf", mlp((16,), 1e-2))]),
        "邏輯迴歸": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED))]),
        "kNN (k=5)": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }


def cv_compare(models: dict[str, Pipeline], X, y) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro", "log_loss": "neg_log_loss"}
    rows = []
    for name, model in models.items():
        t0 = time.perf_counter()
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        rows.append(
            {
                "model": name,
                "acc_mean": res["test_accuracy"].mean(),
                "acc_std": res["test_accuracy"].std(ddof=1),
                "f1_mean": res["test_macro_f1"].mean(),
                "f1_std": res["test_macro_f1"].std(ddof=1),
                "logloss_mean": -res["test_log_loss"].mean(),
                "logloss_std": (-res["test_log_loss"]).std(ddof=1),
                "fit_seconds": time.perf_counter() - t0,
                "fold_accuracies": res["test_accuracy"].tolist(),
                "fold_f1": res["test_macro_f1"].tolist(),
            }
        )
        print(f"  {name:<14} acc={rows[-1]['acc_mean']:.4f}±{rows[-1]['acc_std']:.4f}  "
              f"macro-F1={rows[-1]['f1_mean']:.4f}  log-loss={rows[-1]['logloss_mean']:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    t_start = time.perf_counter()
    df = load_iris_df()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["species"].cat.codes.to_numpy()
    species = df["species"].astype(str).to_numpy()

    print("=" * 70)
    print("步驟 4/4：類神經網路分類模型")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 資料切分
    # ------------------------------------------------------------------
    X_dev, X_test, y_dev, y_test, sp_dev, sp_test = train_test_split(
        X, y, species, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"[切分] 開發集 {len(X_dev)} 筆（訓練＋驗證），測試集 {len(X_test)} 筆（僅最終評估）")
    print(f"       測試集各類別：{ {SPECIES[i]: int((y_test == i).sum()) for i in range(3)} }")

    # ------------------------------------------------------------------
    # 交叉驗證比較
    # ------------------------------------------------------------------
    print("\n[5-fold 分層交叉驗證]（開發集 120 筆）")
    models = build_candidates()
    cv_df = cv_compare(models, X_dev, y_dev)
    cv_df.to_csv(RESULT_DIR / "cv_results.csv", index=False, encoding="utf-8-sig")

    mlp_df = cv_df[cv_df["model"].str.startswith("MLP")].sort_values(
        ["f1_mean", "acc_mean"], ascending=False
    )
    best_name = str(mlp_df.iloc[0]["model"])
    print(f"\n[選擇] 依 macro-F1 選出最佳架構：{best_name}")

    # ------------------------------------------------------------------
    # 最終模型（開發集完整訓練，early stopping）
    # ------------------------------------------------------------------
    best_model = clone(models[best_name])
    best_model.set_params(
        clf__max_iter=3000,
        clf__early_stopping=True,
        clf__validation_fraction=0.15,
        clf__n_iter_no_change=50,
        clf__tol=1e-5,
    )
    best_model.fit(X_dev, y_dev)
    clf = best_model.named_steps["clf"]
    print(f"[訓練] 實際訓練 epoch 數={clf.n_iter_}，early stopping 監控驗證集大小={int(len(X_dev) * 0.15)}")

    history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(clf.loss_curve_) + 1),
            "train_loss": clf.loss_curve_,
            "val_accuracy": clf.validation_scores_,
        }
    )
    assert len(clf.loss_curve_) == len(clf.validation_scores_), "訓練與驗證曲線長度不一致"
    history.to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 測試集評估
    # ------------------------------------------------------------------
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    labels = np.arange(3)

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = sklearn.metrics.f1_score(y_test, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_test, y_pred, average="weighted")
    ll = log_loss(y_test, y_proba, labels=labels)
    auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=labels)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=SPECIES, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred, labels=labels, target_names=SPECIES, digits=4, zero_division=0, output_dict=True
    )

    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(y_test), len(y_test))
        boot.append(accuracy_score(y_test[idx], y_pred[idx]))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    print("\n[測試集結果]")
    print(f"  準確率           {acc:.4f}  (95% CI {ci_low:.4f}–{ci_high:.4f})")
    print(f"  平衡準確率       {bacc:.4f}")
    print(f"  Macro F1         {f1_macro:.4f}")
    print(f"  Log loss         {ll:.4f}")
    print(f"  ROC-AUC (OvR)    {auc_macro:.4f}")
    print(f"  Cohen's kappa    {kappa:.4f}")
    print(f"  MCC              {mcc:.4f}")
    print("\n" + report)
    (RESULT_DIR / "classification_report.txt").write_text(
        f"模型：{best_name}\n測試集 n={len(y_test)}\n\n{report}\n混淆矩陣（列=實際, 欄=預測）:\n{cm}\n"
        f"Accuracy 95% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]\n",
        encoding="utf-8",
    )

    pred_df = pd.DataFrame(
        {
            "sample_id": np.arange(1, len(y_test) + 1),
            "true_label": [SPECIES[i] for i in y_test],
            "pred_label": [SPECIES[i] for i in y_pred],
            "correct": y_test == y_pred,
            "prob_setosa": y_proba[:, 0],
            "prob_versicolor": y_proba[:, 1],
            "prob_virginica": y_proba[:, 2],
            "confidence": y_proba.max(axis=1),
        }
    )
    pred_df.to_csv(RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 基準模型在測試集上的描述性結果（未參與模型選擇）
    baseline_test = {}
    for name, model in models.items():
        if name.startswith("MLP"):
            continue
        model.fit(X_dev, y_dev)
        p = model.predict(X_test)
        baseline_test[name] = {
            "accuracy": float(accuracy_score(y_test, p)),
            "macro_f1": float(sklearn.metrics.f1_score(y_test, p, average="macro")),
        }

    # ------------------------------------------------------------------
    # 圖 9：訓練曲線
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].plot(history["epoch"], history["train_loss"], color="#4c72b0", lw=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("訓練損失（交叉熵）")
    axes[0].set_title("(a) 訓練損失")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))

    axes[1].plot(history["epoch"], history["val_accuracy"], color="#c44e52", lw=1.8)
    best_ep = int(history.loc[history["val_accuracy"].idxmax(), "epoch"])
    best_val = float(history["val_accuracy"].max())
    axes[1].axvline(best_ep, ls="--", color="gray", lw=1)
    axes[1].scatter([best_ep], [best_val], color="#c44e52", zorder=3)
    axes[1].annotate(
        f"最佳 epoch={best_ep}\n驗證準確率={best_val:.3f}",
        (best_ep, best_val),
        textcoords="offset points",
        xytext=(12, -28),
        fontsize=9,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("驗證準確率")
    axes[1].set_ylim(0.75, 1.02)
    axes[1].set_title("(b) 內部驗證準確率")
    fig.suptitle(f"圖 9　{best_name} 訓練歷程（early stopping，共 {clf.n_iter_} epochs）", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_training_curve.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 10：混淆矩陣
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.55 else "#222222",
                fontsize=11,
            )
    ax.set_xticks(range(3), [f"{SPECIES_ZH[s]}\n{s}" for s in SPECIES], fontsize=9)
    ax.set_yticks(range(3), [f"{SPECIES_ZH[s]}\n{s}" for s in SPECIES], fontsize=9)
    ax.set_xlabel("預測類別")
    ax.set_ylabel("實際類別")
    ax.set_title(f"圖 10　測試集混淆矩陣（n={len(y_test)}，準確率 {acc:.1%}）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="列正規化比例")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_confusion_matrix.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 11：各類別指標
    # ------------------------------------------------------------------
    metrics = ["precision", "recall", "f1-score"]
    zh = {"precision": "精確率", "recall": "召回率", "f1-score": "F1 分數"}
    x = np.arange(3)
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for k, m in enumerate(metrics):
        vals = [report_dict[s][m] for s in SPECIES]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=zh[m])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"{SPECIES_ZH[s]}\n(支持數 {int(report_dict[s]['support'])})" for s in SPECIES], fontsize=9)
    ax.set_ylim(0.7, 1.06)
    ax.set_ylabel("分數")
    ax.set_title(f"圖 11　測試集各品種分類指標（macro-F1 = {f1_macro:.3f}）")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_per_class_metrics.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 12：ROC 曲線
    # ------------------------------------------------------------------
    y_bin = label_binarize(y_test, classes=labels)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for i, s in enumerate(SPECIES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=1.8, color=PALETTE[s], label=f"{SPECIES_ZH[s]}（AUC={auc_i:.3f}）")
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    auc_micro = roc_auc_score(y_bin, y_proba, average="micro")
    ax.plot(fpr_micro, tpr_micro, "k--", lw=1.4, label=f"Micro-average（AUC={auc_micro:.3f}）")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":")
    ax.set_xlabel("偽陽性率 (FPR)")
    ax.set_ylabel("真陽性率 (TPR)")
    ax.set_title("圖 12　測試集 ROC 曲線（One-vs-Rest）")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_roc_curves.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 13：PCA 2D 決策邊界
    # ------------------------------------------------------------------
    pipe2d = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=SEED)),
            ("clf", clone(best_model.named_steps["clf"])),
        ]
    )
    pipe2d.named_steps["clf"].set_params(early_stopping=False, max_iter=3000)
    pipe2d.fit(X_dev, y_dev)
    acc2d = pipe2d.score(X_test, y_test)
    Z_dev = pipe2d[:-1].transform(X_dev)
    Z_test = pipe2d[:-1].transform(X_test)

    xx, yy = np.meshgrid(
        np.linspace(Z_dev[:, 0].min() - 0.8, Z_dev[:, 0].max() + 0.8, 400),
        np.linspace(Z_dev[:, 1].min() - 0.8, Z_dev[:, 1].max() + 0.8, 400),
    )
    grid = pipe2d.named_steps["clf"].predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap_bg = ListedColormap([PALETTE[s] for s in SPECIES])

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.contourf(xx, yy, grid, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_bg, alpha=0.13)
    ax.contour(xx, yy, grid, levels=[0.5, 1.5], colors="gray", linewidths=0.8)
    for s_i, s in enumerate(SPECIES):
        m_dev = y_dev == s_i
        m_test = y_test == s_i
        ax.scatter(Z_dev[m_dev, 0], Z_dev[m_dev, 1], s=22, color=PALETTE[s], alpha=0.35, edgecolor="none")
        ax.scatter(
            Z_test[m_test, 0],
            Z_test[m_test, 1],
            s=70,
            color=PALETTE[s],
            edgecolor="black",
            linewidth=0.7,
            marker="o",
            label=SPECIES_ZH[s],
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"圖 13　PCA 前兩主成分上的分類邊界（測試集準確率 {acc2d:.1%}）")
    ax.legend(title="測試樣本", loc="upper right")
    ax.text(
        0.01,
        0.01,
        "背景為模型預測區域；淡色點為開發集，粗邊框點為測試集",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig13_pca_decision_boundary.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 14：模型比較
    # ------------------------------------------------------------------
    cv_sorted = cv_df.sort_values("acc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [COLORS.get(m, "#888888") for m in cv_sorted["model"]]
    y_pos = np.arange(len(cv_sorted))
    ax.barh(y_pos, cv_sorted["acc_mean"], xerr=cv_sorted["acc_std"], color=colors, alpha=0.85, capsize=4)
    for i, (_, row) in enumerate(cv_sorted.iterrows()):
        ax.text(row["acc_mean"] - 0.006, i, f"{row['acc_mean']:.3f}", va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(y_pos, cv_sorted["model"])
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("5-fold 交叉驗證準確率（平均 ± 標準差）")
    ax.set_title("圖 14　候選模型比較（開發集）")
    ax.axvline(0.95, color="gray", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig14_model_comparison.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 15：學習曲線
    # ------------------------------------------------------------------
    # 學習曲線：為觀察「樣本數」效果，關閉 early stopping（固定訓練上限），
    # 避免小訓練集因內部驗證過早停止而低估學習能力。
    lc_model = clone(best_model)
    lc_model.set_params(clf__early_stopping=False, clf__max_iter=2000)
    train_sizes, train_scores, val_scores = learning_curve(
        lc_model,
        X_dev,
        y_dev,
        train_sizes=np.linspace(0.2, 1.0, 6),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="accuracy",
        n_jobs=-1,
        shuffle=True,
        random_state=SEED,
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#4c72b0", label="訓練分數")
    ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1), train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color="#4c72b0")
    ax.plot(train_sizes, val_scores.mean(axis=1), "s-", color="#c44e52", label="交叉驗證分數")
    ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1), val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color="#c44e52")
    ax.set_xlabel("訓練樣本數")
    ax.set_ylabel("準確率")
    ax.set_ylim(0.75, 1.02)
    ax.set_title("圖 15　學習曲線：樣本數與模型表現（停用 early stopping）")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig15_learning_curve.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 16：排列特徵重要性
    # ------------------------------------------------------------------
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=50, random_state=SEED, scoring="accuracy")
    perm_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(
        [FEATURES_ZH[f] for f in perm_df["feature"]],
        perm_df["importance_mean"],
        xerr=perm_df["importance_std"],
        color="#55a868",
        capsize=4,
    )
    ax.set_xlabel("準確率下降幅度")
    ax.set_title("圖 16　排列特徵重要性（測試集，50 次重抽）")
    ax.axvline(0, color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig16_permutation_importance.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 圖 17：多個隨機種子的穩定性
    # ------------------------------------------------------------------
    seeds = [42, 7, 2024, 1, 13, 99, 123, 777, 314, 2025]
    stab_rows = []
    for s in seeds:
        m = clone(best_model)
        m.set_params(clf__random_state=int(s))
        m.fit(X_dev, y_dev)
        p = m.predict(X_test)
        stab_rows.append(
            {
                "seed": int(s),
                "accuracy": float(accuracy_score(y_test, p)),
                "macro_f1": float(f1_score(y_test, p, average="macro")),
                "epochs": int(m.named_steps["clf"].n_iter_),
            }
        )
    stab_df = pd.DataFrame(stab_rows)
    stab_mean, stab_std = stab_df["accuracy"].mean(), stab_df["accuracy"].std(ddof=1)
    print(f"\n[穩定性] 10 個隨機種子：測試準確率 {stab_mean:.4f} ± {stab_std:.4f}"
          f"（min={stab_df['accuracy'].min():.3f}, max={stab_df['accuracy'].max():.3f}）")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.scatter(stab_df["seed"].astype(str), stab_df["accuracy"], s=70, color="#4c72b0", zorder=3)
    ax.axhline(stab_mean, color="#c44e52", lw=1.6, label=f"平均 {stab_mean:.3f}")
    ax.fill_between(
        [-0.5, len(seeds) - 0.5],
        stab_mean - stab_std,
        stab_mean + stab_std,
        color="#c44e52",
        alpha=0.12,
        label=f"±1 標準差（{stab_std:.3f}）",
    )
    ax.set_xlim(-0.5, len(seeds) - 0.5)
    ax.set_ylim(0.75, 1.02)
    ax.set_xlabel("隨機種子（資料切分固定為 seed=42）")
    ax.set_ylabel("測試集準確率")
    ax.set_title(f"圖 17　初始化穩定性：10 次重複訓練（平均 {stab_mean:.1%}）")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig17_seed_stability.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 儲存模型
    # ------------------------------------------------------------------
    model_path = RESULT_DIR / "model_mlp.joblib"
    joblib.dump({"model": best_model, "features": FEATURES, "classes": SPECIES, "model_name": best_name}, model_path)

    # ------------------------------------------------------------------
    # 彙整結果
    # ------------------------------------------------------------------
    metrics_json = {
        "experiment": {
            "seed": SEED,
            "test_size": 0.2,
            "n_dev": int(len(X_dev)),
            "n_test": int(len(X_test)),
            "cv_folds": 5,
            "selection_metric": "macro_f1",
            "model": best_name,
            "hidden_layer_sizes": list(best_model.named_steps["clf"].hidden_layer_sizes),
            "alpha": best_model.named_steps["clf"].alpha,
            "learning_rate_init": best_model.named_steps["clf"].learning_rate_init,
            "epochs_trained": int(clf.n_iter_),
            "elapsed_seconds": None,
        },
        "cv_results": cv_df.drop(columns=["fold_accuracies", "fold_f1"]).round(6).to_dict(orient="records"),
        "cv_fold_accuracies": {r["model"]: [float(v) for v in r["fold_accuracies"]] for _, r in cv_df.iterrows()},
        "test": {
            "accuracy": float(acc),
            "accuracy_ci95": [float(ci_low), float(ci_high)],
            "balanced_accuracy": float(bacc),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted),
            "log_loss": float(ll),
            "roc_auc_ovr_macro": float(auc_macro),
            "roc_auc_micro": float(auc_micro),
            "cohen_kappa": float(kappa),
            "mcc": float(mcc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
            "n_correct": int((y_test == y_pred).sum()),
            "n_errors": int((y_test != y_pred).sum()),
        },
        "training": {
            "epochs": int(clf.n_iter_),
            "best_val_accuracy": best_val,
            "best_epoch": best_ep,
            "final_train_loss": float(clf.loss_curve_[-1]),
            "initial_train_loss": float(clf.loss_curve_[0]),
            "n_iter_no_change": int(clf.n_iter_no_change),
        },
        "pca_visualization": {
            "accuracy_2d": float(acc2d),
            "pc1_pc2_explained": float(
                PCA(n_components=2, random_state=SEED).fit(StandardScaler().fit_transform(X_dev)).explained_variance_ratio_.sum()
            ),
        },
        "permutation_importance": perm_df.round(6).to_dict(orient="records"),
        "seed_stability": {
            "seeds": seeds,
            "runs": stab_df.round(6).to_dict(orient="records"),
            "accuracy_mean": float(stab_mean),
            "accuracy_std": float(stab_std),
            "accuracy_min": float(stab_df["accuracy"].min()),
            "accuracy_max": float(stab_df["accuracy"].max()),
        },
        "learning_curve": {
            "train_sizes": train_sizes.tolist(),
            "train_mean": train_scores.mean(axis=1).tolist(),
            "train_std": train_scores.std(axis=1).tolist(),
            "val_mean": val_scores.mean(axis=1).tolist(),
            "val_std": val_scores.std(axis=1).tolist(),
        },
        "baseline_test_reference": baseline_test,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "platform": platform.platform(),
        },
    }
    metrics_json["experiment"]["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
    save_json(metrics_json, RESULT_DIR / "metrics.json")

    # 給小論文引用的 Markdown 表格
    md = "# 模型實驗結果\n\n"
    md += f"- 模型：**{best_name}**（hidden={clf.hidden_layer_sizes}, alpha={clf.alpha}, lr={clf.learning_rate_init}）\n"
    md += f"- 切分：開發集 {len(X_dev)} 筆 / 測試集 {len(X_test)} 筆（分層抽樣，seed={SEED}）\n"
    md += f"- 訓練：{int(clf.n_iter_)} epochs（early stopping, patience=50）\n\n"
    md += "## 交叉驗證（5-fold, 開發集）\n\n"
    md += "| 模型 | 準確率 | Macro-F1 | Log loss |\n| --- | ---: | ---: | ---: |\n"
    for _, r in cv_df.sort_values("f1_mean", ascending=False).iterrows():
        md += f"| {r['model']} | {r['acc_mean']:.4f} ± {r['acc_std']:.4f} | {r['f1_mean']:.4f} ± {r['f1_std']:.4f} | {r['logloss_mean']:.4f} |\n"
    md += "\n## 測試集\n\n"
    md += "| 指標 | 數值 |\n| --- | ---: |\n"
    md += f"| 準確率 | {acc:.4f}（95% CI {ci_low:.4f}–{ci_high:.4f}） |\n"
    md += f"| 平衡準確率 | {bacc:.4f} |\n"
    md += f"| Macro-F1 | {f1_macro:.4f} |\n"
    md += f"| 加權 F1 | {f1_weighted:.4f} |\n"
    md += f"| Log loss | {ll:.4f} |\n"
    md += f"| ROC-AUC (OvR macro) | {auc_macro:.4f} |\n"
    md += f"| Cohen's κ | {kappa:.4f} |\n"
    md += f"| MCC | {mcc:.4f} |\n\n"
    md += f"- 重複訓練穩定性（10 個種子，固定切分）：{stab_mean:.4f} ± {stab_std:.4f}（min {stab_df['accuracy'].min():.3f}, max {stab_df['accuracy'].max():.3f}）\n\n"
    md += "```\n" + report + "```\n"
    (RESULT_DIR / "model_results.md").write_text(md, encoding="utf-8")

    print(f"\n[完成] 已輸出模型、預測結果、metrics.json 與訓練／評估圖表（圖 9–17，耗時 {metrics_json['experiment']['elapsed_seconds']} 秒）")
    print(f"[模型] 已儲存至 results/model_mlp.joblib（含標準化與分類器）")


if __name__ == "__main__":
    force_utf8_stdout()
    setup_plotting()
    main()
