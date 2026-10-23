# Iris 資料集探索、PCA 3D 視覺化與類神經網路分類

以 Fisher（1936）的經典 Iris 資料集完成一套可重現的端到端機器學習專題：資料取得與勘誤、探索式分析、PCA 三維視覺化、多層感知器（MLP）分類，以及依實際結果撰寫的小論文。

[![事件單](https://img.shields.io/badge/Issue-%233-1f4e79)](https://github.com/fufuvery38/26.ai.new/issues/3)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E)](https://scikit-learn.org/)

## 重點結果

| 項目 | 結果 |
| --- | --- |
| 資料 | 150 筆 × 4 特徵 × 3 品種（各 50 筆），無缺失值 |
| 資料勘誤 | 發現並更正 UCI 原始檔第 35、38 筆共 3 個數值 |
| PCA | PC1–PC3 累積解釋 **99.48%** 變異（72.96% / 22.85% / 3.67%） |
| 模型選擇 | 5-fold CV 比較 4 種 MLP + 3 種基準模型，選出 **4-8-3 MLP** |
| 測試集 | 準確率 **90.0%**（95% CI 80.0–100.0%）、Macro-F1 **0.900**、ROC-AUC **0.993** |
| 穩定性 | 10 個隨機種子：**88.3% ± 3.9%** |
| 錯誤分析 | 3 個錯誤全在 versicolor / virginica 交界，平均信心 0.62（正確樣本 0.92） |

> 結論：setosa 完全分離；versicolor 與 virginica 在花瓣尺寸上連續重疊，是誤判來源。在此小型飽和資料上，MLP 與 SVM 表現相當，未展現明顯優勢。

## 快速開始

```bash
python -m pip install -r requirements.txt
python run_all.py
```

依序執行資料下載、EDA、PCA、模型訓練與論文建置。所有隨機程序固定 `seed = 42`。若 UCI 無法連線，資料步驟會退回 scikit-learn 內建副本並在輸出中提示。

單獨執行各步驟：

```bash
python src/01_get_data.py   # 下載、勘誤、驗證、存檔
python src/02_eda.py       # 圖 1–5、EDA 摘要
python src/03_pca3d.py     # 圖 6–8、PCA 負荷量
python src/04_train_nn.py  # 模型比較、訓練、評估、圖 9–17
python src/05_build_paper.py  # Markdown → 可攜式 HTML
```

## 專案結構

```text
.
├── ISSUE.md                     # 事件單（與 GitHub Issue #3 同步）
├── README.md
├── run_all.py                   # 一鍵重現
├── requirements.txt
├── data/
│   ├── iris_raw.data            # UCI 原始檔（未修改）
│   ├── iris_bezdek.data         # UCI 官方修正版
│   ├── iris.csv                 # 整理後資料（已勘誤）
│   ├── iris.names               # UCI 欄位說明
│   ├── dataset_summary.json     # SHA-256、筆數、更正紀錄
│   └── README.md                # 資料說明
├── src/
│   ├── common.py                # 共用設定
│   ├── 01_get_data.py
│   ├── 02_eda.py
│   ├── 03_pca3d.py
│   ├── 04_train_nn.py
│   └── 05_build_paper.py
├── figures/                     # 16 張 PNG + 1 份互動式 3D HTML
├── results/                     # metrics.json、預測、模型檔
└── paper/
    ├── iris_nn_paper.md         # 小論文（Markdown）
    ├── iris_nn_paper.html       # 自包含 HTML（圖片內嵌）
    └── iris_nn_paper.pdf        # 列印版
```

## 資料說明

| 欄位 | 中文 | 單位 |
| --- | --- | --- |
| `sepal_length` | 花萼長度 | cm |
| `sepal_width` | 花萼寬度 | cm |
| `petal_length` | 花瓣長度 | cm |
| `petal_width` | 花瓣寬度 | cm |
| `species` | 品種（setosa / versicolor / virginica） | — |

資料檢查發現 UCI `iris.data` 與官方修正版有 2 筆、3 個數值不一致；本專案以 `bezdekIris.data` 為準更正，原始檔仍完整保留。更正後與 scikit-learn 內建版本逐值一致，詳細 SHA-256 與差異表見 [`data/README.md`](data/README.md)。

## 實驗設計

- **切分**：分層 80/20（開發集 120、測試集 30，每類 10），測試集僅評估一次。
- **模型選擇**：開發集上 5-fold 分層交叉驗證，以 macro-F1 選模型。
- **候選模型**：MLP-8、MLP-16-8、MLP-32-16、MLP-16（α=0.01）；邏輯迴歸、RBF-SVM、kNN。
- **最終訓練**：Adam、ReLU、L2（α=1e-4）、early stopping（15% 內部驗證、patience=50）。
- **不確定性**：2,000 次 bootstrap、10 個隨機種子、學習曲線、排列特徵重要性。

完整參數與結果見 [`results/metrics.json`](results/metrics.json) 與 [`results/model_results.md`](results/model_results.md)。

## 圖表

| 圖 | 檔案 |
| --- | --- |
| 類別分布 | `figures/fig01_class_distribution.png` |
| 特徵分布／箱型圖 | `fig02_feature_histograms.png`、`fig03_feature_boxplots.png` |
| 相關矩陣／成對散點 | `fig04_correlation_heatmap.png`、`fig05_pairplot.png` |
| PCA 陡坡圖／3D 分布 | `fig06_pca_scree.png`、`fig07_pca_3d.png` |
| 互動式 3D | `figures/fig08_pca_3d_interactive.html` |
| 訓練曲線／混淆矩陣 | `fig09_training_curve.png`、`fig10_confusion_matrix.png` |
| 類別指標／ROC | `fig11_per_class_metrics.png`、`fig12_roc_curves.png` |
| 決策邊界／模型比較 | `fig13_pca_decision_boundary.png`、`fig14_model_comparison.png` |
| 學習曲線／特徵重要性 | `fig15_learning_curve.png`、`fig16_permutation_importance.png` |
| 種子穩定性 | `fig17_seed_stability.png` |

## 小論文

[`paper/iris_nn_paper.md`](paper/iris_nn_paper.md) 包含摘要、緒論、文獻探討、研究方法、實驗結果、討論、結論、參考文獻與附錄。HTML 版將所有圖片以 base64 內嵌，可離線閱讀；PDF 版為 A4 列印格式。

## 限制

- 測試集僅 30 筆，單一準確率的抽樣誤差大（95% CI 寬達 20 個百分點），應搭配多種子結果解讀。
- 僅使用單一 80/20 切分；未採用 nested cross-validation。
- 超參數網格較小，未對 MLP 與基準模型做對等調參。
- PCA 3D 視覺化使用全部資料（非監督），但不參與模型選擇；模型流程中的轉換均在折內擬合。
- Iris 為高度飽和的小型資料集，模型間 1–2 個百分點的差距不宜過度解讀。

## 環境

Python 3.14.4、NumPy 2.5.4、pandas 3.1.0、scikit-learn 1.9.1、Matplotlib 3.11.2、seaborn 0.13.2、Plotly 7.1.0。

## 授權與致謝

資料來源：UCI Machine Learning Repository, Iris Data Set（公開學術資料）。程式碼與文件可自由用於教學與研究。
