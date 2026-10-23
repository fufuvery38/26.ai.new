# AI4I 2020 預測性維護：資料稽核、不平衡分類與故障模式可學習性

以 UCI 的 AI4I 2020 Predictive Maintenance 資料集完成一套可重現的端到端機器學習專題。主軸不是追求最高的 accuracy，而是拆解一個更誠實的問題：**在由規則生成的合成資料上，模型到底學到了什麼？預測力的上限由哪些故障模式決定？**

[![事件單](https://img.shields.io/badge/Issue-%234-1f4e79)](https://github.com/fufuvery38/26.ai.new/issues/4)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E)](https://scikit-learn.org/)

## 目前進度

| 階段 | 內容 | 狀態 |
| --- | --- | --- |
| 0 | 資料取得、雜湊驗證、欄位整理與稽核 | **完成** |
| 1 | 探索式資料分析與視覺化 | **完成** |
| 2 | 特徵工程與洩漏防護 | 待實作 |
| 3 | 不平衡分類模型與規則 baseline | 待實作 |
| 4 | 評估、門檻與不確定性 | 待實作 |
| 5 | 五種故障模式的可學習性分解 | 待實作 |
| 6 | 解釋性與穩健性 | 待實作 |
| 7 | 小論文與一鍵重現 | 待實作 |

工作項目與驗收標準見 [`ISSUE.md`](ISSUE.md)（與 [GitHub Issue #4](https://github.com/fufuvery38/26.ai.new/issues/4) 同步）。

## 快速開始

```bash
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` 會依檔名順序執行 `src/NN_*.py`，目前包含階段 0–1 的腳本。單獨執行：

```bash
python src/01_get_data.py          # 下載、驗證、整理、稽核
python src/01_get_data.py --force  # 強制重新下載原始檔
```

## 專案結構

```text
ai4i2020/
├── ISSUE.md              # 事件單
├── README.md
├── run_all.py            # 一鍵重現（自動探索 src/NN_*.py）
├── requirements.txt
├── data/
│   ├── ai4i2020.zip      # UCI 原始壓縮檔
│   ├── ai4i2020_raw.csv  # 原始 CSV（未修改）
│   ├── ai4i2020.csv      # 整理後資料（snake_case）
│   ├── dataset_summary.json
│   └── README.md         # 欄位字典、雜湊與已知問題
├── src/
│   ├── common.py         # 路徑、常數、繪圖風格、衍生特徵
│   ├── 01_get_data.py    # 階段 0
│   └── 02_eda.py         # 階段 1
├── figures/              # 11 張 EDA 圖表
├── results/
│   ├── data_audit.json
│   ├── data_audit.md
│   ├── eda_summary.json
│   └── eda_summary.md
└── paper/                # 後續階段產生
```

## 資料與階段 0 發現

| 項目 | 結果 |
| --- | --- |
| 規模 | 10,000 筆 × 14 欄；缺失值 0、完全重複列 0 |
| 目標 | `machine_failure` 339 筆（**3.39%**） |
| 模式 | TWF 46、HDF 115、PWF 95、OSF 98、RNF 19 |
| 機型 | L 6,000／M 2,997／H 1,003 |

- **PWF、HDF 可 100% 由物理規則還原**：功率落在 3,500–9,000 W 之外，以及溫差 < 8.6 K 且轉速 < 1,380 rpm。這代表整體分數會被「送分題」拉高，必須拆模式解讀。
- **OSF、TWF 僅部分可學習**：兩者集中在高磨耗區，但無法用單一磨耗窗完全決定。
- **RNF 本質隨機**：適合作為「不可學習」的對照。
- **27 筆標籤不一致**：18 筆 `rnf=1` 但 target=0、9 筆 target=1 卻無任何模式旗標。主分析保留原標籤，另做敏感度分析。

欄位字典、SHA-256 與完整異常清單見 [`data/README.md`](data/README.md)；統計報告見 [`results/data_audit.md`](results/data_audit.md)。

## 階段 1：EDA 產出

11 張圖與完整摘要見 [`results/eda_summary.md`](results/eda_summary.md)。重點：

- 五個感測特徵的邊際分布高度重疊，單一特徵不足以分離故障。
- 全猜多數類即可達 **96.61% accuracy**，確認 accuracy 不能作為主要指標。
- PWF、HDF 的物理邊界可完全還原；OSF／TWF 重疊且含隨機成分；RNF 無一致方向。
- 工具磨耗在 UDI 順序上有 **119 次重置、120 個週期**，平均 83.3 筆一個週期。
- 邊際上 L 故障率高於 H，但 200–240 min 分桶內仍以 L 較高，顯示操作條件與磨耗混雜，不能解讀為機型體質。

| 圖 | 內容 |
| --- | --- |
| `fig01`–`fig02` | 類別不平衡、模式重疊 |
| `fig03`–`fig05` | 特徵分布、箱型圖、模式剖面 |
| `fig06` | Pearson／Spearman 相關性 |
| `fig07`–`fig09` | PWF、HDF、OSF／TWF 的物理邊界 |
| `fig10` | 機型 × 磨耗分桶的故障率 |
| `fig11` | 磨耗累積與重置序列 |

## 重現性

- 原始 zip 與 CSV 皆驗證 SHA-256；不符時腳本會直接失敗，不默默沿用。
- 固定 `seed = 42`（後續建模階段）。
- `dataset_summary.json` 記錄來源、授權、DOI、檔案雜湊與筆數。
- `01_get_data.py` 可重複執行且輸出位元組完全相同。

## 環境

Python 3.14.4、NumPy 2.5.4、pandas 3.1.0、scikit-learn 1.9.1、Matplotlib 3.11.2、seaborn 0.13.2、Plotly 7.1.0。

## 資料來源與授權

資料集：AI4I 2020 Predictive Maintenance Dataset（UCI #601），作者 Stephan Matzka，授權 CC BY 4.0。

> Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

程式碼與文件可自由用於教學與研究。
