# AI4I 2020 預測性維護專題：資料稽核、不平衡分類、故障模式可學習性與小論文

| 項目 | 內容 |
| --- | --- |
| 事件類型 | 資料分析／機器學習專題 |
| 優先級 | 中 |
| 狀態 | **Open（規劃完成，待實作）** |
| 負責人 | @fufuvery38 |
| 建立日期 | 2026-10-23 |
| 資料集 | AI4I 2020 Predictive Maintenance Dataset（UCI #601，10,000 筆 × 14 欄） |
| 專案路徑 | `ai4i2020/` |
| 相關分支 | `main` |
| GitHub Issue | [#4](https://github.com/fufuvery38/26.ai.new/issues/4) |

## 背景與目的

AI4I 2020 是 UCI 的預測性維護資料集，目標欄位為 `Machine failure`，正例僅 339 筆（3.39%）。資料由 Matzka（2020）以理論模型合成，另附五個故障模式旗標：TWF（工具磨耗）、HDF（散熱失效）、PWF（功率失效）、OSF（過度應變）、RNF（隨機失效）。

本專題不以「再刷一次 accuracy」為目的，而是回答一個更誠實的問題：**在這種由規則生成的合成資料上，機器學習模型到底學到了什麼？預測力的上限由哪些故障模式決定？**

具體目的：

1. 建立可重現的端到端流程：資料取得與稽核 → EDA → 特徵工程 → 不平衡分類 → 評估 → 解釋 → 小論文。
2. 拆解整體效能，量化「可規則化」、「部分可學習」與「本質隨機」三類故障模式各自的預測上限。
3. 誠實界定結論邊界：合成資料上的高分不得直接外推為真實產線的維護效益。

## 研究問題

- **RQ1**：在 3.39% 正例、僅 339 個故障樣本的限制下，哪一種模型與決策門檻能在可接受的誤報率下抓到最多故障？
- **RQ2**：整體預測力有多少來自 HDF／PWF 這類可完全規則化的模式，多少來自 OSF／TWF 的部分可學習性，又有多少被 RNF 的隨機性鎖死？
- **RQ3**：模型是否會自動回收文獻中的物理規則（例如功率區間、溫差與轉速門檻）？與硬編碼規則相比，機器學習是否帶來額外增益？
- **RQ4**：在這份合成資料上得到的效能與特徵重要性，能否類推到真實產線？有哪些效度威脅必須明說？

## 資料現況（2026-10-23 實測）

| 項目 | 實測結果 |
| --- | --- |
| 規模 | 10,000 筆 × 14 欄；缺失值 0、完全重複列 0 |
| 目標 | `Machine failure` = 339 筆（3.39%），其餘 9,661 筆 |
| 模式旗標 | TWF 46、HDF 115、PWF 95、OSF 98、RNF 19 |
| 重疊情形 | 324 筆單一模式、23 筆雙模式、1 筆三模式 |
| 機型 | L 6,000／M 2,997／H 1,003；故障率 3.92%／2.77%／2.09% |
| 數值範圍 | 空氣溫 295.3–304.5 K、製程溫 305.7–313.8 K、轉速 1,168–2,886 rpm、扭矩 3.8–76.6 Nm、磨耗 0–253 min |
| 識別欄 | `UDI` 與 `Product ID` 皆唯一；`Product ID` 首碼等於 `Type` |
| 檔案 | `ai4i2020.csv`，SHA-256 `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |

已驗證的物理規則：

- **PWF 可完全規則化**：功率 = 扭矩 × 轉速 × 2π/60；落在 3,500–9,000 W 之外者恰好 95 筆，且全部 PWF=1，零誤判。
- **HDF 可完全規則化**：製程溫 − 空氣溫 < 8.6 K 且轉速 < 1,380 rpm，恰好命中 115 筆、零誤判。
- **OSF／TWF 僅部分可學習**：磨耗 200–240 min 是必要條件，但 TWF 在該區間的 790 筆中僅 46 筆發生，具隨機成分；OSF 亦無法由磨耗與扭矩完全決定。
- **RNF 本質隨機**：19 筆，無可用特徵可區分。

資料品質問題（必須在 EDA 揭露並做敏感度分析）：

- 18 筆 `RNF=1` 但 `Machine failure=0`。
- 9 筆 `Machine failure=1` 卻沒有任何模式旗標。
- 其餘四種模式旗標皆為 `Machine failure` 的子集。
- 上述 27 筆列為疑似標籤不一致；主分析保留原標籤，另做「排除後重跑」的敏感度對照。

## 工作項目與驗收標準

### 0. 資料取得與稽核

- [ ] 從 UCI 官方來源下載 zip，原始檔完整保留（不修改）
- [ ] 記錄來源、DOI、授權（CC BY 4.0）、SHA-256、筆數與欄位字典
- [ ] 檢查缺失值、重複列、數值範圍、單位與物理一致性（製程溫 > 空氣溫等）
- [ ] 稽核 `Machine failure` 與五個模式旗標的一致性，列出 27 筆異常並決定處理規則
- [ ] 輸出整理後 `data/ai4i2020.csv` 與 `data/dataset_summary.json`

### 1. 探索式資料分析（EDA）

- [ ] 呈現類別不平衡與各故障模式占比（長條圖、重疊情形）
- [ ] 各特徵的分布、箱型圖與分模式比較
- [ ] 相關係數熱圖，並說明 Pearson 係數在非線性與不平衡下的限制
- [ ] 扭矩 × 轉速散點圖疊加功率等值線，標示 PWF 邊界
- [ ] 溫差 × 轉速散點圖，標示 HDF 邊界
- [ ] 磨耗 × 扭矩散點圖，標示 OSF／TWF 區域
- [ ] 依機型與磨耗分桶計算故障率，說明混雜因素（機型的操作條件不同）
- [ ] 檢查 `UDI` 順序下的磨耗累積與重置型態
- [ ] 輸出 `results/eda_summary.json` 與 `results/eda_summary.md`

### 2. 特徵工程與洩漏防護

- [ ] 排除 `UDI`、`Product ID` 與五個模式旗標（旗標是 target 的組成，直接使用即為洩漏）
- [ ] 新增領域特徵：功率（W）、溫差（K）、磨耗 × 扭矩、轉速偏離量
- [ ] `Type` 以 one-hot 或有序編碼處理，並在附錄說明選擇理由
- [ ] 建立特徵白名單與自動檢查（不得出現 target 衍生欄位）
- [ ] 以「含旗標」的對照實驗示範洩漏會導致近乎完美的假象（僅作教學對照，不列入正式結果）

### 3. 建模與不平衡處理

- [ ] 分層 80/20 切分（seed=42），測試集只在最後評估一次
- [ ] 開發集上以 repeated stratified 5-fold（3 次）做模型選擇
- [ ] 候選模型：Dummy、邏輯迴歸、隨機森林、HistGradientBoosting、MLP
- [ ] 比較類別權重、SMOTE（僅在折內）、不處理三種策略
- [ ] 建立硬編碼物理規則 baseline（HDF／PWF／OSF／TWF）作為對照
- [ ] 以成本矩陣（FN:FP = 10:1）選擇決策門檻，並保留預設 0.5 的結果供比較

### 4. 評估與不確定性

- [ ] 主要指標為 PR-AUC（average precision），輔以 ROC-AUC、recall@precision≥0.9、MCC、macro-F1、Brier score
- [ ] 輸出 PR／ROC 曲線、混淆矩陣、門檻—成本權衡圖、校準曲線
- [ ] 以 2,000 次 bootstrap 估計主要指標的信賴區間
- [ ] 以 10 個隨機種子檢查穩定性
- [ ] 以 `UDI` 排序的 80/20 切分做穩健性對照（模擬時間外推）
- [ ] 排除 27 筆疑似標籤不一致的資料重跑，確認結論不變
- [ ] 完整保存 `results/metrics.json`、`results/cv_results.csv`、`results/test_predictions.csv`

### 5. 故障模式分解與可學習性

- [ ] 對 TWF、HDF、PWF、OSF 各建立 one-vs-rest 模型
- [ ] 對 RNF 建立對照模型，證明其效能與隨機猜測無異（ROC-AUC ≈ 0.5）
- [ ] 將各模式的 PR-AUC 與規則 baseline 並列，呈現「可規則化／部分可學習／不可學習」三類
- [ ] 說明 HDF／PWF 接近 1.0 是流程正確性的 sanity check，而非研究成果
- [ ] 產出各模式可學習性長條圖與規則對照表

### 6. 解釋性與穩健性

- [ ] permutation importance（測試集、多次重抽）
- [ ] PDP／ICE：扭矩、磨耗、溫差、轉速
- [ ] 以淺層決策樹抽取門檻，與文獻規則比對（規則回收分析）
- [ ] 錯誤分析：偽陰性、臨界樣本、疑似標籤噪音
- [ ] 學習曲線：資料量對效能的影響
- [ ] SHAP 列為選配；若 Python 3.14 環境不支援則以 permutation importance 與 PDP 取代

### 7. 小論文與重現

- [ ] 依實際實驗數據撰寫：摘要、緒論、文獻探討、方法、結果、討論、結論、參考文獻
- [ ] 圖表與數字和 `results/metrics.json` 一致
- [ ] 輸出 Markdown 與自包含 HTML（可再列印為 PDF）
- [ ] 專章討論合成資料的效度威脅與外推限制
- [ ] `python ai4i2020/run_all.py` 可一鍵重跑，固定 seed=42
- [ ] 更新 `ai4i2020/README.md` 與根目錄專案索引

## 實驗設計摘要

| 項目 | 設計 |
| --- | --- |
| 主要切分 | 分層 80/20（開發 8,000／測試 2,000，測試集約 68 個正例） |
| 模型選擇 | 開發集 repeated stratified 5-fold × 3 repeats |
| 穩健性切分 | 依 `UDI` 排序的 80/20（前 8,000 訓練、後 2,000 測試） |
| 不平衡策略 | class weight、SMOTE（折內）、不處理；測試集不重採樣 |
| 主要指標 | PR-AUC；輔以 ROC-AUC、recall@precision≥0.9、MCC、Brier |
| 決策門檻 | 最小化期望成本，FN:FP = 10:1 |
| 不確定性 | 2,000 次 bootstrap、10 個種子、學習曲線 |
| 解釋 | permutation importance、PDP／ICE、淺層決策樹規則回收 |

## 已知風險與效度威脅

- **合成資料本質**：資料由理論模型生成，模型高分代表「學到生成規則」，不代表可轉移到真實產線。
- **模式旗標洩漏**：TWF／HDF／PWF／OSF 是 target 的組成，必須排除；`Product ID` 與 `UDI` 亦須排除。
- **送分題拉高整體分數**：HDF／PWF 可完全規則化，整體 PR-AUC 會偏樂觀，必須拆模式呈現。
- **RNF 天花板**：隨機失效無法預測，任何模型的 recall 都有上限。
- **樣本量**：僅 339 個正例，測試集約 68 個，信賴區間寬，單一 accuracy 差距不具統計意義。
- **隨機切分偏樂觀**：資料無真實時間戳，`UDI` 僅為生成順序；以時間切分對照可部分檢查外推性。
- **標籤不一致**：27 筆異常可能影響訓練與評估，需做敏感度分析。
- **不平衡指標陷阱**：不得以 accuracy 作為主要指標；全猜多數類即可達 96.6%。
- **相關不等於因果**：機型故障率差異受扭矩、轉速等操作條件混雜，不宜直接解讀為機型體質。

## 交付物

| 路徑 | 內容 |
| --- | --- |
| `ai4i2020/data/` | 原始 zip／CSV、整理後資料、欄位字典、SHA-256 |
| `ai4i2020/src/01_get_data.py` ~ `07_build_paper.py` | 可重現的分析流程 |
| `ai4i2020/run_all.py` | 一鍵重跑全部流程 |
| `ai4i2020/figures/` | 約 18 張靜態圖表（含互動式圖另行標註） |
| `ai4i2020/results/` | 指標 JSON、CV 結果、預測結果、模型檔 |
| `ai4i2020/paper/ai4i2020_paper.md` / `.html` / `.pdf` | 小論文 |
| `ai4i2020/README.md` | 專案說明與重現步驟 |
| 根目錄 `README.md` | 加入 Iris 與 AI4I 2020 的專案索引 |

預計專案結構：

```text
ai4i2020/
├── ISSUE.md
├── README.md
├── run_all.py
├── requirements.txt
├── data/
├── src/
│   ├── common.py
│   ├── 01_get_data.py
│   ├── 02_eda.py
│   ├── 03_features.py
│   ├── 04_train.py
│   ├── 05_modes.py
│   ├── 06_explain.py
│   └── 07_build_paper.py
├── figures/
├── results/
└── paper/
```

## 完成定義（Definition of Done）

1. `python ai4i2020/run_all.py` 可在乾淨環境重跑並得到相同結果（固定 seed=42）。
2. 測試集指標、混淆矩陣與小論文數字完全一致。
3. 所有圖表均有標題、軸標籤（含單位）與可辨識的圖例。
4. 五種故障模式分開呈現，並明確標示哪些模式可規則化、哪些不可學習。
5. 小論文如實呈現限制：合成資料、類別不平衡、標籤不一致與外推風險。
6. 專案已提交並推送至 `main` 分支，Issues 與 `ISSUE.md` 狀態同步。

## 參考資料

- Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 69–74. https://doi.org/10.1109/AI4I49448.2020.00023
- Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset.* UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
- UCI Machine Learning Repository #601：https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- 資料授權：CC BY 4.0
