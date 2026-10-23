# 資料說明：AI4I 2020 Predictive Maintenance Dataset

## 來源

| 項目 | 內容 |
| --- | --- |
| 資料集 | AI4I 2020 Predictive Maintenance Dataset |
| 提供者 | UCI Machine Learning Repository（#601） |
| 作者 | Stephan Matzka |
| DOI | [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C) |
| 授權 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 下載網址 | <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset> |
| 取得日期 | 2026-10-23 |

原始檔案完整保留、未做任何修改。所有清理與欄位重新命名都由 `src/01_get_data.py` 產生。

## 檔案

| 檔案 | 說明 | SHA-256 |
| --- | --- | --- |
| `ai4i2020.zip` | UCI 原始壓縮檔 | `f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252` |
| `ai4i2020_raw.csv` | 自 zip 取出的原始 CSV（未修改） | `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |
| `ai4i2020.csv` | 整理後資料（欄位改為 snake_case、固定型別） | `f4cc04cb26aeb77742729e7fccf13a77f9ac13069215943eef7e9fd8d3fe5929` |
| `dataset_summary.json` | 來源、授權、雜湊、筆數等摘要 | — |

## 欄位

| 欄位 | 中文 | 單位 | 說明 |
| --- | --- | --- | --- |
| `udi` | 資料列編號 | — | 1–10,000，僅為識別碼 |
| `product_id` | 產品序號 | — | 首碼等於 `type`，僅為識別碼 |
| `type` | 機型 | — | L（低階）／M（中階）／H（高階） |
| `air_temp_k` | 空氣溫度 | K | 環境溫度 |
| `process_temp_k` | 製程溫度 | K | 製程側溫度 |
| `rotational_speed_rpm` | 轉速 | rpm | 主軸轉速 |
| `torque_nm` | 扭矩 | Nm | 主軸扭矩 |
| `tool_wear_min` | 工具磨耗 | min | 累積磨耗時間 |
| `machine_failure` | 機器故障 | — | 目標欄位（0／1） |
| `twf` | 工具磨耗失效 | — | Tool Wear Failure |
| `hdf` | 散熱失效 | — | Heat Dissipation Failure |
| `pwf` | 功率失效 | — | Power Failure |
| `osf` | 過度應變失效 | — | Overstrain Failure |
| `rnf` | 隨機失效 | — | Random Failure |

衍生特徵不寫入 CSV，由 `src/common.py` 的 `add_derived_features()` 計算：

- `power_w` = `torque_nm × rotational_speed_rpm × 2π / 60`（W）
- `temp_diff_k` = `process_temp_k − air_temp_k`（K）
- `wear_torque` = `tool_wear_min × torque_nm`（min·Nm）

## 資料品質摘要

| 項目 | 結果 |
| --- | --- |
| 筆數 × 欄數 | 10,000 × 14 |
| 缺失值 | 0 |
| 完全重複列 | 0 |
| 目標正例 | 339（3.39%） |
| 模式筆數 | TWF 46、HDF 115、PWF 95、OSF 98、RNF 19 |
| 多模式重疊 | 324 筆單一模式、23 筆雙模式、1 筆三模式 |
| 機型分布 | L 6,000（3.92%）、M 2,997（2.77%）、H 1,003（2.09%） |

完整統計見 [`results/data_audit.md`](../results/data_audit.md) 與 [`results/data_audit.json`](../results/data_audit.json)。

## 已知問題

1. **標籤不一致 27 筆**：
   - 18 筆 `rnf=1` 但 `machine_failure=0`（UDI 1222, 1303, 1749, 2073, 2560, 3066, 3453, 5472, 5490, 5496, 5510, 5554, 5640, 6092, 6914, 6961, 7489, 7869）。
   - 9 筆 `machine_failure=1` 但五個模式旗標皆為 0（UDI 1438, 2750, 4045, 4685, 5537, 5942, 6479, 8507, 9016）。
   - 其餘四種模式旗標皆為 `machine_failure` 的子集。
   - 主分析保留原標籤，並在評估階段做排除後重跑的敏感度分析。
2. **模式旗標不可作為特徵**：`twf`、`hdf`、`pwf`、`osf`、`rnf` 是目標的組成，直接使用即為洩漏。
3. **識別碼不可作為特徵**：`udi` 與 `product_id` 不帶有可外推的物理資訊，且 `product_id` 首碼即 `type`。
4. **合成資料**：本資料由理論模型生成，並非真實產線感測紀錄，結論不可直接外推。

## 已驗證的物理規則

| 模式 | 規則 | 結果 |
| --- | --- | --- |
| PWF | 功率 < 3,500 W 或 > 9,000 W | 95/95 命中，零誤判 |
| HDF | 溫差 < 8.6 K 且轉速 < 1,380 rpm | 115/115 命中，零誤判 |
| OSF | 高磨耗＋高扭矩 | 部分可解釋；正例磨耗 172–253 min、扭矩 46.3–75.4 Nm |
| TWF | 高磨耗區間內隨機發生 | 正例磨耗 198–253 min；200–240 min 區間內 790 筆只有 43 筆 |
| RNF | 無可用規則 | 本質隨機，不可學習 |

## 重新產生

```bash
python src/01_get_data.py          # 沿用既有 zip，驗證雜湊
python src/01_get_data.py --force  # 強制重新下載
```

## 引用

> Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

相關論文：

> Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 69–74. https://doi.org/10.1109/AI4I49448.2020.00023
