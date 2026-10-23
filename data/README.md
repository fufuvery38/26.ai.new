# Iris 資料集

## 來源
- 原始論文：Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179–188.
- 取得管道：UCI Machine Learning Repository（archive.ics.uci.edu/dataset/53/iris）
- 下載網址：<https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data>
- 取得日期：2026-10-23

## 檔案
| 檔案 | 說明 |
| --- | --- |
| `iris_raw.data` | 原始下載檔（未修改，保留完整追溯性） |
| `iris_bezdek.data` | UCI 提供的官方修正版 `bezdekIris.data` |
| `iris.csv` | 整理後資料（已套用官方更正），含欄位名稱，共 150 筆 |
| `iris.names` | UCI 官方欄位說明 |

## 資料規格
- 樣本數：150（三個品種各 50 筆，類別平衡）
- 特徵數：4（皆為連續數值，單位為公分）
- 標籤：`species` ∈ {setosa, versicolor, virginica}
- 缺失值：0
- 完全重複列：1（原始檔為 3 列）
- 與 scikit-learn 內建版本逐值比對：一致

## 欄位
| 欄位 | 中文 | 型態 | 最小值 | 最大值 | 平均 |
| --- | --- | --- | ---: | ---: | ---: |
| `sepal_length` | 花萼長度 | float | 4.3 | 7.9 | 5.84 |
| `sepal_width` | 花萼寬度 | float | 2.0 | 4.4 | 3.06 |
| `petal_length` | 花瓣長度 | float | 1.0 | 6.9 | 3.76 |
| `petal_width` | 花瓣寬度 | float | 0.1 | 2.5 | 1.20 |

## 資料品質更正

UCI 原始檔 `iris.data` 中，第 35 與第 38 筆 setosa 記錄存在流傳已久的數值錯誤。
本專案以 UCI 官方修正版 `bezdekIris.data` 為參照，逐值比對後更正如下：

| 筆數（1-based） | 欄位 | 原始值 | 更正後 |
| ---: | --- | ---: | ---: |
| 35 | 花瓣寬度 | 0.1 | 0.2 |
| 38 | 花萼寬度 | 3.1 | 3.6 |
| 38 | 花瓣長度 | 1.5 | 1.4 |

> 原始檔仍完整保留於 `iris_raw.data`，修正僅作用於 `iris.csv`。

## 完整性
- `iris_raw.data` SHA-256：`6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0`
- `iris_bezdek.data` SHA-256：`0fed2a99db77ec533a62dc66894d3ec6df3b58b6a8f3cf4a6b47e4086b7f97dc`
- `iris.csv` SHA-256：`e404da8a0348eaa780e968c07a8f4dc90fe90eea54ddceeb5b444ce0caae8d30`
