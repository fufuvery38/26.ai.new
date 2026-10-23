"""01 - 取得 Iris 資料集並保存到專案中。

資料來源（原始出處）:
    R. A. Fisher (1936). The use of multiple measurements in taxonomic problems.
    Annals of Eugenics, 7(2), 179-188.
    UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/53/iris

流程:
    1. 從 UCI 下載 iris.data、bezdekIris.data 與欄位說明 iris.names。
    2. 解析並命名欄位；比對官方修正版，套用已知的兩筆資料更正。
    3. 輸出 data/iris.csv（修正後）與 data/iris_bezdek.data（官方修正版參照）。
    4. 與 scikit-learn 內建版本逐值比對，確認資料完整性。
    5. 產生 data/README.md 記錄來源、SHA-256 與資料檢查結果。
"""

from __future__ import annotations

import hashlib
import urllib.request
from datetime import date

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from common import (
    BEZDEK_DATA,
    CSV_DATA,
    DATA_DIR,
    FEATURES,
    FEATURES_ZH,
    NAMES_FILE,
    RAW_DATA,
    SPECIES,
    save_json,
)

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris"
TIMEOUT = 60


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "iris-coursework/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
        if len(data) < 1000:
            raise ValueError(f"檔案過小（{len(data)} bytes）")
        path.write_bytes(data)
        print(f"[下載] {url} -> {path.relative_to(DATA_DIR.parent)} ({len(data)} bytes)")
        return True
    except Exception as exc:  # 網路不可用時仍可完成作業
        print(f"[警告] 無法下載 {url}: {exc}")
        return False


def parse_uci(path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"格式不符: {line!r}")
        rows.append(parts)
    df = pd.DataFrame(rows, columns=FEATURES + ["species_raw"])
    df[FEATURES] = df[FEATURES].astype(float)
    df["species"] = df["species_raw"].str.replace("Iris-", "", regex=False).str.lower()
    return df[FEATURES + ["species"]]


def build_from_sklearn() -> pd.DataFrame:
    bunch = load_iris(as_frame=True)
    df = bunch.data.copy()
    df.columns = FEATURES
    df["species"] = [SPECIES[i] for i in bunch.target]
    return df


def main() -> None:
    print("=" * 70)
    print("步驟 1/4：取得 Iris 資料集")
    print("=" * 70)

    got_uci = download(f"{BASE_URL}/iris.data", RAW_DATA)
    got_bezdek = download(f"{BASE_URL}/bezdekIris.data", BEZDEK_DATA)
    download(f"{BASE_URL}/iris.names", NAMES_FILE)

    if got_uci:
        df = parse_uci(RAW_DATA)
        source = "UCI Machine Learning Repository（archive.ics.uci.edu/dataset/53/iris）"
        corrections: list[dict] = []
        if got_bezdek:
            corrected = parse_uci(BEZDEK_DATA)
            diff_mask = (df[FEATURES].to_numpy() != corrected[FEATURES].to_numpy()).any(axis=1)
            for idx in np.where(diff_mask)[0]:
                for c in FEATURES:
                    before, after = float(df.at[idx, c]), float(corrected.at[idx, c])
                    if before != after:
                        corrections.append(
                            {
                                "row_1based": int(idx + 1),
                                "feature": c,
                                "from": before,
                                "to": after,
                            }
                        )
                df.loc[idx, FEATURES] = corrected.loc[idx, FEATURES].to_numpy()
            print(f"[資料檢查] 與官方修正版比對：發現 {int(diff_mask.sum())} 筆記錄、{len(corrections)} 個數值需更正")
            for item in corrections:
                print(
                    f"           第 {item['row_1based']:>3} 筆 {FEATURES_ZH[item['feature']]}："
                    f"{item['from']} -> {item['to']}"
                )
            assert (
                df[FEATURES].to_numpy() == corrected[FEATURES].to_numpy()
            ).all(), "套用官方修正版後仍不一致"
        else:
            print("[警告] 未取得 bezdekIris.data，改以 scikit-learn 版本作為修正參照。")
            df = build_from_sklearn()
    else:
        df = build_from_sklearn()
        source = "scikit-learn 內建副本（UCI 網路不可用時的 fallback）"
        corrections = []
        df.to_csv(RAW_DATA, index=False)

    # 與 scikit-learn 版本逐值比對（Bézdek 修正版）
    ref = build_from_sklearn()
    features_equal = bool(np.allclose(df[FEATURES].to_numpy(), ref[FEATURES].to_numpy(), atol=1e-9))
    labels_equal = bool((df["species"].to_numpy() == ref["species"].to_numpy()).all())
    if not (features_equal and labels_equal):
        raise AssertionError("資料修正後仍與 scikit-learn 版本不一致，請檢查下載內容。")
    print("[檢查] 修正後資料與 scikit-learn 內建版本逐值一致 ✓")

    df["species"] = pd.Categorical(df["species"], categories=SPECIES, ordered=True)
    df.to_csv(CSV_DATA, index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # 資料檢查
    # ------------------------------------------------------------------
    counts = df["species"].value_counts().reindex(SPECIES)
    n_dup = int(df.duplicated().sum())
    n_dup_raw = int(parse_uci(RAW_DATA).duplicated().sum()) if got_uci else n_dup
    missing = int(df.isna().sum().sum())
    value_ranges = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
        for c in FEATURES
    }

    print(f"[資料] 筆數={len(df)}，欄位={list(df.columns)}")
    print(f"[資料] 類別分布={counts.to_dict()}，缺失值={missing}，完全重複列={n_dup}（原始檔 {n_dup_raw}）")
    print("[資料] 特徵範圍:")
    for c in FEATURES:
        print(f"       {c:<14} min={df[c].min():.1f}  max={df[c].max():.1f}  mean={df[c].mean():.2f}")

    summary = {
        "source": source,
        "source_url": f"{BASE_URL}/iris.data",
        "retrieved": date.today().isoformat(),
        "n_rows": int(len(df)),
        "n_features": len(FEATURES),
        "features": FEATURES,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "missing_values": missing,
        "duplicate_rows": n_dup,
        "duplicate_rows_raw": n_dup_raw,
        "matches_sklearn": features_equal and labels_equal,
        "corrections": corrections,
        "value_ranges": value_ranges,
        "sha256_raw": sha256(RAW_DATA),
        "sha256_bezdek": sha256(BEZDEK_DATA) if BEZDEK_DATA.exists() else None,
        "sha256_csv": sha256(CSV_DATA),
        "license_note": "Iris 資料集為公開發表之學術資料（Fisher, 1936），常作為機器學習教學範例。",
    }
    save_json(summary, DATA_DIR / "dataset_summary.json")

    readme = f"""# Iris 資料集

## 來源
- 原始論文：Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179–188.
- 取得管道：{source}
- 下載網址：<{BASE_URL}/iris.data>
- 取得日期：{summary['retrieved']}

## 檔案
| 檔案 | 說明 |
| --- | --- |
| `iris_raw.data` | 原始下載檔（未修改，保留完整追溯性） |
| `iris_bezdek.data` | UCI 提供的官方修正版 `bezdekIris.data` |
| `iris.csv` | 整理後資料（已套用官方更正），含欄位名稱，共 {len(df)} 筆 |
| `iris.names` | UCI 官方欄位說明 |

## 資料規格
- 樣本數：{len(df)}（三個品種各 {int(counts.min())} 筆，類別平衡）
- 特徵數：{len(FEATURES)}（皆為連續數值，單位為公分）
- 標籤：`species` ∈ {{setosa, versicolor, virginica}}
- 缺失值：{missing}
- 完全重複列：{n_dup}（原始檔為 {n_dup_raw} 列）
- 與 scikit-learn 內建版本逐值比對：{'一致' if summary['matches_sklearn'] else '不一致'}

## 欄位
| 欄位 | 中文 | 型態 | 最小值 | 最大值 | 平均 |
| --- | --- | --- | ---: | ---: | ---: |
"""
    for c in FEATURES:
        r = value_ranges[c]
        readme += f"| `{c}` | {FEATURES_ZH[c]} | float | {r['min']:.1f} | {r['max']:.1f} | {r['mean']:.2f} |\n"

    readme += "\n## 資料品質更正\n\n"
    if corrections:
        readme += (
            "UCI 原始檔 `iris.data` 中，第 35 與第 38 筆 setosa 記錄存在流傳已久的數值錯誤。\n"
            "本專案以 UCI 官方修正版 `bezdekIris.data` 為參照，逐值比對後更正如下：\n\n"
        )
        readme += "| 筆數（1-based） | 欄位 | 原始值 | 更正後 |\n| ---: | --- | ---: | ---: |\n"
        for item in corrections:
            readme += f"| {item['row_1based']} | {FEATURES_ZH[item['feature']]} | {item['from']} | {item['to']} |\n"
        readme += "\n> 原始檔仍完整保留於 `iris_raw.data`，修正僅作用於 `iris.csv`。\n"
    else:
        readme += "原始檔與官方修正版一致，無需更正。\n"

    readme += f"""
## 完整性
- `iris_raw.data` SHA-256：`{summary['sha256_raw']}`
- `iris_bezdek.data` SHA-256：`{summary['sha256_bezdek']}`
- `iris.csv` SHA-256：`{summary['sha256_csv']}`
"""
    (DATA_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"[完成] 已寫入 {CSV_DATA.relative_to(DATA_DIR.parent)}、data/README.md、data/dataset_summary.json")


if __name__ == "__main__":
    from common import force_utf8_stdout

    force_utf8_stdout()
    main()
