"""步驟 0：下載 AI4I 2020、驗證雜湊、輸出整理後資料與資料稽核。"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    COLUMN_MAP,
    CSV_DATA,
    DATA_DIR,
    DATASET_CITATION,
    DATASET_DOI,
    DATASET_LICENSE,
    DATASET_NAME,
    DATASET_URL,
    MODE_FLAGS,
    RAW_CSV,
    RAW_CSV_SHA256,
    RAW_ZIP,
    RAW_ZIP_SHA256,
    RESULT_DIR,
    SENSOR_FEATURES,
    SUMMARY_JSON,
    TARGET,
    TYPE_ORDER,
    force_utf8_stdout,
    save_json,
    sha256_bytes,
    sha256_file,
)

AUDIT_JSON = RESULT_DIR / "data_audit.json"
AUDIT_MD = RESULT_DIR / "data_audit.md"
PROJECT_ROOT = DATA_DIR.parent


# ---------------------------------------------------------------------------
# 取得檔案
# ---------------------------------------------------------------------------
def ensure_zip(force: bool = False) -> None:
    """下載或沿用既有 zip，並驗證 SHA-256。"""
    if RAW_ZIP.exists() and not force:
        digest = sha256_file(RAW_ZIP)
        if digest == RAW_ZIP_SHA256:
            print(f"[1/5] 沿用既有 {RAW_ZIP.name}（SHA-256 相符）")
            return
        print(f"[1/5] 既有 zip 雜湊不符（{digest}），重新下載")

    print(f"[1/5] 下載 {DATASET_URL}")
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "ai4i2020-repro/1.0 (educational research)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()

    digest = sha256_bytes(payload)
    if digest != RAW_ZIP_SHA256:
        raise RuntimeError(
            "下載檔案的 SHA-256 與預期不符，上游可能已更新。\n"
            f"預期: {RAW_ZIP_SHA256}\n實際: {digest}"
        )
    RAW_ZIP.write_bytes(payload)
    print(f"      完成，{len(payload):,} bytes，SHA-256 相符")


def extract_raw_csv() -> None:
    """從 zip 取出原始 CSV，保留未修改內容。"""
    with zipfile.ZipFile(RAW_ZIP) as archive:
        candidates = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(candidates) != 1:
            raise RuntimeError(f"zip 內應只有一個 CSV，實際為 {candidates}")
        payload = archive.read(candidates[0])

    digest = sha256_bytes(payload)
    if digest != RAW_CSV_SHA256:
        raise RuntimeError(
            "原始 CSV 的 SHA-256 與預期不符。\n"
            f"預期: {RAW_CSV_SHA256}\n實際: {digest}"
        )
    RAW_CSV.write_bytes(payload)
    print(f"[2/5] 取出 {candidates[0]}，{len(payload):,} bytes，SHA-256 相符")


# ---------------------------------------------------------------------------
# 整理與稽核
# ---------------------------------------------------------------------------
def build_processed() -> pd.DataFrame:
    """重新命名欄位、固定型別，輸出乾淨的 CSV。"""
    df = pd.read_csv(RAW_CSV)
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise RuntimeError(f"原始檔缺少欄位: {sorted(missing)}")

    df = df.rename(columns=COLUMN_MAP)
    df["type"] = pd.Categorical(df["type"], categories=TYPE_ORDER, ordered=True)
    for col in [TARGET, *MODE_FLAGS]:
        df[col] = df[col].astype("int8")
    for col in ["udi", "rotational_speed_rpm", "tool_wear_min"]:
        df[col] = df[col].astype("int64")
    df["product_id"] = df["product_id"].astype("string")

    df.to_csv(CSV_DATA, index=False, encoding="utf-8", lineterminator="\n")
    print(f"[3/5] 輸出 {CSV_DATA.name}（{len(df):,} 筆 × {df.shape[1]} 欄）")
    return df


def _describe(df: pd.DataFrame) -> dict:
    desc = df[SENSOR_FEATURES].describe().T
    return {
        str(idx): {str(k): round(float(v), 4) for k, v in row.items()}
        for idx, row in desc.iterrows()
    }


def _confusion(pred: pd.Series, actual: pd.Series) -> dict:
    pred = pred.astype(bool)
    actual = actual.astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
    }


def audit(df: pd.DataFrame) -> dict:
    """檢查資料品質、標籤一致性與物理規則。"""
    n = len(df)
    y = df[TARGET]

    mode_counts = {
        flag: {
            "count": int(df[flag].sum()),
            "rate_pct": round(float(df[flag].mean() * 100), 4),
        }
        for flag in MODE_FLAGS
    }
    overlap = df[MODE_FLAGS].sum(axis=1).value_counts().sort_index()

    non_rnf = df[["twf", "hdf", "pwf", "osf"]].max(axis=1)
    rnf_zero_target = df.loc[(df["rnf"] == 1) & (y == 0), "udi"].astype(int).tolist()
    target_no_flag = df.loc[(y == 1) & (df[MODE_FLAGS].sum(axis=1) == 0), "udi"].astype(int).tolist()
    non_rnf_leak = int(((non_rnf == 1) & (y == 0)).sum())

    power_w = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60
    temp_diff_k = df["process_temp_k"] - df["air_temp_k"]
    wear_window = df["tool_wear_min"].between(200, 240)

    pwf_rule = (power_w < 3500) | (power_w > 9000)
    hdf_rule = (temp_diff_k < 8.6) & (df["rotational_speed_rpm"] < 1380)

    osf_in_window = int((wear_window & (df["osf"] == 1)).sum())
    twf_in_window = int((wear_window & (df["twf"] == 1)).sum())
    osf_torque = df.loc[df["osf"] == 1, "torque_nm"]
    twf_wear = df.loc[df["twf"] == 1, "tool_wear_min"]
    osf_wear = df.loc[df["osf"] == 1, "tool_wear_min"]

    type_rows = {}
    for t in TYPE_ORDER:
        sub = df[df["type"] == t]
        type_rows[t] = {
            "count": int(len(sub)),
            "failures": int(sub[TARGET].sum()),
            "failure_rate_pct": round(float(sub[TARGET].mean() * 100), 4),
        }

    return {
        "shape": {"rows": int(n), "columns": int(df.shape[1])},
        "dtypes": {c: str(dtype) for c, dtype in df.dtypes.items()},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows_full": int(df.duplicated().sum()),
        "duplicate_rows_features": int(
            df.drop(columns=["udi", "product_id", *MODE_FLAGS]).duplicated().sum()
        ),
        "unique_ids": {
            "udi": int(df["udi"].nunique()),
            "product_id": int(df["product_id"].nunique()),
        },
        "target": {
            "column": TARGET,
            "counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "positive_rate_pct": round(float(y.mean() * 100), 4),
        },
        "modes": mode_counts,
        "mode_overlap": {
            "flags_per_row": {str(k): int(v) for k, v in overlap.items()},
            "max_flags_per_row": int(df[MODE_FLAGS].sum(axis=1).max()),
        },
        "types": type_rows,
        "numeric_summary": _describe(df),
        "physical_checks": {
            "process_warmer_than_air_all": bool((temp_diff_k > 0).all()),
            "min_temp_diff_k": round(float(temp_diff_k.min()), 4),
            "max_temp_diff_k": round(float(temp_diff_k.max()), 4),
            "tool_wear_range_min": [int(df["tool_wear_min"].min()), int(df["tool_wear_min"].max())],
            "product_id_prefix_matches_type": bool(
                (df["product_id"].str[0] == df["type"].astype(str)).all()
            ),
            "flags_binary": bool(df[MODE_FLAGS].isin([0, 1]).all().all()),
            "type_values": sorted(df["type"].astype(str).unique().tolist()),
        },
        "rules": {
            "power_w": {
                "formula": "torque_nm * rotational_speed_rpm * 2*pi/60",
                "normal_range_w": [3500, 9000],
                "pwf_rule_vs_flag": _confusion(pwf_rule, df["pwf"]),
            },
            "hdf": {
                "rule": "temp_diff_k < 8.6 and rotational_speed_rpm < 1380",
                "rule_vs_flag": _confusion(hdf_rule, df["hdf"]),
            },
            "osf": {
                "wear_window_min": [200, 240],
                "rows_in_window": int(wear_window.sum()),
                "osf_in_window": osf_in_window,
                "window_only_vs_flag": _confusion(wear_window, df["osf"]),
                "osf_torque_range_nm": [
                    round(float(osf_torque.min()), 4),
                    round(float(osf_torque.max()), 4),
                ],
                "osf_torque_median_nm": round(float(osf_torque.median()), 4),
                "osf_wear_range_min": [
                    int(osf_wear.min()),
                    int(osf_wear.max()),
                ],
            },
            "twf": {
                "wear_window_min": [200, 240],
                "rows_in_window": int(wear_window.sum()),
                "twf_in_window": twf_in_window,
                "twf_rate_in_window_pct": round(float(twf_in_window / wear_window.sum() * 100), 4),
                "twf_wear_range_min": [int(twf_wear.min()), int(twf_wear.max())],
            },
            "rnf": {
                "count": int(df["rnf"].sum()),
                "vs_target": {
                    "rnf1_target1": int(((df["rnf"] == 1) & (y == 1)).sum()),
                    "rnf1_target0": int(((df["rnf"] == 1) & (y == 0)).sum()),
                },
            },
        },
        "integrity": {
            "non_rnf_flags_outside_target": non_rnf_leak,
            "rnf_positive_target_zero": len(rnf_zero_target),
            "rnf_positive_target_zero_udi": rnf_zero_target,
            "target_positive_without_any_mode": len(target_no_flag),
            "target_positive_without_any_mode_udi": target_no_flag,
            "target_vs_or_of_flags_mismatch": int(
                (y.astype(bool) != df[MODE_FLAGS].max(axis=1).astype(bool)).sum()
            ),
        },
    }


def write_markdown(df: pd.DataFrame, report: dict) -> None:
    modes = report["modes"]
    types = report["types"]
    rules = report["rules"]
    integrity = report["integrity"]

    lines = [
        "# AI4I 2020 資料稽核報告",
        "",
        "由 `src/01_get_data.py` 自動產生。",
        "",
        "## 來源與檔案",
        "",
        "| 項目 | 內容 |",
        "| --- | --- |",
        f"| 資料集 | {DATASET_NAME}（UCI #601） |",
        f"| DOI | [{DATASET_DOI}](https://doi.org/{DATASET_DOI}) |",
        f"| 授權 | {DATASET_LICENSE} |",
        f"| 原始 zip SHA-256 | `{RAW_ZIP_SHA256}` |",
        f"| 原始 CSV SHA-256 | `{RAW_CSV_SHA256}` |",
        f"| 整理後 CSV SHA-256 | `{sha256_file(CSV_DATA)}` |",
        "",
        "## 基本品質",
        "",
        "| 檢查 | 結果 |",
        "| --- | --- |",
        f"| 筆數 × 欄數 | {report['shape']['rows']:,} × {report['shape']['columns']} |",
        f"| 缺失值 | {report['missing_values']} |",
        f"| 完全重複列 | {report['duplicate_rows_full']} |",
        f"| 特徵重複列（排除 ID 與旗標） | {report['duplicate_rows_features']} |",
        f"| 唯一 UDI／Product ID | {report['unique_ids']['udi']:,}／{report['unique_ids']['product_id']:,} |",
        f"| Product ID 首碼等於 Type | {report['physical_checks']['product_id_prefix_matches_type']} |",
        "",
        "## 目標與故障模式",
        "",
        f"- `machine_failure` 正例：**{report['target']['counts']['1']:,} 筆"
        f"（{report['target']['positive_rate_pct']:.2f}%）**",
        "",
        "| 模式 | 筆數 | 占比 |",
        "| --- | ---: | ---: |",
    ]
    for flag in MODE_FLAGS:
        lines.append(f"| {flag.upper()} | {modes[flag]['count']} | {modes[flag]['rate_pct']:.2f}% |")

    lines += [
        "",
        "同一筆資料可能同時有多個模式旗標：",
        "",
        "| 旗標數 | 筆數 |",
        "| ---: | ---: |",
    ]
    for k, v in report["mode_overlap"]["flags_per_row"].items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## 各機型故障率",
        "",
        "| 機型 | 筆數 | 故障數 | 故障率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for t in TYPE_ORDER:
        row = types[t]
        lines.append(
            f"| {t} | {row['count']:,} | {row['failures']} | {row['failure_rate_pct']:.2f}% |"
        )

    lines += [
        "",
        "## 物理規則驗證",
        "",
        "| 規則 | TP | FP | FN | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    checks = [
        ("PWF：功率 < 3,500 W 或 > 9,000 W", rules["power_w"]["pwf_rule_vs_flag"]),
        ("HDF：ΔT < 8.6 K 且轉速 < 1,380 rpm", rules["hdf"]["rule_vs_flag"]),
    ]
    for name, c in checks:
        lines.append(
            f"| {name} | {c['tp']} | {c['fp']} | {c['fn']} | "
            f"{c['precision']:.3f} | {c['recall']:.3f} |"
        )

    osf = rules["osf"]
    twf = rules["twf"]
    lines += [
        "",
        f"- 磨耗 200–240 min 共 {osf['rows_in_window']} 筆，其中 OSF={osf['osf_in_window']}、"
        f"TWF={twf['twf_in_window']}。",
        f"- 此區間與兩種模式高度相關，但**不是必要條件**：OSF 正例的磨耗範圍為 "
        f"{osf['osf_wear_range_min'][0]}–{osf['osf_wear_range_min'][1]} min，"
        f"TWF 為 {twf['twf_wear_range_min'][0]}–{twf['twf_wear_range_min'][1]} min。",
        f"- OSF 正例的扭矩範圍為 {osf['osf_torque_range_nm'][0]}–{osf['osf_torque_range_nm'][1]} Nm"
        f"（中位數 {osf['osf_torque_median_nm']} Nm），仍需其他條件才能完全分離。",
        f"- TWF 在磨耗 200–240 min 區間內的發生率為 {twf['twf_rate_in_window_pct']:.2f}%，具隨機成分。",
        "",
        "## 標籤一致性問題",
        "",
        f"- 非 RNF 旗標卻 target=0：**{integrity['non_rnf_flags_outside_target']} 筆**",
        f"- RNF=1 卻 target=0：**{integrity['rnf_positive_target_zero']} 筆**"
        f"（UDI：{', '.join(map(str, integrity['rnf_positive_target_zero_udi']))}）",
        f"- target=1 但沒有任何模式旗標：**{integrity['target_positive_without_any_mode']} 筆**"
        f"（UDI：{', '.join(map(str, integrity['target_positive_without_any_mode_udi']))}）",
        f"- `machine_failure` 與五個旗標 OR 不一致：**{integrity['target_vs_or_of_flags_mismatch']} 筆**",
        "",
        "> 合計 27 筆疑似標籤不一致。主分析保留原標籤，並在評估階段做「排除後重跑」的敏感度對照。",
        "",
        "## 數值摘要",
        "",
        "| 特徵 | 平均 | 標準差 | 最小 | 中位數 | 最大 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for col in SENSOR_FEATURES:
        s = report["numeric_summary"][col]
        lines.append(
            f"| {col} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']:.2f} | "
            f"{s['50%']:.2f} | {s['max']:.2f} |"
        )

    lines += [
        "",
        "## 檔案",
        "",
        f"- 原始壓縮檔：`{RAW_ZIP.relative_to(PROJECT_ROOT)}`",
        f"- 原始 CSV：`{RAW_CSV.relative_to(PROJECT_ROOT)}`",
        f"- 整理後 CSV：`{CSV_DATA.relative_to(PROJECT_ROOT)}`",
        f"- 完整稽核 JSON：`{AUDIT_JSON.relative_to(PROJECT_ROOT)}`",
        f"- 資料集摘要：`{SUMMARY_JSON.relative_to(PROJECT_ROOT)}`",
        "",
        "## 引用",
        "",
        f"> {DATASET_CITATION}",
        "",
    ]
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    force_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="即使 zip 已存在仍重新下載")
    args = parser.parse_args()

    ensure_zip(force=args.force)
    extract_raw_csv()
    df = build_processed()

    print("[4/5] 稽核資料品質與物理規則")
    report = audit(df)

    summary = {
        "dataset": DATASET_NAME,
        "source_url": DATASET_URL,
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "retrieved_at": datetime.fromtimestamp(
            RAW_ZIP.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "archive": {
            "file": RAW_ZIP.name,
            "bytes": RAW_ZIP.stat().st_size,
            "sha256": sha256_file(RAW_ZIP),
        },
        "raw_csv": {
            "file": RAW_CSV.name,
            "bytes": RAW_CSV.stat().st_size,
            "sha256": sha256_file(RAW_CSV),
        },
        "processed_csv": {
            "file": CSV_DATA.name,
            "bytes": CSV_DATA.stat().st_size,
            "sha256": sha256_file(CSV_DATA),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "target_positive": int(df[TARGET].sum()),
        "target_rate_pct": round(float(df[TARGET].mean() * 100), 4),
    }
    save_json(summary, SUMMARY_JSON)
    save_json(report, AUDIT_JSON)
    write_markdown(df, report)

    print("[5/5] 完成")
    print(f"      摘要：{SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"      稽核：{AUDIT_JSON.relative_to(PROJECT_ROOT)}、{AUDIT_MD.relative_to(PROJECT_ROOT)}")
    print()
    print(f"筆數：{len(df):,}；欄位：{df.shape[1]}")
    print(
        f"目標：{int(df[TARGET].sum())} 筆（{df[TARGET].mean() * 100:.2f}%）；"
        f"模式："
        + "、".join(f"{f.upper()}={int(df[f].sum())}" for f in MODE_FLAGS)
    )
    print(
        "標籤不一致："
        f"RNF=1/target=0 {report['integrity']['rnf_positive_target_zero']} 筆、"
        f"target=1/無旗標 {report['integrity']['target_positive_without_any_mode']} 筆"
    )
    print(
        "規則檢查："
        f"PWF precision={report['rules']['power_w']['pwf_rule_vs_flag']['precision']}, "
        f"HDF precision={report['rules']['hdf']['rule_vs_flag']['precision']}"
    )


if __name__ == "__main__":
    main()
