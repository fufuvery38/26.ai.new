"""共用設定：路徑、資料集常數、繪圖風格與 JSON 輸出。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# ---------------------------------------------------------------------------
# 路徑
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper"

for _d in (DATA_DIR, FIG_DIR, RESULT_DIR, PAPER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_ZIP = DATA_DIR / "ai4i2020.zip"
RAW_CSV = DATA_DIR / "ai4i2020_raw.csv"
CSV_DATA = DATA_DIR / "ai4i2020.csv"
SUMMARY_JSON = DATA_DIR / "dataset_summary.json"

# ---------------------------------------------------------------------------
# 資料集常數
# ---------------------------------------------------------------------------
SEED = 42

DATASET_NAME = "AI4I 2020 Predictive Maintenance Dataset"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
DATASET_DOI = "10.24432/C5HS5C"
DATASET_LICENSE = "CC BY 4.0"
DATASET_CITATION = (
    "Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C"
)

# 2026-10-23 自 UCI 官方來源下載後記錄；用於偵測上游檔案變動。
RAW_ZIP_SHA256 = "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252"
RAW_CSV_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"

TARGET = "machine_failure"
MODE_FLAGS = ["twf", "hdf", "pwf", "osf", "rnf"]
SENSOR_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
FEATURES = SENSOR_FEATURES  # 正式模型特徵在 03_features.py 定義

COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

TYPE_ORDER = ["L", "M", "H"]
TYPE_ZH = {"L": "低階", "M": "中階", "H": "高階"}

FEATURE_ZH = {
    "air_temp_k": "空氣溫度",
    "process_temp_k": "製程溫度",
    "rotational_speed_rpm": "轉速",
    "torque_nm": "扭矩",
    "tool_wear_min": "工具磨耗",
    "power_w": "功率",
    "temp_diff_k": "製程與空氣溫差",
    "wear_torque": "磨耗 × 扭矩",
    "type": "機型",
    "machine_failure": "機器故障",
}

FEATURE_UNITS = {
    "air_temp_k": "K",
    "process_temp_k": "K",
    "rotational_speed_rpm": "rpm",
    "torque_nm": "Nm",
    "tool_wear_min": "min",
    "power_w": "W",
    "temp_diff_k": "K",
    "wear_torque": "min·Nm",
}

MODE_ZH = {
    "twf": "工具磨耗失效（TWF）",
    "hdf": "散熱失效（HDF）",
    "pwf": "功率失效（PWF）",
    "osf": "過度應變失效（OSF）",
    "rnf": "隨機失效（RNF）",
}

MODE_COLORS = {
    "twf": "#1f77b4",
    "hdf": "#d62728",
    "pwf": "#ff7f0e",
    "osf": "#2ca02c",
    "rnf": "#7f7f7f",
}
FAILURE_COLORS = {0: "#9ecae1", 1: "#d62728"}


# ---------------------------------------------------------------------------
# 共用函式
# ---------------------------------------------------------------------------
def setup_plotting() -> None:
    """設定中文字型與一致的繪圖風格。"""
    candidates = ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [c for c in candidates if c in available]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": chosen + ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def load_data() -> pd.DataFrame:
    """載入整理後的 AI4I CSV，並套用固定欄位型別。"""
    df = pd.read_csv(CSV_DATA)
    df["type"] = pd.Categorical(df["type"], categories=TYPE_ORDER, ordered=True)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """回傳加入物理衍生特徵的副本（不改變原始欄位）。"""
    out = df.copy()
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * np.pi / 60
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["wear_torque"] = out["tool_wear_min"] * out["torque_nm"]
    return out


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover
            pass
