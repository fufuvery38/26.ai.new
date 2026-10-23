"""共用設定：路徑、隨機種子、繪圖風格與資料載入。"""

from __future__ import annotations

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

RAW_DATA = DATA_DIR / "iris_raw.data"
BEZDEK_DATA = DATA_DIR / "iris_bezdek.data"
NAMES_FILE = DATA_DIR / "iris.names"
CSV_DATA = DATA_DIR / "iris.csv"

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
SEED = 42
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURES_ZH = {
    "sepal_length": "花萼長度",
    "sepal_width": "花萼寬度",
    "petal_length": "花瓣長度",
    "petal_width": "花瓣寬度",
}
SPECIES = ["setosa", "versicolor", "virginica"]
SPECIES_ZH = {"setosa": "山鳶尾", "versicolor": "變色鳶尾", "virginica": "維吉尼亞鳶尾"}
PALETTE = {"setosa": "#1f77b4", "versicolor": "#ff7f0e", "virginica": "#2ca02c"}


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


def load_iris_df() -> pd.DataFrame:
    """載入專案內的 Iris CSV，並依 SPECIES 順序排序。"""
    df = pd.read_csv(CSV_DATA)
    df["species"] = pd.Categorical(df["species"], categories=SPECIES, ordered=True)
    return df


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover
            pass
