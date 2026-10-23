"""一鍵重現：依序執行資料取得、EDA、PCA 與神經網路實驗。"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ["01_get_data.py", "02_eda.py", "03_pca3d.py", "04_train_nn.py", "05_build_paper.py"]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    t0 = time.perf_counter()
    for name in SCRIPTS:
        path = ROOT / "src" / name
        print(f"\n{'#' * 70}\n# 執行 {name}\n{'#' * 70}")
        subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env, check=True)
    print(f"\n全部完成，總耗時 {time.perf_counter() - t0:.1f} 秒。")


if __name__ == "__main__":
    main()
