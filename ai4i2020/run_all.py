"""一鍵重現 AI4I 2020 分析流程：依檔名順序執行 src/NN_*.py。"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def discover_scripts() -> list[Path]:
    return sorted(
        (ROOT / "src").glob("[0-9][0-9]_*.py"),
        key=lambda p: p.name,
    )


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover
            pass


def main() -> None:
    _force_utf8()
    scripts = discover_scripts()
    if not scripts:
        raise SystemExit("找不到 src/NN_*.py 腳本。")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    t0 = time.perf_counter()
    for path in scripts:
        print(f"\n{'#' * 70}\n# 執行 {path.name}\n{'#' * 70}")
        subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env, check=True)
    print(f"\n全部完成，共 {len(scripts)} 個步驟，總耗時 {time.perf_counter() - t0:.1f} 秒。")


if __name__ == "__main__":
    main()
