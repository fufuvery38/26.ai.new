"""05 - 將 Markdown 小論文轉為可攜式 HTML（圖片內嵌、可列印為 PDF）。"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

from common import PAPER_DIR, ROOT, force_utf8_stdout

SRC = PAPER_DIR / "iris_nn_paper.md"
OUT = PAPER_DIR / "iris_nn_paper.html"
PDF = PAPER_DIR / "iris_nn_paper.pdf"

CSS = """
:root {
  --ink: #1f2933;
  --muted: #5b6773;
  --line: #d8dee6;
  --accent: #1f4e79;
  --soft: #f4f7fa;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0 auto;
  max-width: 860px;
  padding: 48px 56px 80px;
  color: var(--ink);
  background: #ffffff;
  font-family: "Microsoft JhengHei", "Noto Sans TC", "PingFang TC", "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.85;
  text-align: justify;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
h1 {
  font-size: 1.85rem;
  line-height: 1.45;
  text-align: center;
  margin: 18px 0 10px;
  letter-spacing: 0.02em;
}
h1 + p { text-align: center; font-size: 1.02rem; color: #33414f; margin: 0 0 14px; }
h1 + p + p { text-align: center; color: var(--muted); font-size: 0.92rem; margin: 0 0 20px; }
h2 {
  font-size: 1.34rem;
  color: var(--accent);
  margin: 2.4em 0 0.8em;
  padding-bottom: 0.28em;
  border-bottom: 2px solid var(--line);
  break-after: avoid;
}
h3 {
  font-size: 1.08rem;
  margin: 1.8em 0 0.55em;
  color: #243b53;
  break-after: avoid;
}
p { margin: 0.65em 0; }
hr { border: none; border-top: 1px solid var(--line); margin: 2em 0; }
blockquote {
  margin: 1.2em 0;
  padding: 0.7em 1em;
  background: var(--soft);
  border-left: 4px solid var(--accent);
  color: #33414f;
}
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 1.4em auto 0.45em;
  border: 1px solid #e3e8ee;
  border-radius: 6px;
  background: #fff;
}
p.caption {
  text-align: center;
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.7;
  margin: 0.25em auto 1.8em;
  max-width: 92%;
}
figure {
  margin: 1.4em auto 1.8em;
  break-inside: avoid;
  page-break-inside: avoid;
}
figure img { margin: 0 auto 0.5em; }
figcaption {
  text-align: center;
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.7;
  max-width: 92%;
  margin: 0 auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.1em 0 1.6em;
  font-size: 0.88rem;
  line-height: 1.55;
  break-inside: avoid;
}
th, td { border: 1px solid var(--line); padding: 6px 9px; vertical-align: top; }
th { background: #eef3f8; color: #1c3957; font-weight: 700; text-align: center; }
tr:nth-child(even) td { background: #fbfcfd; }
td { text-align: left; }
code {
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 0.86em;
  background: #f1f4f7;
  padding: 1px 5px;
  border-radius: 4px;
}
pre {
  background: #f6f8fa;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  overflow-x: auto;
  line-height: 1.55;
  break-inside: avoid;
}
pre code { background: none; padding: 0; }
a { color: #1a5d9c; text-decoration: none; }
a:hover { text-decoration: underline; }
strong { color: #12263a; }
.toc {
  background: var(--soft);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px 22px 12px 8px;
  margin: 1.5em 0 2em;
  font-size: 0.92rem;
}
.toc::before {
  content: "目　　錄";
  display: block;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.15em;
  margin: 0 0 6px 14px;
}
.toc ul { margin: 0; padding-left: 1.6em; }
.toc li { margin: 2px 0; }
.toc > ul > li > a { color: #243b53; }
.toc ul ul { font-size: 0.88rem; color: var(--muted); }

.print-btn {
  position: fixed;
  right: 28px;
  bottom: 28px;
  z-index: 10;
  padding: 11px 20px;
  border: none;
  border-radius: 999px;
  background: #1f4e79;
  color: #fff;
  font-family: inherit;
  font-size: 0.92rem;
  cursor: pointer;
  box-shadow: 0 6px 18px rgba(31, 78, 121, 0.28);
}
.print-btn:hover { background: #163a5a; }

@page {
  size: A4;
  margin: 18mm 16mm 20mm;
}
@media print {
  body { max-width: none; padding: 0; font-size: 10.8pt; line-height: 1.75; }
  h1 { font-size: 20pt; margin-top: 0; }
  h2 { font-size: 14pt; margin-top: 1.5em; }
  h3 { font-size: 11.6pt; }
  img { max-height: 108mm; object-fit: contain; }
  figure { break-inside: avoid; page-break-inside: avoid; }
  figcaption { font-size: 8.6pt; }
  .print-btn { display: none; }
  a { color: inherit; }
  .toc { break-after: page; }
}
"""


def embed_images(html: str) -> str:
    def repl(match: re.Match) -> str:
        alt, src = match.group(1), match.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (PAPER_DIR / src).resolve()
        if not path.exists():
            print(f"[警告] 找不到圖片：{path}")
            return match.group(0)
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "svg": "image/svg+xml", "webp": "image/webp"}.get(path.suffix.lower().lstrip("."), "image/png")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        print(f"[內嵌] {path.name} ({len(data) // 1024} KB base64)")
        return f"![{alt}](data:{mime};base64,{data})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, html)


def build_pdf() -> None:
    """若系統有 Edge/Chrome，使用無頭模式列印 A4 PDF（可選步驟）。"""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    browser = next((p for p in candidates if Path(p).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if not browser:
        print("[略過] 未找到 Edge/Chrome，PDF 可由瀏覽器開啟 HTML 後列印。")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_html = Path(tmp) / "iris_paper_print.html"
        shutil.copyfile(OUT, tmp_html)
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={PDF}",
            tmp_html.as_uri(),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=180, text=True)
            if PDF.exists() and PDF.stat().st_size > 100_000:
                print(f"[完成] {PDF.relative_to(ROOT)}（{PDF.stat().st_size / 1024 / 1024:.2f} MB）")
            else:
                print(f"[警告] PDF 產生失敗（exit={proc.returncode}）：{proc.stderr[-300:]}")
        except Exception as exc:
            print(f"[略過] 無法以無頭瀏覽器產生 PDF：{exc}")


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    # 在摘要前插入目錄
    text = text.replace("## 摘要", "[TOC]\n\n---\n\n## 摘要", 1)
    text = embed_images(text)

    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "md_in_html"],
        extension_configs={"toc": {"toc_depth": "2", "permalink": False}},
    )
    # 將「圖 N」「表 N」說明段落轉為 caption 樣式
    html_body = re.sub(r"<p><em>((?:圖|表)\s.*?)</em></p>", r'<p class="caption">\1</p>', html_body, flags=re.S)
    # 將圖片與其說明包成 figure，避免列印時被分頁拆開
    html_body = re.sub(
        r'<p><img ([^>]*?)></p>\s*<p class="caption">(.*?)</p>',
        r"<figure><img \1><figcaption>\2</figcaption></figure>",
        html_body,
        flags=re.S,
    )
    # 將開頭的獨立 --- 移除（避免封面出現多餘分隔線）
    html_body = re.sub(r"^(<hr\s*/?>\s*)+", "", html_body, count=1)

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>以主成分分析與多層感知器於 Iris 資料集之分類研究</title>
<meta name="author" content="26.ai.new 專案團隊">
<meta name="description" content="Iris 資料集之探索分析、PCA 三維視覺化與多層感知器分類小論文">
<style>{CSS}</style>
</head>
<body>
{html_body}
<button class="print-btn" onclick="window.print()">列印 / 另存 PDF</button>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"[完成] {OUT.relative_to(ROOT)}（{size_mb:.2f} MB，圖片已內嵌，可離線閱讀）")
    build_pdf()


if __name__ == "__main__":
    force_utf8_stdout()
    main()
