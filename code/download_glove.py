"""
下载 100 维预训练 GloVe（与 Stanford glove.6B.100d 同源）。
优先：gensim-data 的较小 gz（经 ghfast 镜像）；备选：官方 zip。
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

EMB_DIR = Path(__file__).resolve().parent / "embeddings"
GLOVE_TXT = EMB_DIR / "glove.6B.100d.txt"
GLOVE_GZ = EMB_DIR / "glove-wiki-gigaword-100.gz"
ZIP_PATH = EMB_DIR / "glove.6B.zip"

GZ_URLS = [
    "https://ghfast.top/https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz",
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz",
]

ZIP_URLS = [
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://hf-mirror.com/stanfordnlp/glove/resolve/main/glove.6B.zip",
]


def _download(url: str, dest: Path, min_bytes: int) -> None:
    print(f"[info] 下载: {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    size = dest.stat().st_size
    print(f"[info] 完成 size={size}")
    if size < min_bytes:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"文件过小: {size}")


def main() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    if GLOVE_TXT.exists() and GLOVE_TXT.stat().st_size > 100_000_000:
        print(f"[done] 已有 txt: {GLOVE_TXT}")
        return
    if GLOVE_GZ.exists() and GLOVE_GZ.stat().st_size > 50_000_000:
        print(f"[done] 已有 gz: {GLOVE_GZ}")
        return

    # 1) 先试较小的 gensim gz（~128MB）
    for url in GZ_URLS:
        try:
            _download(url, GLOVE_GZ, min_bytes=50_000_000)
            print(f"[done] → {GLOVE_GZ}")
            return
        except Exception as e:
            print(f"[warn] gz 失败: {e}")
            GLOVE_GZ.unlink(missing_ok=True)

    # 2) 官方 zip，只解压 100d
    for url in ZIP_URLS:
        try:
            _download(url, ZIP_PATH, min_bytes=200_000_000)
            print("[info] 解压 glove.6B.100d.txt ...")
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                with zf.open("glove.6B.100d.txt") as src, GLOVE_TXT.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
            ZIP_PATH.unlink(missing_ok=True)
            print(f"[done] → {GLOVE_TXT}")
            return
        except Exception as e:
            print(f"[warn] zip 失败: {e}")
            ZIP_PATH.unlink(missing_ok=True)

    raise SystemExit("GloVe 下载失败，请检查网络或手动放到 code/embeddings/")


if __name__ == "__main__":
    main()
