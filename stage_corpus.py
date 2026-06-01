"""Download a HuggingFace notes dataset and render its pages to images.

  python stage_corpus.py HumynLabs/Handwritten-Computer-Science-Notes-Dataset corpora/cs cs
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz
from datasets import load_dataset

try:
    from datasets import Pdf
except ImportError:
    from datasets.features import Pdf


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: python stage_corpus.py <hf_repo> <out_dir> <prefix>")
        return 2
    repo, out_dir, prefix = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo, split="train").cast_column("pdf", Pdf(decode=False))
    pages = 0
    for i, row in enumerate(ds):
        cell = row["pdf"]
        data = (cell["bytes"] if isinstance(cell, dict) and cell.get("bytes")
                else open(cell["path"], "rb").read())
        doc = fitz.open(stream=data, filetype="pdf")
        for page_no in range(len(doc)):
            doc[page_no].get_pixmap(dpi=200).save(str(out_dir / f"{prefix}_{i:02d}_{page_no:02d}.png"))
            pages += 1
    print(f"wrote {pages} page images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
