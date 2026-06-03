"""Run OCR over an image directory, caching results to JSON so we never re-bill.

Used by Arm-C calibration and any eval step that needs raw Azure OCR over a
corpus. Cache lives next to the images at `<dir>_ocr/`.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from miso.ocr import make_ocr

log = logging.getLogger(__name__)

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _load_env() -> None:
    env = Path("miso/.env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_ocr_dir(img_dir: str | Path, engine: str = "azure",
                cache_dir: str | Path | None = None) -> dict[str, dict]:
    """OCR every image in `img_dir` (cached). Returns {stem: {"text", "words":[{t,c}]}}."""
    _load_env()
    logging.getLogger("azure").setLevel(logging.WARNING)  # silence per-request HTTP logs
    img_dir = Path(img_dir)
    cache = Path(cache_dir) if cache_dir else img_dir.parent / f"{img_dir.name}_ocr"
    cache.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in img_dir.glob("*")
                  if p.suffix.lower() in _EXTS and ".prepared" not in p.name)
    out: dict[str, dict] = {}
    ocr = None
    for img in imgs:
        cf = cache / f"{img.stem}.json"
        if cf.exists():
            out[img.stem] = json.loads(cf.read_text())
            continue
        if ocr is None:
            ocr = make_ocr(engine)
            log.info("running %s OCR over %s (uncached pages only) ...", engine, img_dir)
        res = ocr.run(img)
        rec = {"text": res.raw_text,
               "words": [{"t": w.text, "c": float(w.confidence)} for w in res.words]}
        cf.write_text(json.dumps(rec, ensure_ascii=False))
        out[img.stem] = rec
        log.info("  %s: %d words", img.stem, len(rec["words"]))
    return out
