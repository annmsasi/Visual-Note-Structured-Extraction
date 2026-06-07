"""Crop figure regions out of a page image and fill each figure block's `image`.

The VLM marks a figure with a normalized `bbox` — `[x, y, width, height]` in 0–1
page coordinates — and leaves `image` empty (see `miso/prompts/extraction_system.md`).
This module is the deferred "later step": it turns each box into a real cropped PNG
and writes its path into the block's `image` slot, which the renderers and the Docs
API embed already know how to use.

Best-effort by design: a missing Pillow, an unreadable page image, or a degenerate
box is logged and skipped — a figure simply keeps its caption — so figure cropping
can never break extraction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def crop_figures(
    doc: dict[str, Any],
    page_image_path: Path | str,
    out_dir: Path | str,
    *,
    note_id: str = "note",
    pad: float = 0.02,
) -> dict[str, Any]:
    """Crop every figure block that carries a usable `bbox` from the page image,
    writing one PNG per figure and setting that block's `image` to the file path.

    Mutates and returns `doc`. Crops land in `<out_dir>/<note_id>/figure_<n>.png`.
    """
    figures = [b for b in (doc.get("blocks") or [])
               if b.get("type") == "figure" and _valid_bbox(b.get("bbox"))]
    if not figures:
        return doc
    try:
        from PIL import Image
    except ImportError:
        log.warning("Pillow not installed; %d figure(s) left without images", len(figures))
        return doc
    try:
        img = Image.open(page_image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        log.warning("cannot open page image %s: %s", page_image_path, e)
        return doc

    dest = Path(out_dir) / note_id
    dest.mkdir(parents=True, exist_ok=True)
    cropped = 0
    for i, block in enumerate(figures):
        box = _pixel_box(block["bbox"], img.width, img.height, pad)
        path = dest / f"figure_{i}.png"
        img.crop(box).save(path, "PNG")
        block["image"] = str(path)
        cropped += 1
    log.info("cropped %d figure(s) from %s -> %s/", cropped, page_image_path, dest)
    return doc


def _valid_bbox(v: Any) -> bool:
    """A bbox usable for cropping: four numbers with positive width and height."""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return False
    try:
        _x, _y, w, h = (float(n) for n in v)
    except (TypeError, ValueError):
        return False
    return w > 0 and h > 0


def _pixel_box(bbox: list[float], width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    """Normalized `[x, y, w, h]` → integer pixel box `(left, top, right, bottom)`.

    The box is padded (VLM coordinates are imprecise) then clamped to the page, and
    is guaranteed at least 1×1 so `Image.crop` never returns an empty image.
    """
    x, y, w, h = (float(n) for n in bbox)
    left = max(0.0, x - pad)
    top = max(0.0, y - pad)
    right = min(1.0, x + w + pad)
    bottom = min(1.0, y + h + pad)
    px0, py0 = int(left * width), int(top * height)
    px1, py1 = int(round(right * width)), int(round(bottom * height))
    return px0, py0, max(px1, px0 + 1), max(py1, py0 + 1)
