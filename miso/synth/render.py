"""Render note text to a degraded handwritten page image.

Layout + per-line jitter/slant (Pillow) gives the handwriting surface; Augraphy
applies the paper/ink/scan degradation that produces *realistic, non-uniform* OCR
errors (the main realism lever — eval_design_v1.md §3.3). Deterministic per seed
so the corpus is reproducible. One consistent font per course = one "hand".
"""
from __future__ import annotations

import logging
import random
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

log = logging.getLogger(__name__)

PAGE_W, PAGE_H = 1700, 2200
MARGIN = 110
_PAPER = (252, 250, 244)


def _wrap_lines(text: str, width_chars: int) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for raw in text.split("\n"):
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if not stripped:
            out.append((indent, ""))
            continue
        wrapped = textwrap.wrap(stripped, width=max(12, width_chars - indent // 2)) or [""]
        for i, seg in enumerate(wrapped):
            out.append((indent if i == 0 else indent + 2, seg))
    return out


def _render_clean(text: str, font_path: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (PAGE_W, PAGE_H), _PAPER)
    size = rng.randint(40, 46)
    font = ImageFont.truetype(font_path, size)
    line_h = int(size * 1.7)
    avg_w = font.getlength("n") or size * 0.5
    width_chars = int((PAGE_W - 2 * MARGIN) / avg_w)
    y = MARGIN
    for indent, seg in _wrap_lines(text, width_chars):
        if y > PAGE_H - MARGIN:
            break
        if not seg:
            y += line_h // 2
            continue
        x = MARGIN + indent * 14
        bbox = font.getbbox(seg)
        lw, lh = int(bbox[2] - bbox[0]) + 24, int(bbox[3] - bbox[1]) + 24
        layer = Image.new("RGBA", (max(lw, 1), max(lh, 1)), (0, 0, 0, 0))
        ink = rng.randint(15, 55)
        ImageDraw.Draw(layer).text(
            (12 - bbox[0], 12 - bbox[1]), seg, font=font,
            fill=(ink, ink, min(255, ink + rng.randint(0, 20)), 255),
        )
        layer = layer.rotate(rng.uniform(-1.8, 1.8), expand=True, resample=Image.BICUBIC)
        img.paste(layer, (x, y + rng.randint(-4, 4)), layer)
        y += line_h + rng.randint(-3, 6)
    return img


def _degrade(img: Image.Image, seed: int, strength: float = 1.0) -> Image.Image:
    """Controlled, deterministic optical degradation that keeps the page LEGIBLE
    but imperfect — soft strokes + blur + paper tint/noise. Stroke-softening
    (downscale→upscale + blur) is the OCR-error inducer: a clean font render is
    trivial for a modern OCR/VLM (the 'too-clean' threat, eval_design_v1.md T1),
    while over-degrading destroys the signal entirely. `strength` (0..~1.5) is the
    calibration knob, tuned against the real anchor (Arm C). 1.0 is a moderate
    default — aim for OCR CER comparable to real handwriting, not near-100%.
    """
    rng = random.Random(seed)
    w, h = img.size
    img = img.rotate(rng.uniform(-1.2, 1.2) * strength, resample=Image.BICUBIC,
                     fillcolor=_PAPER, expand=False)
    f = 1.0 - 0.18 * strength * rng.uniform(0.7, 1.0)  # mild stroke softening
    img = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR).resize(
        (w, h), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.0) * strength))
    img = ImageEnhance.Contrast(img).enhance(1.0 - 0.10 * strength)

    a = np.array(img).astype("float32")
    rs = np.random.RandomState(rng.randint(0, 1 << 30))
    # low-frequency paper tint (gentle blotches) + fine gaussian noise
    lf = rs.normal(0, 1, (max(2, h // 40), max(2, w // 40)))
    lf = np.asarray(Image.fromarray(
        ((lf - lf.min()) / (np.ptp(lf) + 1e-6) * 255).astype("uint8")
    ).resize((w, h), Image.BILINEAR), dtype="float32") / 255.0
    a *= (0.95 + 0.05 * lf)[..., None]
    a += rs.normal(0, 4 * strength, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def _elastic(img: Image.Image, rng: random.Random, alpha: float) -> Image.Image:
    """Per-pixel elastic warp (smoothed random displacement field). This makes the
    uniform font's glyphs *irregular* — the actual driver of realistic handwriting
    OCR errors (substitutions/deletions), which blur alone cannot produce because a
    regular font stays legible under blur (eval_design_v1.md §3.3 / T1)."""
    a = np.asarray(img)
    h, w = a.shape[:2]

    def field() -> np.ndarray:
        small = np.random.RandomState(rng.randint(0, 1 << 30)).uniform(
            -1, 1, (max(2, h // 24), max(2, w // 24))).astype("float32")
        big = Image.fromarray(((small + 1) * 127.5).astype("uint8")).resize(
            (w, h), Image.BILINEAR)
        return np.asarray(big, dtype="float32") / 127.5 - 1.0

    xs = np.clip(np.arange(w)[None, :] + field() * alpha, 0, w - 1).astype(np.intp)
    ys = np.clip(np.arange(h)[:, None] + field() * alpha, 0, h - 1).astype(np.intp)
    return Image.fromarray(a[ys, xs])


def render_note(text: str, font_path: str, out_path: str, seed: int = 0,
                degrade: bool = True, strength: float = 1.0) -> str:
    img = _render_clean(text, font_path, seed)
    if degrade:
        img = _elastic(img, random.Random(seed * 7 + 1), alpha=min(12.0, 5.0 * strength))
        img = _degrade(img, seed, strength)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, quality=90)
    return out_path
