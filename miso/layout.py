"""Recover line and indentation structure from positioned OCR words."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from miso.types import OCRWord

# Vertical-centre departure (as a fraction of line height) that starts a new line.
_LINE_BREAK_RATIO = 0.6
# Left-edge difference (in median word-heights) that marks a deeper indent.
_INDENT_TOLERANCE_HEIGHTS = 1.0


@dataclass
class OCRLine:
    text: str
    depth: int                                  # 0 = leftmost
    bbox: tuple[float, float, float, float] | None  # x, y, w, h


def group_into_lines(words: list[OCRWord]) -> None:
    """Assign `line_id` to each word in reading order, top to bottom. Mutates in place."""
    geo = [w for w in words if w.bbox is not None]
    if not geo:
        for w in words:
            w.line_id = 0
        return

    order = sorted(geo, key=lambda w: w.bbox[1] + w.bbox[3] / 2)
    line_id = -1
    anchor_cy: float | None = None
    line_h = 0.0
    for w in order:
        _, y, _, h = w.bbox
        cy = y + h / 2
        if anchor_cy is None or abs(cy - anchor_cy) > _LINE_BREAK_RATIO * max(line_h, h):
            line_id += 1
            anchor_cy = cy
            line_h = h
        else:
            line_h = max(line_h, h)
        w.line_id = line_id

    # Trail any words without geometry.
    for w in words:
        if w.line_id is None:
            line_id += 1
            w.line_id = line_id


def iter_lines(words: list[OCRWord]) -> list[OCRLine]:
    """Group words into logical lines with inferred indent depth, merging soft wraps."""
    phys = _physical_lines(words)
    if not phys:
        return []

    boxed_words = [w for w in words if w.bbox is not None]
    heights = [w.bbox[3] for w in boxed_words]
    widths = [w.bbox[2] for w in boxed_words]
    med_h = median(heights) if heights else 0.0
    med_w = median(widths) if widths else 0.0
    rights = [p["right"] for p in phys if p["bbox"]]
    lefts = [p["x0"] for p in phys if p["bbox"]]
    block_right = max(rights) if rights else 0.0
    gap = med_h  # rough inter-word gap estimate
    # Only merge soft wraps when there is a real text column to wrap within.
    block_width = block_right - (min(lefts) if lefts else 0.0)
    wrap_enabled = med_w > 0 and block_width > 3 * med_w

    depths = _depths_from_left_edges(
        [p["x0"] for p in phys],
        tol=(med_h * _INDENT_TOLERANCE_HEIGHTS) if med_h else 0.0,
    )

    merged: list[dict] = []
    prev: dict | None = None
    for p, depth in zip(phys, depths):
        wrapped = (
            wrap_enabled
            and prev is not None and prev["bbox"] is not None and p["bbox"] is not None
            # previous line had less room left than this line's first word needs
            and (block_right - prev["right"]) < (p["first_w"] + gap)
            # and this line doesn't dedent
            and depth >= merged[-1]["depth"]
        )
        if wrapped and merged:
            merged[-1]["text"] += " " + p["text"]
            merged[-1]["right"] = p["right"]   # carry the wrap front so chains continue
        else:
            merged.append({**p, "depth": depth})
        prev = merged[-1]

    return [OCRLine(text=m["text"], depth=m["depth"], bbox=m["bbox"]) for m in merged]


def _physical_lines(words: list[OCRWord]) -> list[dict]:
    """Build one entry per `line_id`, words left-ordered, with geometry for wrap logic."""
    if not any(w.line_id is not None for w in words):
        group_into_lines(words)

    buckets: dict[int, list[OCRWord]] = {}
    for w in words:
        buckets.setdefault(w.line_id if w.line_id is not None else 0, []).append(w)

    rows: list[dict] = []
    for lid in sorted(buckets):
        ws = sorted(buckets[lid], key=lambda w: (w.bbox[0] if w.bbox else 0.0))
        text = " ".join(w.text for w in ws)
        boxed = [w for w in ws if w.bbox]
        if boxed:
            x0 = min(w.bbox[0] for w in boxed)
            y0 = min(w.bbox[1] for w in boxed)
            right = max(w.bbox[0] + w.bbox[2] for w in boxed)
            y1 = max(w.bbox[1] + w.bbox[3] for w in boxed)
            rows.append({"x0": x0, "right": right, "text": text,
                         "bbox": (x0, y0, right - x0, y1 - y0),
                         "h": max(w.bbox[3] for w in boxed),
                         "first_w": boxed[0].bbox[2]})
        else:
            rows.append({"x0": 0.0, "right": 0.0, "text": text,
                         "bbox": None, "h": 0.0, "first_w": 0.0})
    return rows


def render_layout_text(words: list[OCRWord]) -> str:
    """Newline-joined lines, each prefixed by two spaces per indent level."""
    lines = iter_lines(words)
    if not lines:
        return ""
    return "\n".join("  " * ln.depth + ln.text for ln in lines)


def _depths_from_left_edges(x0s: list[float], *, tol: float) -> list[int]:
    """Map left edges to indent levels by snapping them to a few margin stops."""
    if not x0s or tol <= 0:
        return [0] * len(x0s)
    stops: list[float] = []
    for x in sorted(set(x0s)):
        if not stops or x - stops[-1] > tol:
            stops.append(x)
    depths = []
    for x in x0s:
        depth = sum(1 for s in stops if s <= x - tol)
        depths.append(depth)
    return depths
