"""Second pass: turn each figure block into a Mermaid diagram and render it.

The extraction pass leaves every `figure` block with a `description` (and maybe a
`bbox`) but empty `mermaid`/`image` slots. This module fills them: for each figure
it asks the same VLM — given the page image and the figure's caption/location — to
emit Mermaid source, then renders that source to a PNG with the Mermaid CLI (mmdc).

Best-effort by design, like the crop step it replaces. A figure the model declines
to diagram (`NO_MERMAID`), a missing `mmdc`, or source that won't parse simply
leaves a slot empty — the figure keeps its caption — so the Mermaid pass can never
break extraction. When `mmdc` reports an error and `repair=True`, the model is
re-prompted once with that error before the figure gives up.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from miso.prompts import load as load_prompt

log = logging.getLogger(__name__)

# The model returns this token (alone) when a figure cannot be a Mermaid diagram.
_NO_MERMAID = "NO_MERMAID"
_RULES = load_prompt("figure_mermaid")

# A "vision_text" callable: (page_image_path, prompt) -> the model's text reply.
# Each extractor exposes one (see miso/extraction.py).
VisionText = Callable[[Any, str], str]


def add_mermaid(
    doc: dict[str, Any],
    page_image_path: Path | str,
    vision_text: VisionText,
    out_dir: Path | str | None = None,
    *,
    note_id: str = "note",
    repair: bool = True,
) -> dict[str, Any]:
    """Fill every figure block's `mermaid` (and render `image`) from the page image.

    Mutates and returns `doc`. PNGs land in `<out_dir>/<note_id>/figure_<n>.png`
    with the `.mmd` source beside them; pass `out_dir=None` to fill only the Mermaid
    source and skip rendering.
    """
    figures = [b for b in (doc.get("blocks") or [])
               if b.get("type") == "figure" and (b.get("description") or "").strip()]
    if not figures:
        return doc

    dest = None
    if out_dir is not None:
        dest = Path(out_dir) / note_id
        dest.mkdir(parents=True, exist_ok=True)
    # Only render (and only attempt repair) when the CLI is actually present; without
    # it, figures still get their Mermaid source — HTML/Markdown render that directly.
    can_render = dest is not None and shutil.which("mmdc") is not None
    if dest is not None and not can_render:
        log.info("mmdc not installed; figures keep Mermaid source but render no PNG")

    drawn = 0
    for i, block in enumerate(figures):
        code = _ask_mermaid(vision_text, page_image_path, block)
        if not code:
            continue                       # NO_MERMAID / empty reply → caption only
        block["mermaid"] = code
        if dest is None:
            continue
        png = dest / f"figure_{i}.png"
        if not can_render:
            png.with_suffix(".mmd").write_text(code, encoding="utf-8")   # keep for later
            continue
        ok, err = render_mermaid(code, png)
        if not ok and repair and err:
            fixed = _ask_repair(vision_text, page_image_path, code, err)
            if fixed and fixed != code:
                block["mermaid"] = fixed
                ok, err = render_mermaid(fixed, png)
        if ok:
            block["image"] = str(png)
            drawn += 1
        elif err:
            log.warning("figure %d Mermaid did not render: %s", i, err.splitlines()[-1])
    if can_render:
        log.info("Mermaid pass: %d/%d figure(s) rendered -> %s/", drawn, len(figures), dest)
    return doc


def _ask_mermaid(vision_text: VisionText, page_image_path: Path | str,
                 block: dict[str, Any]) -> str:
    try:
        reply = vision_text(page_image_path, _build_prompt(block))
    except Exception as e:                 # a flaky model call must not break the note
        log.warning("Mermaid pass model call failed: %s", e)
        return ""
    return _clean(reply)


def _ask_repair(vision_text: VisionText, page_image_path: Path | str,
                code: str, err: str) -> str:
    prompt = (
        f"{_RULES}\n\n"
        "The Mermaid below did not render. Fix it and output only the corrected "
        f"Mermaid (no fences).\n\nRenderer error:\n{err.strip()}\n\nMermaid:\n{code}"
    )
    try:
        return _clean(vision_text(page_image_path, prompt))
    except Exception as e:
        log.warning("Mermaid repair call failed: %s", e)
        return ""


def _build_prompt(block: dict[str, Any]) -> str:
    parts = [_RULES, "", f"Figure description:\n{block.get('description', '')}"]
    bbox = block.get("bbox")
    if bbox:
        parts.append(f"\nApproximate location, normalized [x, y, width, height]:\n{bbox}")
    return "\n".join(parts)


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _clean(text: str) -> str:
    """Strip Markdown code fences and drop the NO_MERMAID sentinel."""
    text = (text or "").strip()
    text = _FENCE.sub("", text).strip()       # leading ```mermaid / trailing ```
    text = _FENCE.sub("", text).strip()       # in case both ends were fenced
    if not text or text.upper().startswith(_NO_MERMAID):
        return ""
    return text


def render_mermaid(code: str, out_path: Path | str) -> tuple[bool, str]:
    """Render Mermaid source to a PNG with the Mermaid CLI. Returns (ok, error).

    A missing `mmdc` is a clean `(False, "<msg>")`, never an exception, so callers
    skip rendering without special-casing it. The `.mmd` source is written beside the
    PNG so a figure that won't render can be inspected.
    """
    out_path = Path(out_path)
    mmd = out_path.with_suffix(".mmd")
    mmd.write_text(code, encoding="utf-8")
    mmdc = shutil.which("mmdc")
    if not mmdc:
        return False, "mmdc (mermaid-cli) not installed"
    cmd = [mmdc, "-i", str(mmd), "-o", str(out_path), "-b", "white", "-t", "default"]
    cfg = Path("puppeteer-config.json")       # lets mmdc run Chromium with --no-sandbox
    if cfg.exists():
        cmd += ["-p", str(cfg)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if proc.returncode == 0 and out_path.exists():
        return True, ""
    return False, (proc.stderr or proc.stdout or "mmdc failed").strip()
