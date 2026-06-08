"""Quick figures eval — standalone and decoupled from the text metrics.

Measures the figure-extraction sub-tasks the text metrics don't:

  * DETECTION  — page-level precision / recall / F1 plus a per-page figure-count
                 error, scored against the `[figure]` placeholders already in the
                 gold. Deterministic and free (no API, no annotated boxes).

  * LOCALIZATION — optional VLM "crop hit-rate": does each cropped image actually
                 contain a figure (rather than only text / blank)? A cheap proxy
                 for bbox quality that needs no annotated boxes.

This module is deliberately ADDITIVE: it does not import or modify metrics.py's
text functions (cer / term_recall / …) or analyze.py's scoring loop, so adding it
cannot change any text metric. The single read-only import is `bootstrap_ci`, used
for a confidence interval on the per-page count error.

Gold figures are counted from the `## Structured note` section only — the `[figure]`
token also appears in `## Transcription`, and counting both would double it.

    # detection only (free) — <sys> holds the pipeline's per-page IR <note_id>.json:
    python -m miso.eval.figure_eval --gold corpora/cse138_gold --sys out/
    # add the crop hit-rate judge (one vision call per cropped figure):
    python -m miso.eval.figure_eval --gold corpora/cse138_gold --sys out/ --judge claude-opus-4-8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FIGURE_PLACEHOLDER = "[figure]"

try:  # read-only; used only for a CI on the count error. Never modifies text metrics.
    from miso.eval.metrics import bootstrap_ci
except Exception:  # pragma: no cover
    bootstrap_ci = None


# --------------------------------------------------------------------------- counting

# Only these top-level headers delimit gold sections. Any OTHER `## ` line is a
# content heading the drafter wrote INSIDE a section (e.g. `## State & Events`), so
# it must NOT end the section. Drafts also sometimes repeat a header (`## Transcription
# (verbatim)` then `## Transcription`); both map to the same canonical section.
_KNOWN_SECTIONS = ("transcription", "structured note", "distinctive terms")


def _sections(md: str) -> dict[str, str]:
    """Split gold markdown into {canonical_section: text}, robust to repeated section
    headers and to `## ` content headings nested inside a section."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in md.splitlines():
        if line.startswith("## "):
            head = line[3:].strip().lower()
            canon = next((k for k in _KNOWN_SECTIONS if head.startswith(k)), None)
            if canon is not None:
                current = canon
                out.setdefault(canon, [])
                continue
            # unknown `## ` heading → content of the current section, not a boundary
        if current is not None:
            out.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def count_gold_figures(md: str) -> int:
    """Gold figure count for a page: the MAX of the `[figure]` tokens in the
    `## Transcription` and `## Structured note` sections.

    Max (not sum) dedupes the figure that appears in BOTH sections, while still
    catching pages where the Opus draft put `[figure]` in only one section — the
    drafts are internally inconsistent about this, so neither section alone is a
    reliable count. Still silver gold: human-verify the per-page counts for a
    headline number.
    """
    secs = _sections(md)
    return max(secs.get("transcription", "").count(FIGURE_PLACEHOLDER),
               secs.get("structured note", "").count(FIGURE_PLACEHOLDER))


def count_sys_figures(doc: dict[str, Any]) -> int:
    """Number of `figure` blocks in a system document IR."""
    return sum(1 for b in (doc.get("blocks") or []) if b.get("type") == "figure")


def load_gold_figures(gold_dir: Path) -> dict[str, int]:
    """Map note_id -> gold figure count, reading `<gold_dir>/<note_id>/<note_id>.md`.

    Only the canonical per-page file (stem == its parent dir) is read, so stray
    alternates like `002.md` next to `cse138-002.md` are ignored.
    """
    out: dict[str, int] = {}
    for md in sorted(gold_dir.glob("*/*.md")):
        if md.stem != md.parent.name:
            continue
        out[md.stem] = count_gold_figures(md.read_text())
    return out


def load_sys_figures(sys_dir: Path) -> tuple[dict[str, int], dict[str, dict]]:
    """Map note_id -> (figure count, full IR) from `<sys_dir>/<note_id>.json`."""
    counts: dict[str, int] = {}
    docs: dict[str, dict] = {}
    for path in sorted(sys_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        counts[path.stem] = count_sys_figures(doc)
        docs[path.stem] = doc
    return counts, docs


# --------------------------------------------------------------------------- detection

def detection_scores(gold: dict[str, int], sys: dict[str, int]) -> dict[str, Any]:
    """Page-level detection P/R/F1 (a page is positive if it has >=1 figure) plus the
    per-page absolute count error. Evaluated over the union of note ids; a missing
    side counts as zero figures."""
    note_ids = sorted(set(gold) | set(sys))
    tp = fp = fn = 0
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for nid in note_ids:
        g, s = gold.get(nid, 0), sys.get(nid, 0)
        if g > 0 and s > 0:
            tp += 1
        elif g == 0 and s > 0:
            fp += 1
        elif g > 0 and s == 0:
            fn += 1
        errors.append(abs(g - s))
        rows.append({"note_id": nid, "gold": g, "sys": s, "abs_err": abs(g - s)})
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mae = sum(errors) / len(errors) if errors else 0.0
    result = {
        "pages": len(note_ids),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "count_mae": mae,
        "gold_total": sum(gold.values()), "sys_total": sum(sys.values()),
        "rows": rows,
    }
    if bootstrap_ci is not None and errors:
        result["count_mae_ci"] = bootstrap_ci(errors)[1:]   # (lo, hi)
    return result


# --------------------------------------------------------------------------- crop judge

_JUDGE_PROMPT = (
    "This image is a region cropped from a page of handwritten lecture notes. Does it "
    "primarily contain a figure — a diagram, chart, graph, circuit, plot, or drawing — "
    "as opposed to only handwritten text or blank space? Answer with one word: yes or no."
)


def judge_crop(image_path: Path, model: str, client) -> bool | None:
    """VLM yes/no: does the crop actually contain a figure? None on any failure."""
    import base64

    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > 1500:                       # keep the judge call cheap
            s = 1500 / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)))
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, "JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        msg = client.messages.create(
            model=model, max_tokens=8,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": _JUDGE_PROMPT},
            ]}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return text.strip().lower().startswith("y")
    except Exception as e:  # pragma: no cover - network/SDK path
        log.warning("crop judge failed for %s: %s", image_path, e)
        return None


def crop_hit_rate(docs: dict[str, dict], model: str) -> dict[str, Any]:
    """Fraction of cropped figure images the VLM judges to actually contain a figure."""
    import os

    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    hits = total = skipped = 0
    misses: list[str] = []
    for nid, doc in docs.items():
        for b in doc.get("blocks") or []:
            if b.get("type") != "figure":
                continue
            image = (b.get("image") or "").strip()
            if not image or not Path(image).exists():
                skipped += 1
                continue
            verdict = judge_crop(Path(image), model, client)
            if verdict is None:
                skipped += 1
                continue
            total += 1
            if verdict:
                hits += 1
            else:
                misses.append(image)
    return {"hits": hits, "total": total, "skipped": skipped,
            "hit_rate": (hits / total if total else None), "misses": misses}


# --------------------------------------------------------------------------- CLI

def _print_report(det: dict[str, Any], crop: dict[str, Any] | None) -> None:
    print("\nFigure detection (per page):")
    print(f"  {'note_id':<16} {'gold':>4} {'sys':>4}  {'ok':>3}")
    for r in det["rows"]:
        ok = "·" if r["gold"] == 0 and r["sys"] == 0 else ("✓" if r["gold"] == r["sys"] else "✗")
        print(f"  {r['note_id']:<16} {r['gold']:>4} {r['sys']:>4}  {ok:>3}")
    print(f"\n  pages={det['pages']}  TP={det['tp']} FP={det['fp']} FN={det['fn']}")
    print(f"  precision={det['precision']:.3f}  recall={det['recall']:.3f}  F1={det['f1']:.3f}")
    mae = f"{det['count_mae']:.3f}"
    if det.get("count_mae_ci"):
        lo, hi = det["count_mae_ci"]
        mae += f"  (95% CI {lo:.3f}–{hi:.3f})"
    print(f"  figure-count MAE={mae}")
    print(f"  total figures: gold={det['gold_total']}  detected={det['sys_total']}")
    if crop is not None:
        rate = "n/a" if crop["hit_rate"] is None else f"{crop['hit_rate']:.3f}"
        print(f"\nCrop hit-rate (VLM): {crop['hits']}/{crop['total']} = {rate}"
              f"   (skipped {crop['skipped']})")
        for m in crop["misses"]:
            print(f"  miss: {m}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Quick figures eval (detection + optional crop judge).")
    ap.add_argument("--gold", required=True, type=Path,
                    help="gold dir of <note_id>/<note_id>.md")
    ap.add_argument("--sys", type=Path,
                    help="dir of system IR <note_id>.json (omit to just print gold counts)")
    ap.add_argument("--judge", metavar="MODEL",
                    help="also run the crop hit-rate judge with this vision model")
    args = ap.parse_args(argv)

    gold = load_gold_figures(args.gold)
    if args.sys is None:
        print("Gold figure counts (no --sys given):")
        for nid in sorted(gold):
            print(f"  {nid:<16} {gold[nid]}")
        print(f"  total: {sum(gold.values())} figures across {len(gold)} pages")
        return 0

    sys_counts, sys_docs = load_sys_figures(args.sys)
    det = detection_scores(gold, sys_counts)
    crop = None
    if args.judge:
        _load_env_if_present()
        crop = crop_hit_rate(sys_docs, args.judge)
    _print_report(det, crop)
    return 0


def _load_env_if_present() -> None:
    """Best-effort: load a root `.env` so --judge finds ANTHROPIC_API_KEY."""
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            import os
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


if __name__ == "__main__":
    raise SystemExit(main())
