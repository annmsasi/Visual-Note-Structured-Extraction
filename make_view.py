"""Build a side-by-side HTML viewer (original vs boxed words) from ocr_dump.json."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    dump = json.loads((HERE / "ocr_dump.json").read_text())
    page = dump["pages"][0]
    W, H = page["width"], page["height"]
    words = page["words"]

    boxes = []
    for w in words:
        p = w["polygon"]
        if len(p) < 8:
            continue
        xs, ys = p[0::2], p[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        boxes.append({
            "t": w["text"], "c": round(w["confidence"], 2),
            "l": x0 / W * 100, "tp": y0 / H * 100,
            "w": (x1 - x0) / W * 100, "h": (y1 - y0) / H * 100,
        })

    data = json.dumps(boxes)
    low = sum(b["c"] < 0.70 for b in boxes)
    mean = sum(b["c"] for b in boxes) / len(boxes)
    stats = f"{len(boxes)} words &middot; mean {mean:.2f} &middot; {low} below 0.70"

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>OCR side-by-side</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #1a1a1a; color: #eee; }}
  header {{ padding: 10px 16px; background: #111; position: sticky; top: 0; }}
  header b {{ color: #fff; }} header span {{ color: #aaa; margin-left: 10px; }}
  .legend {{ float: right; }}
  .legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px;
               margin: 0 4px 0 12px; vertical-align: -1px; }}
  .panes {{ display: flex; gap: 12px; padding: 12px; align-items: flex-start; }}
  .pane {{ flex: 1; min-width: 0; }}
  .pane h2 {{ font-size: 13px; color: #aaa; margin: 0 0 6px; font-weight: 600; }}
  .stage {{ position: relative; width: 100%; }}
  .stage img {{ width: 100%; display: block; image-orientation: from-image; border-radius: 4px; }}
  .anno img {{ filter: brightness(0.45) grayscale(0.3); }}
  .box {{ position: absolute; border: 1px solid; border-radius: 2px;
          display: flex; align-items: center; justify-content: center;
          overflow: hidden; cursor: default; }}
  .box span {{ font-size: 9px; line-height: 1; padding: 0 1px; white-space: nowrap;
               text-shadow: 0 0 2px #000, 0 0 2px #000; }}
  .box:hover {{ z-index: 9; box-shadow: 0 0 0 2px #fff; }}
  .box:hover::after {{ content: attr(data-c); position: absolute; top: -14px; left: 0;
                       background: #fff; color: #000; font-size: 9px; padding: 0 3px;
                       border-radius: 2px; }}
</style>
<header>
  <b>OCR side-by-side</b><span>{dump['image']} &middot; {stats}</span>
  <span class="legend">confidence
    <i style="background:#d33"></i>&lt;.5
    <i style="background:#e90"></i>.5–.7
    <i style="background:#dd0"></i>.7–.9
    <i style="background:#3c3"></i>&gt;.9
  </span>
</header>
<div class="panes">
  <div class="pane"><h2>Original</h2>
    <div class="stage"><img src="{dump['image']}"></div></div>
  <div class="pane"><h2>Recognized words (hover for confidence)</h2>
    <div class="stage anno" id="anno"><img src="{dump['image']}"></div></div>
</div>
<script>
const WORDS = {data};
function color(c) {{
  if (c < 0.5) return '#d33';
  if (c < 0.7) return '#e90';
  if (c < 0.9) return '#dd0';
  return '#3c3';
}}
const stage = document.getElementById('anno');
for (const w of WORDS) {{
  const d = document.createElement('div');
  d.className = 'box';
  d.style.left = w.l + '%'; d.style.top = w.tp + '%';
  d.style.width = w.w + '%'; d.style.height = w.h + '%';
  d.style.borderColor = color(w.c);
  d.style.background = color(w.c) + '33';
  d.dataset.c = w.t + '  ' + w.c.toFixed(2);
  const s = document.createElement('span');
  s.textContent = w.t; s.style.color = color(w.c);
  d.appendChild(s);
  stage.appendChild(d);
}}
</script>
"""
    out = HERE / "ocr_view.html"
    out.write_text(html)
    print(f"wrote {out.name} ({len(boxes)} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
