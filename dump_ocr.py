"""Run Azure prebuilt-read and dump page and per-word geometry to ocr_dump.json."""
from __future__ import annotations

import json
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    here = Path(__file__).parent
    _load_dotenv(here / ".env")
    image = here / "Journal-example-1-2378482629.jpg"

    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(
        os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
        AzureKeyCredential(os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]),
    )
    with open(image, "rb") as fh:
        poller = client.begin_analyze_document(
            "prebuilt-read", body=fh, content_type="application/octet-stream"
        )
    result = poller.result()

    pages = []
    for page in (result.pages or []):
        words = [
            {"text": w.content,
             "confidence": float(getattr(w, "confidence", 0.0) or 0.0),
             "polygon": list(getattr(w, "polygon", []) or [])}
            for w in (page.words or [])
        ]
        pages.append({
            "width": page.width, "height": page.height,
            "angle": getattr(page, "angle", 0.0), "unit": page.unit,
            "words": words,
        })

    out = here / "ocr_dump.json"
    out.write_text(json.dumps({"image": image.name, "pages": pages}, indent=2))
    first = pages[0]
    print(f"saved {out.name}: page {first['width']}x{first['height']} {first['unit']}, "
          f"angle={first['angle']}, {len(first['words'])} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
