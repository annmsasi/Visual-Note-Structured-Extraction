"""Run AzureOCR on a single image and print each word's confidence plus stats.

    python test_ocr.py [path/to/image.jpg]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def main() -> int:
    here = Path(__file__).parent
    _load_dotenv(here / ".env")

    image = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "Journal-example-1-2378482629.jpg"
    if not image.exists():
        print(f"Image not found: {image}", file=sys.stderr)
        return 1

    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
    if not endpoint or not key:
        print("Missing Azure creds. Fill in .env (endpoint + key) and re-run.", file=sys.stderr)
        return 2

    from miso.ocr import AzureOCR

    print(f"OCR on: {image.name}  ({image.stat().st_size // 1024} KB)\n")
    result = AzureOCR(endpoint, key).run(image)

    if not result.words:
        print("(no words returned)")
        return 0

    confs = [w.confidence for w in result.words]
    for w in result.words:
        flag = "  <-- low" if w.confidence < 0.7 else ""
        print(f"  {w.confidence:5.2f}  {w.text}{flag}")

    n = len(confs)
    low = sum(c < 0.7 for c in confs)
    print(f"\n{n} words | mean conf {sum(confs)/n:.3f} | "
          f"min {min(confs):.2f} | {low} below 0.70 ({low/n:.0%})")
    print(f"\n--- raw_text ---\n{result.raw_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
