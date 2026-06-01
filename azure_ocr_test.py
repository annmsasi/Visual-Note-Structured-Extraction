"""Azure prebuilt-read OCR step; alternative to ocr_test.py (Tesseract).

Reads the raw image from data/inbox (Azure does its own deskew/binarization, so
the OpenCV grayscale output is bypassed), prints the text with a per-word
confidence summary, and writes ocr_output.txt for extract_test.py.

.env: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY
"""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _pick_image() -> Path:
    """Return the first image in data/inbox."""
    imgs = sorted(p for p in Path("data/inbox").glob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not imgs:
        raise SystemExit("No image found in data/inbox")
    return imgs[0]


def _prepare_for_azure(path: Path, max_edge: int = 4000) -> Path:
    """Downscale/re-encode to fit Azure's size and format limits."""
    try:
        from PIL import Image
    except ImportError:
        return path
    img = Image.open(path)
    if max(img.size) <= max_edge and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return path
    img = img.convert("RGB")
    if max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    out = path.with_name(path.stem + ".azure.jpg")
    img.save(out, "JPEG", quality=90)
    return out


def main() -> None:
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not (endpoint and key):
        raise SystemExit(
            "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
            "AZURE_DOCUMENT_INTELLIGENCE_KEY in .env"
        )

    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    image_path = _prepare_for_azure(_pick_image())
    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    with open(image_path, "rb") as fh:
        poller = client.begin_analyze_document(
            "prebuilt-read", body=fh, content_type="application/octet-stream"
        )
    result = poller.result()

    words = [w for page in (result.pages or []) for w in (page.words or [])]
    text = result.content or " ".join(w.content for w in words)

    Path("ocr_output.txt").write_text(text)

    print("IMAGE:", image_path.name)
    print("OCR TEXT:")
    print(text)
    if words:
        confs = [w.confidence or 0.0 for w in words]
        low = sum(c < 0.70 for c in confs)
        print(f"\n{len(words)} words | mean conf {sum(confs) / len(confs):.2f} "
              f"| {low} below 0.70")
    print("\nWrote ocr_output.txt (consumed by extract_test.py)")


if __name__ == "__main__":
    main()
