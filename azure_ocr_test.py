"""Azure Document Intelligence OCR step  (branch: azure-preprocessing).

Drop-in alternative to ocr_test.py (Tesseract). It reads the RAW image from
data/inbox, runs Azure's `prebuilt-read` model, prints the recognized text with
a per-word confidence summary, and writes ocr_output.txt — the file
extract_test.py already expects but that the Tesseract path never produced.

It deliberately does NOT use the OpenCV grayscale + equalizeHist output: Azure
already does its own deskew / binarization / contrast normalization, and an A/B
on this repo's note showed that extra preprocessing *lowered* mean confidence
(0.88 -> 0.84) and nearly doubled low-confidence words (10% -> 19%). The only
prep Azure needs is conforming the file to its size/format limits, which
`_prepare_for_azure` handles. (The Tesseract path still uses the OpenCV
preprocessing, where it helps.)

Self-contained: depends only on the azure SDK + python-dotenv, not on the miso
pipeline (that lives on the `full-pipeline` branch).

.env must contain:
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
    AZURE_DOCUMENT_INTELLIGENCE_KEY=<key>
"""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _pick_image() -> Path:
    """Use the raw inbox image — Azure handles its own preprocessing internally,
    so the OpenCV grayscale/equalize output is intentionally bypassed here.
    """
    imgs = sorted(p for p in Path("data/inbox").glob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not imgs:
        raise SystemExit("No image found in data/inbox")
    return imgs[0]


def _prepare_for_azure(path: Path, max_edge: int = 4000) -> Path:
    """Azure caps upload size and dimensions; downscale + re-encode if needed.
    No-op for already-small JPEG/PNG so the preprocessed file is used as-is.
    """
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
