from pathlib import Path
from PIL import Image
import pytesseract

image_path = next(Path("data/output").glob("*"))

text = pytesseract.image_to_string(Image.open(image_path))

print("IMAGE:", image_path.name)
print("OCR TEXT:")
print(text)