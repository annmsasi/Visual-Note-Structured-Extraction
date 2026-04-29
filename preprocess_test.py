from pathlib import Path
import cv2
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
image_path = next(Path("data/inbox").glob("*"))

img = cv2.imread(str(image_path))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Improve contrast
cleaned = cv2.equalizeHist(gray)

output_path = Path("data/output") / f"{image_path.stem}_clean.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output_path), cleaned)

print("Original image:", image_path)
print("Cleaned image saved to:", output_path)