from pathlib import Path
import base64
from openai import OpenAI

client = OpenAI()

image_path = next(Path("data/inbox").glob("*"))

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Load OCR text
ocr_text = open("ocr_output.txt", "r").read() if Path("ocr_output.txt").exists() else ""

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"""
This is a handwritten notes image.

OCR text (may be wrong):
{ocr_text}

Use BOTH the image and OCR to reconstruct clean structured notes.

Extract:
- Title
- Headings
- Bullet points
- Key ideas
- Definitions
- Any structure (arrows, hierarchy)

Fix OCR mistakes.

Return clean JSON.
"""
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}"
                }
            ]
        }
    ]
)

print(response.output_text)