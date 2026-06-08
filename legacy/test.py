from pathlib import Path

for img in Path("data/inbox").glob("*"):
    print(img.name)