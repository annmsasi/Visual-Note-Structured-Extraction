"""Editable LLM prompts.

Every system prompt the pipeline sends to the model lives here as a Markdown file.
`load(name)` returns the text of `<name>.md` with `<!-- ... -->` editor comments
stripped, so you can leave notes in a prompt file without sending them to the model.
To change the system's behaviour, edit these files — no code changes needed.

    extraction_system  — turn a page image into a structured note (the main prompt)
    combine_system     — merge a multi-page document's pages into one note
    summary_system     — the retrieval summary written for each note
"""
from __future__ import annotations

import re
from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    """Return the prompt in `<name>.md`, with HTML editor comments stripped."""
    text = (_DIR / f"{name}.md").read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text.strip()
