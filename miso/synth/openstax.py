"""Fetch + parse OpenStax CNXML books into ordered, structured 'note' sources.

Each module becomes a structured document (title + blocks) that doubles as the
extraction gold. The collection XML gives chronological module order. Downloads
are cached under .synth_cache/ so re-runs and ablations don't re-fetch.
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

log = logging.getLogger(__name__)

_CACHE = Path(".synth_cache/openstax")
_RAW = "https://raw.githubusercontent.com/{repo}/{br}/{path}"

# course key -> (github repo, collection file, branch). CC BY-NC-SA (research use).
COURSES: dict[str, tuple[str, str, str]] = {
    "biology": ("openstax/osbooks-biology-bundle", "biology-2e.collection.xml", "main"),
}

# End-matter / reference modules that aren't good "lecture note" content.
_SKIP_TITLES = {
    "preface", "key terms", "chapter summary", "visual connection questions",
    "review questions", "critical thinking questions", "critical thinking",
    "the periodic table of elements", "measurements and the metric system",
    "geological time",
}


@dataclass
class SourceNote:
    module_id: str
    title: str
    blocks: list[dict]

    @property
    def text_len(self) -> int:
        n = 0
        for b in self.blocks:
            n += len(b.get("text", "")) + len(b.get("latex", ""))
            n += sum(len(i.get("text", "")) for i in b.get("items", []))
        return n


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "miso-synth"})
    return urllib.request.urlopen(req, timeout=30).read()


def _cached(repo: str, br: str, path: str) -> bytes:
    key = _CACHE / repo.replace("/", "_") / path.replace("/", "_")
    if key.exists():
        return key.read_bytes()
    data = _get(_RAW.format(repo=repo, br=br, path=path))
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_bytes(data)
    return data


def _ln(el) -> str:
    return etree.QName(el).localname


# Normalize OpenStax typography to ASCII so CER isn't inflated by curly-quote /
# dash / ellipsis mismatches against OCR output (those aren't recognition errors).
_PUNCT_MAP = {
    0x2019: "'", 0x2018: "'", 0x201C: '"', 0x201D: '"',
    0x2014: "-", 0x2013: "-", 0x2026: "...", 0x00A0: " ",
}


def _text(el) -> str:
    return " ".join("".join(el.itertext()).translate(_PUNCT_MAP).split())


def _parse_blocks(container, level: int, out: list[dict]) -> None:
    for el in container:
        tag = _ln(el)
        if tag == "section":
            t = el.find("{*}title")
            if t is not None and _text(t):
                out.append({"type": "heading", "level": min(level, 3), "text": _text(t)})
            _parse_blocks(el, level + 1, out)
        elif tag == "para":
            txt = _text(el)
            if txt:
                out.append({"type": "paragraph", "text": txt})
        elif tag == "list":
            items = [{"text": _text(it), "level": 0}
                     for it in el.findall("{*}item") if _text(it)]
            if items:
                out.append({"type": "list", "items": items})
        elif tag == "equation":
            txt = _text(el)
            if txt:
                out.append({"type": "equation", "latex": txt})
        # skip: title (already consumed), figure/media/caption, note, example, exercise


def parse_module(xml: bytes) -> SourceNote:
    root = etree.fromstring(xml)
    title = "Untitled"
    blocks: list[dict] = []
    module_id = ""
    for child in root:
        if _ln(child) == "title":
            title = _text(child) or title
        elif _ln(child) == "content":
            _parse_blocks(child, 2, blocks)
    for cid in root.iter("{*}content-id"):
        module_id = _text(cid)
        break
    return SourceNote(module_id=module_id, title=title, blocks=blocks)


def module_order(collection_xml: bytes) -> list[str]:
    root = etree.fromstring(collection_xml)
    return [m.get("document") for m in root.iter("{*}module") if m.get("document")]


def iter_course(course: str, min_text: int = 400):
    """Yield content-bearing modules in chronological (collection) order, lazily —
    end-matter, front-matter, and stubs filtered out. Callers chunk these into
    page-sized notes, so this fetches only as many modules as needed."""
    repo, coll, br = COURSES[course]
    for mid in module_order(_cached(repo, br, f"collections/{coll}")):
        try:
            sn = parse_module(_cached(repo, br, f"modules/{mid}/index.cnxml"))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", mid, e)
            continue
        if sn.title.strip().lower() in _SKIP_TITLES or sn.text_len < min_text:
            continue
        yield sn


def fetch_course(course: str, limit: int | None = None, min_text: int = 400) -> list[SourceNote]:
    """Eager wrapper over iter_course (limit = number of modules)."""
    out: list[SourceNote] = []
    for sn in iter_course(course, min_text):
        out.append(sn)
        if limit and len(out) >= limit:
            break
    return out
