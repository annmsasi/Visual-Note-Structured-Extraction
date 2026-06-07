"""Build two-level gold (per-page AND per-note) for an eval corpus.

Three corpora, two provenances, ONE GoldNote JSON shape ({note_id, transcription,
extracted_json, distinctive_terms}):

  * IMAGE-NATIVE  (TIM, messy): per-page gold is LLM-drafted from the page image by
    `miso.eval.armb draft` (now figure-aware). This module then GROUPS the pages of
    a multi-page note and writes the per-note gold. TIM needs no hand-correction
    ("the draft is the gold"); the messy corpus will.

  * TRANSCRIPTION-NATIVE (Bentham): an authoritative VERBATIM archaic transcription
    already exists (Transcribe Bentham). Re-drafting it from the image would modernise
    the spelling and DESTROY the verbatim gold, so we never do that — we PRESERVE the
    transcription and only ADD the missing `distinctive_terms` (+ light structure),
    derived from the text, then group pages into notes.

Output layout, ready for `run_grid --gold <dir>`:
    <gold_dir>/pages/<note_id>.json     # one per page
    <gold_dir>/notes/<doc_id>.json      # one per multi-page note (merged)

    # TIM / messy: draft per-page (bills API), then build the note level:
    python -m miso.eval.armb draft corpora/tim172a --course tim172a --model claude-opus-4-8
    python -m miso.eval.gold_build notes corpora/tim172a_gold --out corpora/tim172a_gold \
        --group-regex '(\\d+)\\.p\\d+'                 # note = lecture, page = .pNNN

    # Bentham: preserve the verbatim transcription, add terms, then the note level:
    python -m miso.eval.gold_build bentham bentham_eval_data/bentham_box071_gold \
        --out corpora/bentham_gold                      # note = item (071_<item>)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from miso.document import validate
from miso.eval.gold import GoldNote, load_gold

log = logging.getLogger(__name__)

# Placeholder for a figure/diagram — kept consistent with the gold-drafter and the
# system prompt (miso.augment) so neither side is penalised for the other's choice.
# A separate step extracts the figures themselves.
FIGURE_PLACEHOLDER = "[figure]"

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{3,}")  # candidate term token (>= 4 chars)


# --------------------------------------------------------------------------- #
# distinctive-term derivation (for transcription-native gold like Bentham)
# --------------------------------------------------------------------------- #

def _common_words() -> set[str]:
    try:
        from miso.wordlists import load_common_words
        return load_common_words()
    except Exception as e:  # pragma: no cover - wordlist optional
        log.warning("common-word list unavailable (%s); term filter is length-only", e)
        return set()


def heuristic_terms(text: str, common: set[str], max_terms: int = 20) -> list[str]:
    """No-API distinctive-term guess: tokens that are NOT general-English, ranked to
    favour proper nouns then longer then more-frequent words. Crude, and it still admits
    archaic function-words the stoplist misses ('shall', 'thereof'), so it is a SMOKE
    DEFAULT only — use `--terms <model>` to get gold-quality terms. First-seen casing is
    preserved so 'Gunpowder' stays capitalised; capitalised stopwords ('But', 'See') are
    already removed by the common-word filter, so capitalisation is a safe proper-noun cue.
    """
    counts: dict[str, int] = {}
    first_case: dict[str, str] = {}
    for tok in _WORD.findall(text):
        low = tok.lower()
        if low in common:
            continue
        counts[low] = counts.get(low, 0) + 1
        first_case.setdefault(low, tok)
    ranked = sorted(
        counts,
        key=lambda l: (first_case[l][:1].isupper(), len(l), counts[l]),
        reverse=True,
    )
    return [first_case[l] for l in ranked[:max_terms]]


def _llm_terms(text: str, model: str, max_terms: int = 25) -> list[str]:
    """Text-only distinctive-term extraction — reads the gold TRANSCRIPTION (never the
    image, so the verbatim archaic spelling stays authoritative) and returns the
    technical / proper-noun terms a course lexicon would target.
    """
    import os

    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    schema = {
        "type": "object",
        "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
        "required": ["terms"],
    }
    prompt = (
        "Below is a VERBATIM transcription of one handwritten page. List the distinctive "
        "terms a course-specific lexicon would target: technical vocabulary, proper nouns, "
        "named laws/cases, and recurring domain jargon — NOT general-English words. Keep each "
        "term spelled EXACTLY as it appears in the transcription (do not modernise archaic "
        "spelling). Call emit_terms.\n\n" + text
    )
    msg = client.messages.create(
        model=model, max_tokens=1024,
        tools=[{"name": "emit_terms", "description": "Distinctive terms on the page.",
                "input_schema": schema}],
        tool_choice={"type": "tool", "name": "emit_terms"},
        messages=[{"role": "user", "content": prompt}],
    )
    payload = next((b.input for b in msg.content
                    if getattr(b, "type", None) == "tool_use" and b.name == "emit_terms"), None)
    terms = (payload or {}).get("terms", []) if payload else []
    return [t for t in terms if isinstance(t, str) and t.strip()][:max_terms]


# --------------------------------------------------------------------------- #
# transcription cleaning + light structure (Bentham)
# --------------------------------------------------------------------------- #

def clean_bentham_text(text: str) -> str:
    """Normalise a Transcribe-Bentham transcription FOR TERM EXTRACTION ONLY — the
    stored gold transcription is left byte-for-byte intact. Joins '='/':' line-break
    continuations ("confine= / =ment" -> "confinement") and drops <gap/>-style TEI tags.
    """
    t = re.sub(r"<[^>]+>", " ", text)                 # <gap/>, <add>, ...
    t = re.sub(r"[=:]\s*\n\s*[=:]?", "", t)            # rejoin split words
    return t


def _light_structure(text: str) -> dict:
    """Minimal heuristic IR for transcription-native gold: first line = title, the rest
    as paragraph blocks. Structure here is intentionally weak (Bentham's value is the
    verbatim text + terms, not TEDS); enough to keep structural_f1 non-degenerate.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return validate({})
    title, body = lines[0], "\n".join(lines[1:])
    chunks = [c.strip().replace("\n", " ") for c in re.split(r"\n\s*\n", body) if c.strip()]
    blocks = [{"type": "paragraph", "text": c} for c in chunks] or [{"type": "paragraph", "text": body}]
    return validate({"title": title, "blocks": blocks,
                     "summary_topic_line": title, "summary_gist": ""})


# --------------------------------------------------------------------------- #
# page -> note merge (deterministic; gold must not hallucinate cross-page joins)
# --------------------------------------------------------------------------- #

def _merge_extracted(docs: list[dict]) -> dict:
    docs = [d for d in docs if d]
    if not docs:
        return validate({})
    title = next((d.get("title") for d in docs if d.get("title")), "(untitled)")
    blocks = [b for d in docs for b in d.get("blocks", [])]
    topic = next((d.get("summary_topic_line") for d in docs if d.get("summary_topic_line")), "")
    gist = " ".join(d.get("summary_gist", "") for d in docs).strip()
    return validate({"title": title, "blocks": blocks,
                     "summary_topic_line": topic, "summary_gist": gist})


def _union_terms(pages: list[GoldNote]) -> list[str]:
    """Case-insensitive union, first-seen casing preserved — the note's vocabulary is
    everything its pages teach the lexicon."""
    seen: set[str] = set()
    out: list[str] = []
    for p in pages:
        for t in p.distinctive_terms:
            k = t.lower()
            if k and k not in seen:
                seen.add(k)
                out.append(t)
    return out


def merge_pages_to_note(doc_id: str, pages: list[GoldNote]) -> dict:
    """Deterministically combine ordered per-page GoldNotes into one note-level gold.

    Concatenate the page transcriptions, concatenate their blocks, union their terms.
    Faithful and reproducible — no model in the loop, so the gold never invents a merge
    the system would then be graded against.
    """
    return {
        "note_id": doc_id,
        "transcription": "\n".join(p.transcription for p in pages if p.transcription),
        "extracted_json": _merge_extracted([p.extracted_json for p in pages]),
        "distinctive_terms": _union_terms(pages),
        "pages": [p.note_id for p in pages],
    }


# --------------------------------------------------------------------------- #
# grouping (page note_id -> ordered list of page note_ids)
# --------------------------------------------------------------------------- #

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", s).strip("-")


def group_by_manifest(manifest_path: Path) -> dict[str, list[str]]:
    """Group via an armb `stage-local` manifest ([{note_id, source_pdf, page}]):
    note = the source PDF, pages ordered by their page number."""
    entries = json.loads(Path(manifest_path).read_text())
    groups: dict[str, list[tuple[int, str]]] = {}
    for e in entries:
        doc = _slug(Path(e["source_pdf"]).stem)
        groups.setdefault(doc, []).append((int(e.get("page", 0)), e["note_id"]))
    return {doc: [nid for _, nid in sorted(pgs)] for doc, pgs in groups.items()}


def group_by_regex(note_ids: list[str], pattern: str) -> dict[str, list[str]]:
    """Group by a regex over the note_id; capture group 1 (or the whole match) is the
    note id. Pages with no match become their own single-page note."""
    rx = re.compile(pattern)
    groups: dict[str, list[str]] = {}
    for nid in note_ids:
        m = rx.search(nid)
        doc = (m.group(1) if (m and m.groups()) else m.group(0)) if m else nid
        groups.setdefault(doc, []).append(nid)
    return {doc: sorted(pgs) for doc, pgs in groups.items()}


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def _write_gold(out_dir: Path, gold: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{gold['note_id']}.json").write_text(
        json.dumps(gold, indent=2, ensure_ascii=False))


def _pages_dir(path: Path) -> Path:
    """Accept either a gold dir containing pages/ or a pages dir directly."""
    return path / "pages" if (path / "pages").is_dir() else path


def build_notes(pages_gold_dir: Path, out_dir: Path, grouping: dict[str, list[str]]) -> int:
    pages = load_gold(_pages_dir(pages_gold_dir))
    notes_dir = out_dir / "notes"
    written = 0
    for doc_id, page_ids in sorted(grouping.items()):
        members = [pages[pid] for pid in page_ids if pid in pages]
        if not members:
            log.warning("note %s: no pages found, skipped", doc_id)
            continue
        _write_gold(notes_dir, merge_pages_to_note(doc_id, members))
        written += 1
        log.info("note %s <- %d page(s): %s", doc_id, len(members),
                 ", ".join(p.note_id for p in members))
    print(f"wrote {written} per-note gold file(s) to {notes_dir}/")
    return written


def augment_bentham(src_dir: Path, out_dir: Path, terms_mode: str, note_regex: str) -> int:
    """Read existing per-page Bentham gold (*.json with a `transcription`, or raw *.txt),
    PRESERVE each transcription verbatim, add distinctive_terms (+ light structure), and
    write both the per-page and per-note gold."""
    common = _common_words() if terms_mode == "heuristic" else set()
    pages_dir = out_dir / "pages"
    sources = sorted(p for p in src_dir.glob("*") if p.suffix.lower() in (".json", ".txt"))
    if not sources:
        print(f"no .json/.txt gold sources in {src_dir}", file=sys.stderr)
        return 0
    for src in sources:
        if src.suffix.lower() == ".json":
            data = json.loads(src.read_text())
            note_id = data.get("note_id") or src.stem
            transcription = data.get("transcription") or ""
        else:
            note_id, transcription = src.stem, src.read_text()
        if not transcription.strip():
            log.warning("%s: empty transcription, skipped", note_id)
            continue
        cleaned = clean_bentham_text(transcription)
        terms = (heuristic_terms(cleaned, common) if terms_mode == "heuristic"
                 else _llm_terms(transcription, terms_mode))
        _write_gold(pages_dir, {
            "note_id": note_id,
            "transcription": transcription,          # verbatim, untouched
            "extracted_json": _light_structure(cleaned),
            "distinctive_terms": terms,
        })
        log.info("page %s: %d terms", note_id, len(terms))
    n_pages = len(list(pages_dir.glob("*.json")))
    print(f"wrote {n_pages} per-page gold file(s) to {pages_dir}/ (terms: {terms_mode})")
    page_ids = [p.stem for p in pages_dir.glob("*.json")]
    return build_notes(out_dir, out_dir, group_by_regex(page_ids, note_regex))


# --------------------------------------------------------------------------- #
# edited gold markdown (draft_gold_md format) -> GoldNote JSON
# --------------------------------------------------------------------------- #

# The three top-level sections of a gold-markdown doc. Only these `## ` headers
# delimit sections; any OTHER `## ` line is a sub-heading inside the structured-note
# body (e.g. "## How to implement?") and must stay with its section, not start a new one.
_SECTION_HEADERS = ("transcription", "structured note", "distinctive terms")


def _split_sections(md: str) -> dict[str, str]:
    """Split a gold markdown doc into its known `## `-delimited sections (lowercased
    keys). A `## ` line whose heading is not a known section header is kept as body
    content (a structured-note sub-heading), not treated as a section boundary."""
    sections: dict[str, str] = {}
    current, buf = None, []
    for line in md.splitlines():
        heading = line[3:].strip().lower() if line.startswith("## ") else None
        if heading is not None and any(heading.startswith(h) for h in _SECTION_HEADERS):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current, buf = heading, []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _section(sections: dict[str, str], prefix: str) -> str:
    for k, v in sections.items():
        if k.startswith(prefix):
            return v
    return ""


def _md_to_ir(md: str) -> dict:
    """Parse the '## Structured note' markdown into the validated document IR."""
    title, blocks, items = "", [], []
    eq: list[str] | None = None

    def flush():
        if items:
            blocks.append({"type": "list", "items": list(items)})
            items.clear()

    for raw in md.splitlines():
        s = raw.strip()
        if eq is not None:                          # inside a $$ ... $$ block
            if s.endswith("$$"):
                eq.append(s[:-2]); flush()
                blocks.append({"type": "equation", "latex": "\n".join(eq).strip()}); eq = None
            else:
                eq.append(s)
            continue
        if not s:
            continue
        if s.startswith("# "):
            flush(); title = s[2:].strip(); continue
        m = re.match(r"^(#{2,6})\s+(.+)", s)
        if m:
            flush()
            blocks.append({"type": "heading", "level": min(3, len(m.group(1)) - 1),
                           "text": m.group(2).strip()})
            continue
        if s == FIGURE_PLACEHOLDER:
            flush(); blocks.append({"type": "paragraph", "text": FIGURE_PLACEHOLDER}); continue
        lm = re.match(r"^(\s*)[-*+]\s+(.+)", raw)
        if lm:
            items.append({"text": lm.group(2).strip(), "level": len(lm.group(1)) // 2}); continue
        if s.startswith("$$"):
            body = s[2:]
            if body.endswith("$$") and len(body) >= 2:
                flush(); blocks.append({"type": "equation", "latex": body[:-2].strip()})
            else:
                eq = [body] if body else []
            continue
        flush(); blocks.append({"type": "paragraph", "text": s})
    if eq is not None:
        flush(); blocks.append({"type": "equation", "latex": "\n".join(eq).strip()})
    flush()
    return validate({"title": title or "(untitled)", "blocks": blocks,
                     "summary_topic_line": title, "summary_gist": ""})


def parse_gold_md(md_path: Path, note_id: str | None = None) -> dict:
    """Edited per-page gold markdown -> GoldNote dict (transcription + IR + terms)."""
    sections = _split_sections(md_path.read_text())
    terms_md = _section(sections, "distinctive terms")
    terms = [re.sub(r"^[-*+]\s+", "", l).strip()
             for l in terms_md.splitlines() if l.lstrip().startswith(("-", "*", "+"))]
    return {
        "note_id": note_id or md_path.stem,
        "transcription": _section(sections, "transcription"),
        "extracted_json": _md_to_ir(_section(sections, "structured note")),
        "distinctive_terms": [t for t in terms if t],
    }


def parse_gold_dir(gold_dir: Path, out_dir: Path, grouping: dict[str, list[str]] | None) -> int:
    """Parse every `<nid>/<nid>.md` (or `<nid>.md`) under `gold_dir` into per-page
    GoldNote JSON, then optionally build the per-note gold."""
    mds = sorted(gold_dir.glob("*/*.md")) or sorted(gold_dir.glob("*.md"))
    pages_dir = out_dir / "pages"
    for md in mds:
        _write_gold(pages_dir, parse_gold_md(md, md.stem))
    print(f"parsed {len(mds)} markdown gold file(s) -> {pages_dir}/")
    if grouping is not None:
        build_notes(out_dir, out_dir, grouping)
    return len(mds)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build two-level (page + note) eval gold.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("notes", help="Group existing per-page gold into per-note gold.")
    n.add_argument("pages_gold_dir", help="dir with per-page gold (or its pages/ subdir)")
    n.add_argument("--out", required=True, help="gold dir; notes written to <out>/notes/")
    g = n.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="armb stage-local manifest.json (note = source PDF)")
    g.add_argument("--group-regex", help="regex over note_id; group 1 = note id")
    g.add_argument("--singletons", action="store_true", help="each page is its own note")

    b = sub.add_parser("bentham", help="Augment verbatim Bentham gold with terms (+ notes).")
    b.add_argument("src_dir", help="dir of existing gold (*.json with transcription, or *.txt)")
    b.add_argument("--out", required=True)
    b.add_argument("--terms", default="heuristic",
                   help="heuristic (no API) | a Claude model id for text-only term extraction")
    b.add_argument("--note-regex", default=r"(.+)_\d+$",
                   help="group pages into items; default groups 071_<item>_<page> by item")

    p = sub.add_parser("parse", help="Parse edited gold markdown (<nid>/<nid>.md) into GoldNote JSON (+ notes).")
    p.add_argument("gold_dir", help="dir of per-page gold-markdown folders")
    p.add_argument("--out", required=True, help="gold dir; pages/ + notes/ written here")
    gp = p.add_mutually_exclusive_group()
    gp.add_argument("--manifest", help="armb stage-local manifest.json (note = source PDF)")
    gp.add_argument("--group-regex", help="regex over note_id; group 1 = note id")
    gp.add_argument("--singletons", action="store_true", help="each page is its own note")

    args = ap.parse_args(argv)

    if args.cmd == "notes":
        pages_gold_dir = Path(args.pages_gold_dir)
        if args.manifest:
            grouping = group_by_manifest(Path(args.manifest))
        else:
            ids = list(load_gold(_pages_dir(pages_gold_dir)).keys())
            grouping = ({i: [i] for i in ids} if args.singletons
                        else group_by_regex(ids, args.group_regex))
        return 0 if build_notes(pages_gold_dir, Path(args.out), grouping) else 1

    if args.cmd == "bentham":
        from miso.eval.ocr_runner import _load_env
        if args.terms != "heuristic":
            _load_env()  # need ANTHROPIC_API_KEY for LLM term extraction
        return 0 if augment_bentham(Path(args.src_dir), Path(args.out),
                                    args.terms, args.note_regex) else 1

    if args.cmd == "parse":
        gd = Path(args.gold_dir)
        ids = [m.stem for m in (sorted(gd.glob("*/*.md")) or sorted(gd.glob("*.md")))]
        if args.manifest:
            grouping = group_by_manifest(Path(args.manifest))
        elif args.singletons:
            grouping = {i: [i] for i in ids}
        elif args.group_regex:
            grouping = group_by_regex(ids, args.group_regex)
        else:
            grouping = None
        return 0 if parse_gold_dir(gd, Path(args.out), grouping) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
