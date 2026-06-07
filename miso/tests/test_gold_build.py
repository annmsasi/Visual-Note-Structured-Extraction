"""Tests for the two-level gold builder — pure logic, no API, no venv.

Runs under bare `python -m miso.tests.test_gold_build` or `pytest`.
"""
from __future__ import annotations

from miso.eval.gold import GoldNote
from miso.eval.gold_build import (
    FIGURE_PLACEHOLDER, clean_bentham_text, group_by_manifest, group_by_regex,
    heuristic_terms, merge_pages_to_note,
)


def _page(note_id, transcription, terms, blocks=None):
    return GoldNote(
        note_id=note_id,
        transcription=transcription,
        extracted_json={"title": note_id, "blocks": blocks or [],
                        "summary_topic_line": "", "summary_gist": ""},
        distinctive_terms=terms,
    )


def test_heuristic_terms_filters_common_and_ranks():
    common = {"the", "of", "and", "to", "for", "be", "in", "a"}
    text = ("the law of Gunpowder and the Gunpowder store ; Embezzlement of the goods "
            "to be punished for Embezzlement and theft")
    terms = heuristic_terms(text, common, max_terms=5)
    assert "the" not in [t.lower() for t in terms], terms       # common filtered
    assert "Gunpowder" in terms and "Embezzlement" in terms, terms
    # frequency + capitalisation rank the proper nouns to the top
    assert terms[0] in ("Gunpowder", "Embezzlement"), terms


def test_clean_bentham_joins_continuations_and_drops_tags():
    raw = "confine=\n=ment exceeding <gap/> that term"
    cleaned = clean_bentham_text(raw)
    assert "confinement" in cleaned, cleaned
    assert "<gap/>" not in cleaned and "=" not in cleaned, cleaned
    # colon-style continuation too
    assert "neighbourhood" in clean_bentham_text("neigh:\n:bourhood")


def test_merge_concatenates_unions_and_lists_pages():
    p0 = _page("doc-000", "page zero text", ["Phasor", "KCL"],
               blocks=[{"type": "heading", "level": 1, "text": "A"}])
    p1 = _page("doc-001", "page one text", ["kcl", "Impedance"],
               blocks=[{"type": "paragraph", "text": "body"}])
    note = merge_pages_to_note("doc", [p0, p1])

    assert note["transcription"] == "page zero text\npage one text"
    # case-insensitive term union, first-seen casing kept, no dup of KCL/kcl
    assert note["distinctive_terms"] == ["Phasor", "KCL", "Impedance"], note["distinctive_terms"]
    # blocks from both pages survive the merge
    assert len(note["extracted_json"]["blocks"]) == 2, note["extracted_json"]["blocks"]
    assert note["pages"] == ["doc-000", "doc-001"]


def test_merge_preserves_figure_placeholder():
    p = _page("d-000", f"intro line\n{FIGURE_PLACEHOLDER}\nafter the figure", [])
    note = merge_pages_to_note("d", [p])
    assert FIGURE_PLACEHOLDER in note["transcription"], note["transcription"]


def test_group_by_regex_items_with_singleton_fallback():
    ids = ["071_159_001", "071_159_002", "071_160_001", "bentham-000"]
    groups = group_by_regex(ids, r"(.+)_\d+$")
    assert groups["071_159"] == ["071_159_001", "071_159_002"], groups
    assert groups["071_160"] == ["071_160_001"], groups
    # no _\d+ suffix -> its own single-page note
    assert groups["bentham-000"] == ["bentham-000"], groups


def test_group_by_manifest_orders_by_page():
    import json
    import tempfile
    from pathlib import Path

    manifest = [
        {"note_id": "tim-002", "source_pdf": "lec01.pdf", "page": 3},
        {"note_id": "tim-000", "source_pdf": "lec01.pdf", "page": 1},
        {"note_id": "tim-001", "source_pdf": "lec01.pdf", "page": 2},
        {"note_id": "tim-003", "source_pdf": "lec02.pdf", "page": 1},
    ]
    with tempfile.TemporaryDirectory() as d:
        mpath = Path(d) / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        groups = group_by_manifest(mpath)
    assert groups["lec01"] == ["tim-000", "tim-001", "tim-002"], groups   # page order
    assert groups["lec02"] == ["tim-003"], groups


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
