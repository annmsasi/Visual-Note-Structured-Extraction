-- Miso cache schema. The dense embedding lives in a separate sqlite-vec
-- virtual table (`summary_vectors`), created in db.py when the extension loads.

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Cold storage of raw OCR + extracted JSON, kept for provenance and debugging.
CREATE TABLE IF NOT EXISTS notes (
    note_id           TEXT PRIMARY KEY,
    course_id         TEXT NOT NULL,
    image_path        TEXT NOT NULL,
    raw_ocr_json      TEXT,
    extracted_json    TEXT,
    processing_order  INTEGER NOT NULL,
    pipeline_version  TEXT NOT NULL,
    timestamp         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_course_order ON notes(course_id, processing_order);

-- The retrievable / injectable unit. One row per note.
-- bm25_text is pre-joined for in-memory BM25 (rebuilt on demand from this column).
CREATE TABLE IF NOT EXISTS summaries (
    note_id           TEXT PRIMARY KEY REFERENCES notes(note_id) ON DELETE CASCADE,
    course_id         TEXT NOT NULL,
    topic_line        TEXT NOT NULL,
    gist              TEXT NOT NULL,
    processing_order  INTEGER NOT NULL,
    timestamp         TEXT NOT NULL,
    bm25_text         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_course_order ON summaries(course_id, processing_order);

-- Per-course admitted terms.
CREATE TABLE IF NOT EXISTS lexicon_terms (
    course_id        TEXT NOT NULL,
    term             TEXT NOT NULL,
    frequency        INTEGER NOT NULL DEFAULT 1,
    context_snippet  TEXT,
    first_seen       TEXT NOT NULL,
    last_seen        TEXT NOT NULL,
    PRIMARY KEY (course_id, term)
);
CREATE INDEX IF NOT EXISTS idx_lexicon_course ON lexicon_terms(course_id);

-- Pending sightings — a term must recur ≥ N times before promotion to lexicon_terms.
CREATE TABLE IF NOT EXISTS lexicon_sightings (
    course_id        TEXT NOT NULL,
    term             TEXT NOT NULL,
    first_seen       TEXT NOT NULL,
    last_seen        TEXT NOT NULL,
    sighting_count   INTEGER NOT NULL DEFAULT 1,
    context_snippet  TEXT,
    PRIMARY KEY (course_id, term)
);
CREATE INDEX IF NOT EXISTS idx_sightings_course ON lexicon_sightings(course_id);

-- One row per replay run, with the serialised config that produced it.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    config_tag      TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT
);
