# Visual-Note-Structured-Extraction — eval

The full pipeline (vendored `miso/`) plus the tooling used to build corpora and
evaluate the cache (lexicon + retrieval). Branched from `full-pipeline`.

## Install
```bash
./install.sh        # .venv + requirements (includes sentence-transformers)
```

## Configure
Same `.env` as full-pipeline:
```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
ANTHROPIC_API_KEY=...
```

## Scripts
```
stage_corpus.py     download a HuggingFace notes dataset -> page images
run_corpus.py       run a corpus through the pipeline as one course; report cache activation
clean_bilingual.py  delete notes whose extraction contains Devanagari (non-English)
bentham_eval.py     raw vs lexicon-corrected CER against Bentham gold
test_ocr.py         single-image Azure OCR with per-word confidence
dump_ocr.py         dump OCR geometry (words + polygons) to ocr_dump.json
make_view.py        build a side-by-side image/OCR HTML viewer from ocr_dump.json
test_extract.py     single-note extraction via the Anthropic extractor
miso/eval/          metrics harness (CER, WER, structural F1, bootstrap CIs)
```

## Build a modern corpus and exercise the cache
```bash
.venv/bin/python stage_corpus.py HumynLabs/Handwritten-Computer-Science-Notes-Dataset corpora/cs cs
.venv/bin/python run_corpus.py corpora/cs --course cs --config full
```
`--config` is `full` (lexicon + retrieval), `lexicon` (lexicon only), or
`nocache` (neither). `run_corpus.py` prints per-note lexicon size, corrections,
glossary size, and retrieval injections, then the promoted lexicon terms.
OCR results are cached on disk (`.ocr_cache/`), so re-runs and ablations do not
re-bill Azure.

Drop non-English notes:
```bash
.venv/bin/python clean_bilingual.py --db cache_cs_full.db --course cs --corpus corpora/cs --apply
```

## Bentham CER ablation (against ground truth)
1. Download Bentham R0 (images + PAGE-XML gold) from
   https://zenodo.org/records/44519 and extract one archive box (e.g. `071_*`).
2. Build page-level gold by concatenating each page's PAGE-XML `TextLine`
   transcriptions into `corpora/bentham_sub_gold/<note_id>.json` as
   `{"note_id": ..., "transcription": ...}`, keyed in the order `run_corpus.py`
   assigns ids (`bentham-000`, `bentham-001`, ...).
3. Run and evaluate:
```bash
.venv/bin/python run_corpus.py corpora/bentham_sub --course bentham --config lexicon
.venv/bin/python bentham_eval.py
```
`bentham_eval.py` reports mean CER for raw OCR vs lexicon-corrected text and the
lexicon's correction precision/recall.

## Metrics harness
```bash
.venv/bin/python -m miso.eval analyze --runs runs/ --gold corpora/bentham_sub_gold
```

## Note on results
On the Bentham gold the lexicon slightly *raised* CER (correction precision was
low): it corrects toward the LLM's modernized terms while the gold is verbatim
archaic text. A fair quality test needs modern handwriting with verbatim
transcriptions. The cache mechanism (lexicon fill, retrieval injection) works;
its quality benefit is not yet demonstrated.
