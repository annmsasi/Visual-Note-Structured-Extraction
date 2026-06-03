# Miso Evaluation — v1 Design (decided)

**Status.** Resolved v1 evaluation design. It **supersedes `cache_design_v1.md` §6** on two load-bearing points that turned out to be wrong in practice:
1. **Ground-truth substrate.** §6.1 assumed *team-own chronological course notes*. The team has none and cannot produce them at low effort. → Replaced by a **synthesize-from-open-courseware** corpus plus a **small LLM-assisted real-handwriting set**.
2. **Primary metric.** §6.4 named *CER vs corrected ground truth* as primary. Miso is a **normalizer / structurer / summarizer, not a transcriber** — CER against a *verbatim* reference penalizes exactly the cleanup the system is designed to do. The team's Bentham run proved this empirically (lexicon *raised* CER). → Replaced by a **capability-aligned metric suite** (term-recall headline + structured/normalized gold).

Everything in `cache_design_v1.md` §1–§5 (the system under test) stands. The instrumentation (config-switchable pipeline, replay-from-empty-cache, per-note JSON traces, the 2×2 attribution in `miso/eval/analyze.py`) is built and sound; the gap was never the harness — it was *what we measure against*.

---

## 1. The core reframe

Miso takes a handwritten page image and emits a **cleaned, restructured, modernized, summarized** structured document (see `miso/examples/electric_power_distribution.json`: OCR `"18805... Pearl street... wisconsin... plante"` → extraction `"1880s – Early Electric Distribution"` with fixed spelling, invented headings, dropped OCR garbage, and a 150-token abstract). Three capabilities are bundled, and the cache touches each differently:

- **Recognition of recurring course vocabulary** — the lexicon read-path's job.
- **Structured extraction** — the LLM's job, weakly helped by the summary read-path + glossary.
- **Summarization** — piggybacked, feeds the cache.

**Measure each capability at its own altitude, against the reference it actually targets.** Do not collapse all three into one transcription-style number.

The cache's value also has a hard *structural* prerequisite: **a chronological single-author "course" with recurring distinctive vocabulary and topical continuity.** No public dataset combines {modern handwritten register, longitudinal single-author course structure, ground-truth gold}. That is why the corpus must be partly manufactured.

---

## 2. Three-arm evaluation architecture

| Arm | What it is | Gold | What it carries |
|---|---|---|---|
| **A — Synthetic courses** | Open courseware → telegraphic note-rewrite → realism-maximized handwriting render | Free & exact: verbatim note text + structured JSON, both derived from the clean source | The **powered** experiments: 2×2 ablation, term-recall vs #prior-exposures, ramp curves, clean-vs-self propagation tax, all knob sweeps. Scales to hundreds of pages. |
| **B — Real handwriting** | A small slice of existing real handwritten STEM notes (HumynLabs HF datasets) | **~30–40 pages**, LLM-drafted then human-corrected: verbatim text + structured JSON | **External validity**: do Arm-A findings hold on real handwriting? Honest real-world absolute numbers. |
| **C — Calibration anchors** | IAM (general English) + HME100K / MathWriting (handwritten math) | Ships with the datasets (line transcriptions / LaTeX) | Proves Arm A isn't "too clean": compares Azure CER, error-type confusion, and confidence distributions on synthetic vs real. |

**Division of labor:** Arm A gives statistical power and controlled ablation; Arm B gives credibility; Arm C ties them together. Because Arm B carries the real-world claim, **Arm A realism only needs to pass Arm C's calibration band**, not be photo-real.

---

## 3. Arm A — synthetic course corpus

### 3.1 Source text (= the gold)
2–3 openly-licensed "courses", each a chronological single-subject sequence with dense recurring vocabulary:

- **Algorithms** — MIT OCW **6.006** lecture notes (CC BY-NC-SA). ~20 lectures out of the box; vocabulary like *amortized, invariant, AVL, relaxation, memoization, DAG*. Closest to the CS domain.
- **Data Science** — OpenStax **Principles of Data Science** (CC BY-NC-SA, **CNXML source** → headings/lists/MathML parse cleanly into structured gold). ~18–22 note chunks.
- **Quantum computing** — arXiv:2311.08445 lecture notes (CC BY-NC-SA, **LaTeX source**) or arXiv:2204.04198 (CC BY). Extremely distinctive recurring vocab (*qubit, unitary, ansatz, stabilizer, eigenphase*).

License note: NC permits research use; SA only constrains redistribution of derivatives. Keep the corpus internal to be safe.

### 3.2 Note-shaping
LLM-rewrite each chunk into **telegraphic student-note style** (bullets, abbreviations, fragments — the register the system actually targets), **pinned lossless on technical terms** (constrain to reword/abbreviate; spot-check that distinctive vocabulary survives). Real student notes ≠ polished prose; rendering verbatim textbook prose would overstate accuracy and under-exercise the lexicon.

Emit **two gold artifacts per note**, in `miso/eval/gold.py` `GoldNote` format (dir of per-note JSON: `note_id`, `extracted_json`, `transcription`):
- `transcription` — the note-style text (verbatim reference for OCR/lexicon scoring).
- `extracted_json` — structured gold derived from the **original source markup**, *not* the rewrite (breaks the circularity of grading an LLM extraction against an LLM rewrite).

### 3.3 Rendering (realism-maximized, calibration-bounded)
- One consistent "hand" per course (consistent writer); vary glyphs *within* a course (jitter / sub-styles) so a recurring term errs *variably* — see threat T2.
- **Backbone:** handwriting fonts + Pillow page layout (titles, indentation, nested bullets) → **Augraphy** degradation (paper texture, ink fade, bleed-through, scan shadow). *Realism comes from the degradation + layout layer, not the glyph generator.*
- **Realism upgrade:** splice **One-DM** (ECCV 2024, MIT, pretrained, single-GPU inference) generated words for true letterform variety on a subset.
- **Acceptance = Arm C calibration band** (§5), not visual "looks real."

### 3.4 What Arm A enables that team-own-notes never could
Because we control *which terms recur and exactly when each first appears*, we can measure **term-recall as a function of #prior exposures** — a clean, powered curve. Plus clean chronology for ramp curves and the clean-vs-self tax, and arbitrary N for tight bootstrap CIs.

---

## 4. Arm B — real handwriting

No public dataset of real handwritten *technical* notes ships transcription/structured gold (verified: NoTeS-Bank is unreleased and anti-OCR by design; HumynLabs HF notes are images-only; IAM is generic prose). So Arm B = **make a little gold on real notes**:

- **Images:** HumynLabs HF datasets — `Handwritten-Computer-Science-Notes`, `English-Handwritten-Math-Notes`, plus Physics/Biology variants (CC BY 4.0, right register, real handwriting). Group by source into pseudo-"courses" where possible.
- **Gold (~30–40 pages):** LLM drafts the structured extraction + transcription, **human corrects** (the `cache_design_v1.md` §6.1 LLM-assisted workflow, applied to a small *real* set instead of 60–80 own notes). Same `GoldNote` format as Arm A.
- **Carries:** rerun the core 2×2 + term-restricted metrics on real handwriting. If Arm-A directional findings replicate (even with wide CIs at n≈35), the synthetic-primary claim holds. If the real lexicon gain is much smaller, that catches Arm-A's "artificial recurrence" inflation (T2) — and is itself a finding.

---

## 5. Arm C — calibration (the validity backbone)

Real handwriting + verbatim gold, ~zero effort, for OCR-error calibration only:
- **IAM Handwriting DB** — general English CER baseline (free, registration).
- **HME100K** / **MathWriting** (CC BY-SA) — handwritten-math LaTeX gold for equation CER.

Run frozen Azure (and optionally Textract) on Arm A, Arm B, and Arm C, then compare on three axes:
1. **CER/WER distributions** (per-document; KS test / bootstrap CIs). Require overlap and **synthetic-mean-CER ≥ real-mean-CER** (synthetic at least as hard).
2. **Error-type taxonomy + character-confusion matrix** (substitution/insertion/deletion/segmentation/layout). Top-k confusion pairs should overlap — proves errors are *the same kind* (homoglyphic, systematic — the post-OCR-correction premise).
3. **OCR confidence histograms** (Azure is well-calibrated, ECE ~1–3). Synthetic confidences must not be systematically higher than real.

**Pre-register the realism band** (the acceptable synthetic CER range, anchored to Arm C) *before* running, so a too-clean corpus can't be rationalized post-hoc. Tune Augraphy strength + glyph realism until the three axes align.

---

### 5.1 Calibration findings (measured 2026-06-03, first run)

Built and ran the Arm-C tooling (`miso/eval/calibrate.py`, `miso/eval/ocr_runner.py`,
`miso/synth/iam.py`) on 6 synthetic Biology pages vs 6 real IAM pages, Azure `prebuilt-read`:

| corpus | CER | OCR confidence (mean / p10) | error mix |
|---|---|---|---|
| synthetic (font render, degrade+elastic) | **~1.5%** | 0.95 / 0.87 | mostly whitespace insertions |
| real IAM | **~5.6%** | 0.90 / 0.70 | 43% substitution, 44% deletion (genuine misreads) |

**Finding (robust across degrade-strength 1.0–2.5 + elastic warp + punctuation normalization):**
a clean handwriting **font is too regular for Azure to misread at the character level** —
blur/noise/elastic warp lower confidence and add a few real errors but **cannot close the
CER gap** to real handwriting (synthetic plateaus ~1–1.6%; real ~5.6%). This confirms the
HTG literature: realistic recognition errors come from *glyph irregularity*, not image
degradation. Consequences for how the arms are used:

- **Synthetic (Arm A) = controlled, powered *relative* measurement** (2×2 deltas, term-recall
  vs exposures, propagation tax) at an **optimistically-clean operating point**. Report its
  absolute CER as a lower bound, not a real-world estimate.
- **Real (Arm B) carries the *absolute* OCR-difficulty and realism claim.** This is why the
  three-arm design is necessary — synthetic alone would overstate the system.
- **Upgrade path if absolute-matching synthetic is wanted:** generative handwriting glyphs
  (One-DM / DiffusionPen, pretrained) instead of fonts — deferred (torch/GPU); the `render.py`
  seam (`_render_clean`) is where they'd drop in. Re-run calibration to confirm the band.

---

## 6. Metric suite (capability-aligned)

| Capability | Stage scored | Reference | Metrics |
|---|---|---|---|
| **Recognition of recurring terms** (lexicon) | **OCR-stage output**, not LLM output | **verbatim**, **term-restricted** to distinctive terms | term-restricted CER/WER (raw vs lexicon-corrected); correction precision/recall; **over-correction rate** (release criterion) |
| **Extraction quality** (headline) | structured JSON | **structured/normalized** gold | **term-recall** (headline), field/content-F1, structural-F1 (built); normalized-CER vs *cleaned* gold only as a legibility number; LLM-as-judge faithfulness |
| **Retrieval contribution** | end-to-end | structured gold | 2×2 attribution + bootstrap CIs (built); distractor stress test; ramp curve; optional recall@k spot-check |
| **Error-propagation tax** | end-to-end | both gold sets | clean-vs-self CER/term-recall gap (`EvalConfig.cache_from_corrected_ground_truth` already exists) |
| **Systems** | all | — | per-stage latency (traced), cost/note, cache activation/hit rates (reported by `run_corpus.py`) |

**Headline = term-recall / term-restricted CER, not global CER.** The cache only touches the ~5% of tokens that are recurring distinctive vocabulary; global CER stays flat even when the cache helps (this is precisely why the global-CER Bentham run read as a null). Foreground the term-restricted numbers.

**CER-vs-cleaned-gold is valid; CER-vs-verbatim is the bug.** `miso/eval/analyze.py` already flattens *both* the structured gold and structured hypothesis (apples-to-apples) — keep that. The eval-branch `bentham_eval.py` regressed to scoring against verbatim transcription — retire it.

### Harness fixes required (small, corpus-independent — can start now)
1. **Term-recall metric** + a per-note distinctive-term list (the corpus knows which terms are the recurring targets).
2. **Term-restricted CER/WER** at the OCR stage (score `ocr_raw` vs `corrected_ocr` against the term spans).
3. **Alignment-based** `correction_precision_recall` (current version is token-set-membership — order-insensitive and noisy; align corrections to gold tokens).
4. **Decouple summary from body** in `analyze.py` flattening (currently `_flatten_strings` mixes `summary_gist` into the body CER comparison).
5. **LLM-as-judge faithfulness** (`EvalConfig.enable_faithfulness_check` is stubbed) — one call/note, each extracted field checked against image/OCR.

---

## 7. Ablation grid (unchanged shape, now powered)

The `cache_design_v1.md` §6.2 grid stands and maps to the existing `RunConfig` factories:
- **6 core configs** (OCR-only, LLM-only-image, LLM+OCR `config_3`, +lexicon `config_4`, +retrieval `config_5`, full `config_6`); configs 3–6 form the **2×2 (lexicon × retrieval)**.
- **Knob sweeps** from config 6: N (1/2/3), top-k (1/3/5), distractor (0/1/2/4), gate on/off, reranker on/off.
- Every cache-on config reported as a **ramp curve**. Run on Arm A (powered) and the core 2×2 on Arm B (validity).

---

## 8. Honest framing / threats to validity

- **T0 — the VLM-does-the-work confound (the real threat to the thesis).** A frontier VLM reading the image directly may already disambiguate most recurring terms, leaving little marginal room for OCR-correction or summary-injection. The 6-config grid (image-only vs image+OCR vs full) isolates this. **A null on a read-path is a legitimate capstone finding**, not a failure — report it as one.
- **T1 — "too clean" synthetic.** The realism bar is set by the *strongest reader* (the VLM), not just Azure. Degrade until both visibly struggle; gate on the §5 band.
- **T2 — artificial recurrence.** One font mis-reads a term identically every time → overstates the lexicon. Vary within-course glyphs; report Arm-B lexicon gain separately as the honest check.
- **T3 — content-distribution dominance.** Transfer is hurt more by content mismatch than visual style — the telegraphic note-rewrite (§3.2) is doing real validity work.
- **T4 — under-power.** Arm B is small (n≈35); report CIs and lean on Arm A for power. Arm A is synthetic; lean on Arm B + Arm C for realism. Neither arm alone is sufficient — the triangulation *is* the argument.

---

## 9. Build plan

1. **Metric-harness fixes** (§6) — corpus-independent, start immediately.
2. **Arm A generator** — source fetch/parse → note-rewrite → render+degrade → gold emit (`GoldNote` format). Fast path (fonts+Augraphy) first; add One-DM if calibration demands.
3. **Arm C calibration script** — Azure over synthetic/real; the three-axis comparison; lock the realism band.
4. **Arm B annotation pass** — LLM-draft + human-correct ~30–40 HF pages.
5. **Run the grid** (Arm A powered, Arm B core 2×2) → ramp curves, 2×2 attribution, clean-vs-self tax, term-recall-vs-exposures.

Effort: Arm-A generator ≈ 2–3 days (fast path), then push-button to scale. Arm-B annotation ≈ the only real manual cost, bounded at ~30–40 pages.

---

## 10. Retired

- **Bentham** as a cache/quality benchmark (wrong register + verbatim-gold mismatch). At most a single cited "OCR-engine sanity CER on standard HTR," decoupled from the cache.
- **`bentham_eval.py`** (scores against verbatim transcription).
- **CER-vs-verbatim** as a system metric.
- **Team-own notes** as the substrate (infeasible).
- **NoTeS-Bank** (unreleased; no transcription/structured gold).

---

## Open items
- Confirm HumynLabs HF dataset sizes / per-source grouping for Arm B pseudo-courses.
- Decide Arm-A scale (pages/course) and final course count (2 vs 3).
- Choose the Arm-B annotation tool (extend `miso/eval` or a small standalone corrector UI).
- Decide whether equations get a dedicated metric (HME100K/MathWriting suggest an equation-CER sub-metric is cheap).
