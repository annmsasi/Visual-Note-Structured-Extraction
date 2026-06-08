"""Hybrid retrieval (BM25 + dense via RRF), cross-encoder rerank, then gate."""
from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime
from typing import Protocol

from miso.config import RetrievalConfig
from miso.summary_store import SummaryStore
from miso.types import RetrievalResult, RetrievedSummary, Summary

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, candidates: list[str]) -> list[float]: ...


class _StubReranker:
    def score(self, query: str, candidates: list[str]) -> list[float]:
        q = set(_tokenise(query))
        if not q:
            return [0.0] * len(candidates)
        return [_jaccard(q, set(_tokenise(c))) for c in candidates]


def _tokenise(s: str) -> list[str]:
    return [t for t in s.lower().split() if t]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _MiniBM25:
    """In-Python BM25."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.n_docs = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n_docs, 1)
        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf = {
            term: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)
            for term, n in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for i, doc in enumerate(self.corpus):
            if not doc:
                continue
            dl = len(doc)
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            score = 0.0
            for q in query:
                if q not in self.idf:
                    continue
                f = tf.get(q, 0)
                if f == 0:
                    continue
                score += self.idf[q] * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                )
            scores[i] = score
        return scores


class RetrievalLayer:
    def __init__(
        self,
        conn: sqlite3.Connection,
        store: SummaryStore,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.conn = conn
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or _StubReranker()

    def retrieve(
        self,
        query_text: str,
        course_id: str,
        cfg: RetrievalConfig,
    ) -> RetrievalResult:
        if self.store.count_for_course(course_id) < cfg.cold_start_note_count:
            return RetrievalResult(query_text, [], [], cold_start_skip=True, filter_empty=False)

        corpus = self.store.bm25_corpus(course_id)
        if not corpus:
            return RetrievalResult(query_text, [], [], cold_start_skip=True, filter_empty=False)

        note_ids = [nid for nid, _ in corpus]
        texts = [t for _, t in corpus]

        bm25 = _MiniBM25([_tokenise(t) for t in texts])
        bm25_ranks = _ranks_from_scores(bm25.get_scores(_tokenise(query_text)))

        if self.embedder is not None:
            query_vec = self.embedder.encode([query_text])[0]
            doc_vecs = self.embedder.encode(texts)
            dense_ranks = _ranks_from_scores([_cosine(query_vec, v) for v in doc_vecs])
        else:
            # No embedder: all tied last, so BM25 decides.
            dense_ranks = [len(texts)] * len(texts)

        rrf = [
            1.0 / (cfg.rrf_k + bm25_ranks[i]) + 1.0 / (cfg.rrf_k + dense_ranks[i])
            for i in range(len(texts))
        ]
        order = sorted(range(len(texts)), key=lambda i: -rrf[i])[: cfg.top_k_candidates]
        top_summaries = self._load_summaries(note_ids[i] for i in order)
        candidates = [
            RetrievedSummary(summary=s, retrieval_score=rrf[i], reranker_score=None)
            for s, i in zip(top_summaries, order)
        ]

        if cfg.reranker_enabled and candidates:
            texts_for_rerank = [c.summary.topic_line + " " + c.summary.gist for c in candidates]
            for c, sc in zip(candidates, self.reranker.score(query_text, texts_for_rerank)):
                c.reranker_score = sc
            kept = [c for c in candidates
                    if (c.reranker_score or 0.0) >= cfg.reranker_threshold]
            if cfg.recency_tie_break:
                kept.sort(key=lambda c: (-(c.reranker_score or 0.0),
                                         -c.summary.processing_order))
            else:
                kept.sort(key=lambda c: -(c.reranker_score or 0.0))
            injected = kept[: cfg.top_k_inject]
        else:
            injected = candidates[: cfg.top_k_inject]

        return RetrievalResult(
            query_text=query_text,
            candidates_top10=candidates,
            injected=injected,
            cold_start_skip=False,
            filter_empty=(not injected),
        )

    def _load_summaries(self, note_ids) -> list[Summary]:
        ids = list(note_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT note_id, course_id, topic_line, gist, processing_order, timestamp "
            f"FROM summaries WHERE note_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {
            r["note_id"]: Summary(
                note_id=r["note_id"],
                course_id=r["course_id"],
                topic_line=r["topic_line"],
                gist=r["gist"],
                processing_order=r["processing_order"],
                pipeline_version="v1.0.0",
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in rows
        }
        return [by_id[i] for i in ids if i in by_id]


def _ranks_from_scores(scores: list[float]) -> list[int]:
    """1-based ranks, ties broken by original index."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
