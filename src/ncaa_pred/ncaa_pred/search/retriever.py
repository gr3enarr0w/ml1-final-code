from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .bm25 import tokenize
from .corpus_builder import load_corpus
from .embedding_backend import EmbeddingBackend, cosine_scores
from .summarizer import summarize_results
from .types import SearchDocument, SearchResult


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _snippet(text: str, max_chars: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _normalize_filter_list(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(x) for x in raw}
    return {str(raw)}


class SearchEngine:
    def __init__(
        self,
        documents: list[SearchDocument],
        bm25,
        embeddings: np.ndarray,
        backend: EmbeddingBackend,
        bm25_weight: float,
        dense_weight: float,
    ):
        self.documents = documents
        self.bm25 = bm25
        self.embeddings = embeddings
        self.backend = backend
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        self.doc_by_id = {d.doc_id: d for d in documents}

    @classmethod
    def load(cls, index_dir: str | Path) -> "SearchEngine":
        idx = Path(index_dir)
        docs = load_corpus(idx / "documents.jsonl")

        with (idx / "bm25.pkl").open("rb") as fh:
            bm25 = pickle.load(fh)

        embeddings = np.load(idx / "embeddings.npy")
        backend = EmbeddingBackend.load(idx)

        meta = json.loads((idx / "index_meta.json").read_text(encoding="utf-8"))
        bm25_weight = float(meta.get("bm25_weight", 0.45))
        dense_weight = float(meta.get("dense_weight", 0.55))

        return cls(
            documents=docs,
            bm25=bm25,
            embeddings=embeddings,
            backend=backend,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def _matches_filters(self, doc: SearchDocument, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True

        source_types = _normalize_filter_list(filters.get("source_type"))
        if source_types and doc.source_type not in source_types:
            return False

        seasons = _normalize_filter_list(filters.get("season"))
        if seasons and str(doc.season) not in seasons:
            return False

        conferences = _normalize_filter_list(filters.get("conference"))
        if conferences and (doc.conference is None or str(doc.conference) not in conferences):
            return False

        team_codes = _normalize_filter_list(filters.get("team_code"))
        if team_codes and not set(doc.team_codes).intersection(team_codes):
            return False

        return True

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        q_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)

        q_emb = self.backend.encode_query(query)
        dense_scores = cosine_scores(self.embeddings, q_emb)

        candidate_indices = [i for i, d in enumerate(self.documents) if self._matches_filters(d, filters)]
        if not candidate_indices:
            return []

        bm25_sub = bm25_scores[candidate_indices]
        dense_sub = dense_scores[candidate_indices]
        bm25_norm = _minmax(bm25_sub)
        dense_norm = _minmax(dense_sub)
        total = self.bm25_weight * bm25_norm + self.dense_weight * dense_norm

        order = np.argsort(-total)[:top_k]

        results: list[SearchResult] = []
        for rank_idx in order:
            global_idx = candidate_indices[int(rank_idx)]
            doc = self.documents[global_idx]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    score_total=float(total[int(rank_idx)]),
                    score_bm25=float(bm25_norm[int(rank_idx)]),
                    score_dense=float(dense_norm[int(rank_idx)]),
                    title=doc.title,
                    snippet=_snippet(doc.text),
                    citation=doc.citation,
                    metadata_json=doc.metadata_json,
                )
            )
        return results

    def answer(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> dict:
        results = self.search(query=query, filters=filters, top_k=top_k)
        summary = summarize_results(query=query, results=results)
        return {
            "query": query,
            "answer": summary["answer"],
            "citations": summary["citations"],
            "results": [r.to_dict() for r in results],
        }
