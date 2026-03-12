from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .bm25 import BM25Index, tokenize
from .corpus_builder import build_corpus, load_corpus, save_corpus
from .embedding_backend import EmbeddingBackend
from .types import SearchDocument


def _doc_text(doc: SearchDocument) -> str:
    return f"{doc.title}\n{doc.text}"


def build_index(
    data_root: str | Path,
    out_dir: str | Path,
    league: str = "MBB",
    model_name: str = "intfloat/e5-small-v2",
    bm25_weight: float = 0.45,
    dense_weight: float = 0.55,
    max_rows_per_table: int = 10000,
    max_docs: int | None = None,
    corpus_path: str | Path | None = None,
    force_tfidf: bool = False,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs: list[SearchDocument]
    if corpus_path is not None and Path(corpus_path).exists():
        docs = load_corpus(corpus_path)
    else:
        docs = build_corpus(data_root=data_root, league=league, max_rows_per_table=max_rows_per_table)

    docs = sorted(docs, key=lambda d: d.doc_id)
    if max_docs is not None:
        docs = docs[:max_docs]
    if not docs:
        raise RuntimeError("No documents were produced for indexing.")

    doc_path = out / "documents.jsonl"
    save_corpus(docs, doc_path)

    texts = [_doc_text(d) for d in docs]
    tokenized = [tokenize(t) for t in texts]

    bm25 = BM25Index().fit(tokenized)
    with (out / "bm25.pkl").open("wb") as fh:
        pickle.dump(bm25, fh)

    backend = EmbeddingBackend(model_name=model_name, force_tfidf=force_tfidf)
    embeddings = backend.fit_transform_passages(texts)
    np.save(out / "embeddings.npy", embeddings)
    backend.save(out)

    meta = {
        "league": league,
        "doc_count": len(docs),
        "embedding_dim": int(embeddings.shape[1]),
        "bm25_weight": float(bm25_weight),
        "dense_weight": float(dense_weight),
        "model_name": backend.config.model_name,
        "embedding_backend": backend.config.backend,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta
