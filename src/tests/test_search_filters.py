from __future__ import annotations

from pathlib import Path

from ncaa_pred.search.corpus_builder import save_corpus
from ncaa_pred.search.index_builder import build_index
from ncaa_pred.search.retriever import SearchEngine
from ncaa_pred.search.types import SearchDocument


def test_search_filters_constrain_results(tmp_path: Path) -> None:
    docs = [
        SearchDocument(
            doc_id="duke_2026",
            source_type="artifact_csv",
            title="Duke 2026",
            text="Duke in ACC 2026 season",
            season=2026,
            team_codes=["DUKE"],
            conference="ACC",
            citation="c1",
        ),
        SearchDocument(
            doc_id="duke_2025",
            source_type="artifact_csv",
            title="Duke 2025",
            text="Duke in ACC 2025 season",
            season=2025,
            team_codes=["DUKE"],
            conference="ACC",
            citation="c2",
        ),
        SearchDocument(
            doc_id="ucla_2026",
            source_type="artifact_csv",
            title="UCLA 2026",
            text="UCLA in BIG10 2026 season",
            season=2026,
            team_codes=["UCLA"],
            conference="BIG10",
            citation="c3",
        ),
    ]

    corpus_path = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    save_corpus(docs, corpus_path)
    build_index(data_root=tmp_path, out_dir=index_dir, corpus_path=corpus_path, force_tfidf=True)

    engine = SearchEngine.load(index_dir)
    results = engine.search(
        "season profile",
        filters={"season": ["2026"], "conference": ["ACC"], "team_code": ["DUKE"]},
        top_k=10,
    )

    assert results
    assert {r.doc_id for r in results} == {"duke_2026"}
