from __future__ import annotations

from pathlib import Path

from ncaa_pred.search.corpus_builder import save_corpus
from ncaa_pred.search.index_builder import build_index
from ncaa_pred.search.retriever import SearchEngine
from ncaa_pred.search.types import SearchDocument


def _mini_docs() -> list[SearchDocument]:
    return [
        SearchDocument(
            doc_id="doc_duke",
            source_type="artifact_csv",
            title="Duke bracket projection",
            text="Duke Blue Devils projected as a top seed in 2026 bracket.",
            season=2026,
            team_codes=["DUKE"],
            conference="ACC",
            citation="artifact.csv:row=1",
        ),
        SearchDocument(
            doc_id="doc_ucla",
            source_type="artifact_csv",
            title="UCLA team profile",
            text="UCLA Bruins offense and pace profile for 2026 season.",
            season=2026,
            team_codes=["UCLA"],
            conference="BIG10",
            citation="artifact.csv:row=2",
        ),
    ]


def test_retriever_ranking_is_deterministic(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    save_corpus(_mini_docs(), corpus_path)

    build_index(
        data_root=tmp_path,
        out_dir=index_dir,
        corpus_path=corpus_path,
        force_tfidf=True,
    )

    engine = SearchEngine.load(index_dir)
    r1 = engine.search("Duke bracket projection", top_k=2)
    r2 = engine.search("Duke bracket projection", top_k=2)

    assert r1[0].doc_id == "doc_duke"
    assert [x.doc_id for x in r1] == [x.doc_id for x in r2]
