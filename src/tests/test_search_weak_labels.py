from __future__ import annotations

from ncaa_pred.search.types import SearchDocument
from ncaa_pred.search.weak_labels import generate_weak_labels


def test_generate_weak_labels_has_pos_neg_and_unique_ids() -> None:
    docs = [
        SearchDocument(
            doc_id="d1",
            source_type="artifact_csv",
            title="Duke selection",
            text="Duke has high selection probability.",
            season=2026,
            team_codes=["DUKE"],
            conference="ACC",
            citation="x:1",
        ),
        SearchDocument(
            doc_id="d2",
            source_type="artifact_csv",
            title="UNC selection",
            text="UNC selection profile.",
            season=2026,
            team_codes=["UNC"],
            conference="ACC",
            citation="x:2",
        ),
        SearchDocument(
            doc_id="d3",
            source_type="artifact_csv",
            title="UCLA selection",
            text="UCLA selection profile.",
            season=2026,
            team_codes=["UCLA"],
            conference="BIG10",
            citation="x:3",
        ),
    ]

    pairs = generate_weak_labels(docs, max_pairs=20, seed=7)

    assert pairs
    ids = {p.pair_id for p in pairs}
    assert len(ids) == len(pairs)
    assert all(p.positive_doc_id != p.negative_doc_id for p in pairs)
