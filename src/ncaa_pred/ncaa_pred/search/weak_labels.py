from __future__ import annotations

import json
import random
from pathlib import Path

from .types import SearchDocument, WeakLabelPair


def _query_templates(doc: SearchDocument) -> list[str]:
    queries: list[str] = []
    base = doc.title.strip()
    if base:
        queries.append(base)

    if doc.team_codes:
        team = doc.team_codes[0]
        queries.append(f"{team} NCAA MBB stats")
        if doc.season is not None:
            queries.append(f"{team} season {doc.season} performance")

    if doc.conference:
        conf = doc.conference
        if doc.season is not None:
            queries.append(f"top {conf} teams by winpct {doc.season}")
        else:
            queries.append(f"{conf} conference MBB rankings")

    if doc.round:
        queries.append(f"{doc.round} bracket projection")

    if doc.source_type == "artifact_csv":
        queries.append("ncaa bracket projection probabilities")
        queries.append("selection model output")

    if not queries:
        queries.append(f"NCAA MBB {doc.source_type}")

    seen: set[str] = set()
    uniq: list[str] = []
    for q in queries:
        q_norm = " ".join(q.split()).strip()
        if q_norm and q_norm not in seen:
            seen.add(q_norm)
            uniq.append(q_norm)
    return uniq[:3]


def _has_team_overlap(a: SearchDocument, b: SearchDocument) -> bool:
    return bool(set(a.team_codes).intersection(set(b.team_codes)))


def _pick_hard_negative(doc: SearchDocument, docs: list[SearchDocument], rng: random.Random) -> SearchDocument | None:
    if len(docs) < 2:
        return None

    same_source = [d for d in docs if d.doc_id != doc.doc_id and d.source_type == doc.source_type]
    same_season = [d for d in same_source if doc.season is not None and d.season == doc.season]
    same_conf = [d for d in same_season if doc.conference is not None and d.conference == doc.conference]

    for pool in (same_conf, same_season, same_source, docs):
        candidates = [d for d in pool if d.doc_id != doc.doc_id and not _has_team_overlap(doc, d)]
        if candidates:
            return rng.choice(candidates)
    return None


def generate_weak_labels(
    documents: list[SearchDocument],
    max_pairs: int = 5000,
    seed: int = 42,
) -> list[WeakLabelPair]:
    rng = random.Random(seed)
    pairs: list[WeakLabelPair] = []

    shuffled = documents[:]
    rng.shuffle(shuffled)

    for doc in shuffled:
        negative = _pick_hard_negative(doc, shuffled, rng)
        if negative is None:
            continue

        for query in _query_templates(doc):
            pair = WeakLabelPair(
                pair_id=f"wl_{len(pairs)}",
                query=query,
                positive_doc_id=doc.doc_id,
                negative_doc_id=negative.doc_id,
                source_type=doc.source_type,
                season=doc.season,
                conference=doc.conference,
            )
            pairs.append(pair)
            if len(pairs) >= max_pairs:
                return pairs

    return pairs


def save_weak_labels(pairs: list[WeakLabelPair], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_weak_labels(path: str | Path) -> list[WeakLabelPair]:
    src = Path(path)
    items: list[WeakLabelPair] = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(WeakLabelPair.from_dict(json.loads(line)))
    return items
