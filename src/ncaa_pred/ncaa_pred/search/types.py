from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SearchDocument:
    doc_id: str
    source_type: str
    title: str
    text: str
    season: int | None = None
    team_codes: list[str] = field(default_factory=list)
    conference: str | None = None
    round: str | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)
    citation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchDocument":
        return cls(**payload)


@dataclass
class SearchResult:
    doc_id: str
    score_total: float
    score_bm25: float
    score_dense: float
    title: str
    snippet: str
    citation: str
    metadata_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeakLabelPair:
    pair_id: str
    query: str
    positive_doc_id: str
    negative_doc_id: str
    source_type: str
    season: int | None = None
    conference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeakLabelPair":
        return cls(**payload)
