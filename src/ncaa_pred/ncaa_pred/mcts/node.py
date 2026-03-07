from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BasketMatchup:
    team_a: str
    team_b: str


@dataclass
class InferenceNode:
    team_a: str
    team_b: str
    p_team_a_win: float
    predicted_winner: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceBatch:
    rows: list[InferenceNode]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "team_a": r.team_a,
                "team_b": r.team_b,
                "p_team_a_win": r.p_team_a_win,
                "predicted_winner": r.predicted_winner,
                "metadata": r.metadata,
            }
            for r in self.rows
        ]
