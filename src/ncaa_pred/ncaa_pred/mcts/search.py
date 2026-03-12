from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .node import InferenceBatch, InferenceNode


@dataclass
class BASKETConfig:
    random_state: int = 42
    max_iter: int = 1000
    probability_threshold: float = 0.5


class BASKET:
    """
    Inference sandbox for basketball matchup prediction.

    The sandbox learns a lightweight matchup model from team stats and game outcomes,
    then supports single, batch, and Monte-Carlo style inference for what-if testing.
    """

    def __init__(self, config: BASKETConfig | None = None):
        self.config = config or BASKETConfig()
        self.model = LogisticRegression(
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self.team_stats: pd.DataFrame | None = None
        self.feature_names_: list[str] = []
        self.team_a_col_: str | None = None
        self.team_b_col_: str | None = None
        self.label_col_: str | None = None
        self.is_fitted_: bool = False
        self.training_summary_: dict[str, Any] = {}

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip().lower().replace("-", "_") for c in out.columns]
        return out

    @staticmethod
    def _first_present(columns: list[str], candidates: list[str]) -> str | None:
        for c in candidates:
            if c in columns:
                return c
        return None

    def _detect_game_columns(self, games: pd.DataFrame, label_col: str | None = None) -> tuple[str, str, str]:
        cols = list(games.columns)
        team_a = self._first_present(cols, ["team_a", "home_code", "home", "homeid", "home_id"])
        team_b = self._first_present(cols, ["team_b", "visitor_code", "vis_code", "visitor", "away_code", "away", "visitorid", "visitor_id"])

        if team_a is None or team_b is None:
            raise ValueError(
                "Could not infer team columns from games table. "
                "Expected team_a/team_b or home/visitor-style columns."
            )

        if label_col is None:
            label_col = self._first_present(cols, ["result", "home_win", "target", "label"])
            if label_col is None and "winner_code" in cols:
                label_col = "winner_code"

        if label_col is None:
            raise ValueError(
                "Could not infer label column. Provide one of: result, home_win, target, label, winner_code."
            )

        return team_a, team_b, label_col

    def _prepare_team_stats(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        stats = self._normalize_columns(stats_df)
        team_col = self._first_present(list(stats.columns), ["team_code", "code", "teamid", "team_id", "team"])
        if team_col is None:
            raise ValueError("Could not infer team id column in stats table.")

        numeric_cols = [c for c in stats.columns if c != team_col and pd.api.types.is_numeric_dtype(stats[c])]
        if not numeric_cols:
            raise ValueError("No numeric columns found in team stats table.")

        agg = stats.groupby(team_col, as_index=False)[numeric_cols].mean()
        agg = agg.set_index(team_col)
        return agg

    def _label_to_binary(
        self,
        row: pd.Series,
        team_a_col: str,
        label_col: str,
    ) -> int | None:
        val = row[label_col]
        if pd.isna(val):
            return None

        # Handle direct numeric labels first.
        try:
            if label_col in {"result", "home_win", "target", "label"}:
                v = int(float(val))
                return 1 if v > 0 else 0
        except (TypeError, ValueError):
            pass

        if label_col == "winner_code":
            return 1 if str(val) == str(row[team_a_col]) else 0

        # Last-resort parse for true/false text.
        sval = str(val).strip().lower()
        if sval in {"1", "true", "yes", "y", "win", "home"}:
            return 1
        if sval in {"0", "false", "no", "n", "loss", "away"}:
            return 0

        return None

    def _feature_vector(self, team_a: str, team_b: str) -> np.ndarray:
        if self.team_stats is None:
            raise RuntimeError("Team stats are not loaded.")

        if team_a not in self.team_stats.index:
            raise KeyError(f"Team not found in stats table: {team_a}")
        if team_b not in self.team_stats.index:
            raise KeyError(f"Team not found in stats table: {team_b}")

        a = self.team_stats.loc[team_a].to_numpy(dtype=float)
        b = self.team_stats.loc[team_b].to_numpy(dtype=float)
        diff = a - b

        return np.concatenate([a, b, diff])

    def fit(self, games_df: pd.DataFrame, stats_df: pd.DataFrame, label_col: str | None = None) -> dict[str, Any]:
        games = self._normalize_columns(games_df)
        self.team_a_col_, self.team_b_col_, self.label_col_ = self._detect_game_columns(games, label_col=label_col)
        self.team_stats = self._prepare_team_stats(stats_df)

        numeric_cols = list(self.team_stats.columns)
        self.feature_names_ = (
            [f"a_{c}" for c in numeric_cols]
            + [f"b_{c}" for c in numeric_cols]
            + [f"diff_{c}" for c in numeric_cols]
        )

        X: list[np.ndarray] = []
        y: list[int] = []
        dropped = 0

        for _, row in games.iterrows():
            team_a = row[self.team_a_col_]
            team_b = row[self.team_b_col_]
            label = self._label_to_binary(row, self.team_a_col_, self.label_col_)
            if label is None:
                dropped += 1
                continue

            try:
                vec = self._feature_vector(str(team_a), str(team_b))
            except KeyError:
                dropped += 1
                continue

            X.append(vec)
            y.append(label)

        if len(X) < 20:
            raise RuntimeError(f"Not enough aligned training rows to fit model (got {len(X)}).")

        X_arr = np.vstack(X)
        y_arr = np.asarray(y, dtype=int)

        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True

        self.training_summary_ = {
            "n_train": int(len(X_arr)),
            "n_dropped": int(dropped),
            "feature_dim": int(X_arr.shape[1]),
            "positive_rate": float(y_arr.mean()),
            "team_a_col": self.team_a_col_,
            "team_b_col": self.team_b_col_,
            "label_col": self.label_col_,
        }
        return self.training_summary_

    def predict_proba(self, team_a: str, team_b: str) -> float:
        if not self.is_fitted_:
            raise RuntimeError("BASKET sandbox is not fitted. Call fit(...) first.")

        vec = self._feature_vector(str(team_a), str(team_b)).reshape(1, -1)
        return float(self.model.predict_proba(vec)[0, 1])

    def infer(self, team_a: str, team_b: str) -> InferenceNode:
        p = self.predict_proba(team_a, team_b)
        winner = team_a if p >= self.config.probability_threshold else team_b
        return InferenceNode(
            team_a=team_a,
            team_b=team_b,
            p_team_a_win=p,
            predicted_winner=winner,
            metadata={
                "threshold": self.config.probability_threshold,
                "feature_dim": len(self.feature_names_),
            },
        )

    def infer_batch(self, matchups: list[tuple[str, str]]) -> InferenceBatch:
        rows = [self.infer(a, b) for a, b in matchups]
        return InferenceBatch(rows=rows)

    def simulate(self, team_a: str, team_b: str, n_sims: int = 1000) -> dict[str, Any]:
        if n_sims <= 0:
            raise ValueError("n_sims must be positive.")

        p = self.predict_proba(team_a, team_b)
        rng = np.random.default_rng(self.config.random_state)
        wins = rng.binomial(1, p, size=n_sims)
        team_a_wins = int(wins.sum())

        return {
            "team_a": team_a,
            "team_b": team_b,
            "n_sims": int(n_sims),
            "team_a_win_rate": float(team_a_wins / n_sims),
            "team_b_win_rate": float(1.0 - (team_a_wins / n_sims)),
            "base_probability": float(p),
        }


class bbHell:
    """
    Thin wrapper around the BASKET inference sandbox.

    This class is the user-facing integration point requested for experimentation.
    """

    def __init__(self, config: BASKETConfig | None = None):
        self.basket = BASKET(config=config)

    def train(self, games_df: pd.DataFrame, stats_df: pd.DataFrame, label_col: str | None = None) -> dict[str, Any]:
        return self.basket.fit(games_df=games_df, stats_df=stats_df, label_col=label_col)

    def predict(self, team_a: str, team_b: str) -> InferenceNode:
        return self.basket.infer(team_a=team_a, team_b=team_b)

    def predict_many(self, matchups: list[tuple[str, str]]) -> InferenceBatch:
        return self.basket.infer_batch(matchups)

    def sandbox(self, team_a: str, team_b: str, n_sims: int = 1000) -> dict[str, Any]:
        return self.basket.simulate(team_a=team_a, team_b=team_b, n_sims=n_sims)

    @property
    def training_summary(self) -> dict[str, Any]:
        return self.basket.training_summary_
