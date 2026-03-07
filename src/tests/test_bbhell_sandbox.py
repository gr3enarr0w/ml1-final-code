from __future__ import annotations

import numpy as np
import pandas as pd

from ncaa_pred.mcts.search import BASKETConfig, bbHell


def _toy_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_code": ["DUKE", "UNC", "UCLA", "KU"],
            "off_rating": [118.0, 112.0, 108.0, 110.0],
            "def_rating": [95.0, 99.0, 101.0, 100.0],
            "pace": [69.0, 71.0, 67.0, 68.0],
        }
    )


def _toy_games(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["DUKE", "UNC", "UCLA", "KU"]
    strength = {"DUKE": 1.3, "UNC": 0.6, "UCLA": -0.5, "KU": -0.2}

    rows = []
    for _ in range(n):
        a, b = rng.choice(teams, size=2, replace=False)
        noise = rng.normal(0, 0.4)
        a_wins = (strength[a] - strength[b] + noise) > 0
        rows.append(
            {
                "home_code": a,
                "visitor_code": b,
                "winner_code": a if a_wins else b,
            }
        )
    return pd.DataFrame(rows)


def test_bbhell_train_predict_and_simulate() -> None:
    games = _toy_games()
    stats = _toy_stats()

    sandbox = bbHell(config=BASKETConfig(random_state=11))
    summary = sandbox.train(games_df=games, stats_df=stats)

    assert summary["n_train"] > 50
    pred = sandbox.predict("DUKE", "UCLA")
    assert 0.0 <= pred.p_team_a_win <= 1.0
    assert pred.predicted_winner in {"DUKE", "UCLA"}

    batch = sandbox.predict_many([("DUKE", "UNC"), ("UCLA", "KU")])
    assert len(batch.rows) == 2

    sim = sandbox.sandbox("DUKE", "UNC", n_sims=500)
    assert sim["team_a"] == "DUKE"
    assert sim["team_b"] == "UNC"
    assert 0.0 <= sim["team_a_win_rate"] <= 1.0
