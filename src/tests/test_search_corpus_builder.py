from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ncaa_pred.search.corpus_builder import build_corpus


def _write_notebook(path: Path) -> None:
    payload = {
        "cells": [
            {"cell_type": "markdown", "source": ["# NCAA Notes\n", "Duke projection details."]},
            {"cell_type": "code", "source": ["print('x')"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_corpus_emits_documents(tmp_path: Path) -> None:
    nat = tmp_path / "NatSat_MBB_data"
    art = tmp_path / "artifacts_modeling"
    nbs = tmp_path / "src" / "notebooks"
    nat.mkdir(parents=True)
    art.mkdir(parents=True)
    nbs.mkdir(parents=True)

    pd.DataFrame(
        [
            {"id": 1, "season": 2026, "home-code": "DUKE", "visitor-code": "UNC", "score-home": 80, "score-vis": 72},
            {"id": 2, "season": 2026, "home-code": "UCLA", "visitor-code": "USC", "score-home": 65, "score-vis": 70},
        ]
    ).to_parquet(nat / "games.parquet", index=False)

    pd.DataFrame(
        [
            {"team_code": "DUKE", "season": 2026, "o_ppp": 1.2, "pace": 68.0},
            {"team_code": "UNC", "season": 2026, "o_ppp": 1.1, "pace": 70.0},
        ]
    ).to_parquet(nat / "team_stats.parquet", index=False)

    pd.DataFrame(
        [{"code": "DUKE", "name": "Duke Blue Devils"}, {"code": "UNC", "name": "North Carolina Tar Heels"}]
    ).to_parquet(nat / "teams.parquet", index=False)

    pd.DataFrame(
        [{"code": "DUKE", "team": "Duke Blue Devils", "elo": 2100, "elorank": 1}]
    ).to_parquet(nat / "elo.parquet", index=False)

    pd.DataFrame(
        [{"season": 2026, "team_code": "DUKE", "selection_prob": 0.98, "team_name": "Duke Blue Devils"}]
    ).to_csv(art / "live_2026_selection_mbb.csv", index=False)

    (art / "source_config.json").write_text('{"seed": 42}', encoding="utf-8")
    _write_notebook(nbs / "NCAA_MBB_v7.ipynb")
    (tmp_path / "project_outline").write_text("NCAA plan text", encoding="utf-8")
    (tmp_path / "natstat API v3.5 Documentation").write_text("NatStat endpoint reference", encoding="utf-8")

    docs = build_corpus(tmp_path)

    assert len(docs) > 10
    source_types = {d.source_type for d in docs}
    assert "games" in source_types
    assert "team_stats" in source_types
    assert "teams" in source_types
    assert "artifact_csv" in source_types
    assert "notebook" in source_types
    assert "doc_text" in source_types
