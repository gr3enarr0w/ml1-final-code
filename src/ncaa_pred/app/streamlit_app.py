from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ncaa_pred.data.loader import load_games, load_team_stats, load_teams
from ncaa_pred.env.environment import PredictionEnvironment
from ncaa_pred.models.predictor import WinPredictor
from ncaa_pred.search.retriever import SearchEngine


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "NatSat_MBB_data"
DEFAULT_INDEX_DIR = REPO_ROOT / "artifacts_search" / "mbb_index"


@st.cache_data(show_spinner=False)
def _load_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def _load_search_engine(index_dir: str) -> SearchEngine:
    return SearchEngine.load(index_dir)


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


st.title("NCAA Men's Basketball Prediction + Search")

st.header("MBB Search")
index_dir = st.text_input("Index directory", value=str(DEFAULT_INDEX_DIR))
query = st.text_input("Search query", value="Duke bracket projection")

engine = None
load_error = None
if Path(index_dir).exists():
    try:
        engine = _load_search_engine(index_dir)
    except Exception as exc:
        load_error = str(exc)
else:
    load_error = f"Index not found: {index_dir}"

if load_error:
    st.warning(load_error)
    st.caption(
        "Build/rebuild command: python -m ncaa_pred.search.cli build-index "
        "--data-root /path/to/repo --out-dir /path/to/index"
    )

source_options: list[str] = []
season_options: list[str] = []
conference_options: list[str] = []
if engine is not None:
    source_options = sorted({d.source_type for d in engine.documents if d.source_type})
    season_options = sorted({str(d.season) for d in engine.documents if d.season is not None})
    conference_options = sorted({str(d.conference) for d in engine.documents if d.conference})

col1, col2, col3 = st.columns(3)
with col1:
    selected_sources = st.multiselect("Source type", options=source_options, default=[])
with col2:
    selected_seasons = st.multiselect("Season", options=season_options, default=[])
with col3:
    selected_conferences = st.multiselect("Conference", options=conference_options, default=[])

team_codes_raw = st.text_input("Team codes (comma separated)", value="")
top_k = st.slider("Top K", min_value=3, max_value=25, value=10, step=1)

if st.button("Search"):
    if engine is None:
        st.error("Search index is not loaded.")
    elif not query.strip():
        st.error("Enter a query.")
    else:
        filters = {
            "source_type": selected_sources,
            "season": selected_seasons,
            "conference": selected_conferences,
            "team_code": [x.strip() for x in team_codes_raw.split(",") if x.strip()],
        }
        answer = engine.answer(query=query, filters=filters, top_k=top_k)

        st.subheader("Answer")
        st.write(answer["answer"])

        st.subheader("Cited Results")
        for i, row in enumerate(answer["results"], start=1):
            st.markdown(f"**{i}. {row['title']}**")
            st.caption(
                f"score={row['score_total']:.3f} | bm25={row['score_bm25']:.3f} | "
                f"dense={row['score_dense']:.3f}"
            )
            st.write(row["snippet"])
            st.caption(f"citation: {row['citation']}")

st.divider()
st.header("Prediction (Existing Module)")

candidates = [
    Path("/mnt/data/games.parquet"),
    DEFAULT_DATA_ROOT / "games.parquet",
]
games_path = _first_existing(candidates)
stats_path = _first_existing([
    Path("/mnt/data/team_stats.parquet"),
    DEFAULT_DATA_ROOT / "team_stats.parquet",
])
teams_path = _first_existing([
    Path("/mnt/data/teams.parquet"),
    DEFAULT_DATA_ROOT / "teams.parquet",
])

if not games_path or not stats_path or not teams_path:
    st.info("Prediction data files were not found. Search remains available.")
else:
    games = load_games(str(games_path))
    stats = load_team_stats(str(stats_path))
    _ = load_teams(str(teams_path))

    required_game_cols = {"team_a", "team_b", "result"}
    if not required_game_cols.issubset(set(games.columns)):
        st.info(
            "Prediction environment expects columns team_a/team_b/result. "
            "Current dataset schema differs; search module is unaffected."
        )
    else:
        env = PredictionEnvironment(games, stats)
        predictor = WinPredictor()

        st.subheader("Train Model")
        if st.button("Train", key="train_predictor"):
            X = []
            y = []
            for idx, row in games.iterrows():
                s = env.make_state(idx)
                X.append(s.as_array())
                y.append(row.result)
            X = np.array(X)
            y = np.array(y)
            predictor.fit(X, y)
            st.success("Model trained.")

        st.subheader("Predict Game Outcome")
        selected = st.selectbox("Select game ID", games.index, key="predict_game_id")
        state = env.make_state(selected)
        prob = predictor.predict_proba(state.as_array().reshape(1, -1))[0]
        st.write(f"Win probability for Team A: {prob:.3f}")
