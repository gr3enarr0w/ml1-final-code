# Project structure for a UV-based NCAA prediction package

# Directory:
# ncaa_pred/
#   __init__.py
#   data/
#       __init__.py
#       loader.py
#   env/
#       __init__.py
#       state.py
#       environment.py
#   models/
#       __init__.py
#       predictor.py
#   mcts/
#       __init__.py
#       node.py
#       search.py
# app/
#   streamlit_app.py
# pyproject.toml

# =====================
# pyproject.toml
# =====================
pyproject_content = """
[project]
name = "ncaa_pred"
version = "0.1.0"
description = "NCAA men's basketball prediction-market package using state/env logic"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project.dependencies]
pandas = "*"
pyarrow = "*"
streamlit = "*"
numpy = "*"
scikit-learn = "*"
"""

# =====================
# data/loader.py
# =====================
data_loader_code = """
import pandas as pd
from pathlib import Path

def load_games(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))

def load_team_stats(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))

def load_teams(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))
"""

# =====================
# env/state.py
# =====================
state_code = """
import numpy as np

class GameState:
    def __init__(self, team_a: int, team_b: int, features: np.ndarray):
        self.team_a = team_a
        self.team_b = team_b
        self.features = features

    def as_array(self) -> np.ndarray:
        return self.features
"""

# =====================
# env/environment.py
# =====================
env_code = """
from .state import GameState
import numpy as np

class PredictionEnvironment:
    def __init__(self, games_df, stats_df):
        self.games = games_df
        self.stats = stats_df

    def make_state(self, game_id: int) -> GameState:
        row = self.games.loc[game_id]
        team_a = row.team_a
        team_b = row.team_b
        feat_a = self.stats.loc[team_a].values
        feat_b = self.stats.loc[team_b].values
        features = np.concatenate([feat_a, feat_b])
        return GameState(team_a, team_b, features)
"""

# =====================
# models/predictor.py
# =====================
predictor_code = """
import numpy as np
from sklearn.linear_model import LogisticRegression

class WinPredictor:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)[:, 1]
"""

# =====================
# app/streamlit_app.py
# =====================
streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np
from ncaa_pred.data.loader import load_games, load_team_stats, load_teams
from ncaa_pred.env.environment import PredictionEnvironment
from ncaa_pred.models.predictor import WinPredictor

st.title("NCAA Men's Basketball Prediction Market")

games = load_games("/mnt/data/games.parquet")
stats = load_team_stats("/mnt/data/team_stats.parquet")
teams = load_teams("/mnt/data/teams.parquet")

env = PredictionEnvironment(games, stats)
predictor = WinPredictor()

st.subheader("Train Model")
if st.button("Train"):
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
selected = st.selectbox("Select game ID", games.index)
state = env.make_state(selected)
prob = predictor.predict_proba(state.as_array().reshape(1, -1))[0]
st.write(f"Win probability for Team A: {prob:.3f}")
"""
