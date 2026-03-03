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