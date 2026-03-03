import pandas as pd
from pathlib import Path

def load_games(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))

def load_team_stats(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))

def load_teams(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))
