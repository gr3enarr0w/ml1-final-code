from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .types import SearchDocument

# this is parsing the column names to normalize them into snake_case
# we have begin model building
TOKEN_RE = re.compile(r"[^a-zA-Z0-9]+")


def normalize_column_name(name: str) -> str:
    return TOKEN_RE.sub("_", name.strip().lower()).strip("_")


def _normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(str(c)) for c in out.columns]
    return out


def _safe_to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        return None
    if num < 1900 or num > 2100:
        return None
    return num


def _extract_first_present(row: pd.Series, candidates: Iterable[str]) -> Any:
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    return None


def _extract_team_codes(row: pd.Series) -> list[str]:
    candidates = [
        "team_code",
        "home_code",
        "visitor_code",
        "vis_code",
        "winner_code",
        "loser_code",
        "code",
    ]
    team_codes: list[str] = []
    for c in candidates:
        if c in row and pd.notna(row[c]):
            value = str(row[c]).strip()
            if value and value not in team_codes:
                team_codes.append(value)
    return team_codes


def _row_text(row: pd.Series, keep_cols: list[str] | None = None) -> str:
    parts: list[str] = []
    cols = keep_cols if keep_cols is not None else list(row.index)
    for c in cols:
        if c not in row:
            continue
        value = row[c]
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        parts.append(f"{c}: {value_str}")
    return " | ".join(parts)


def _build_docs_from_df(
    df: pd.DataFrame,
    source_type: str,
    file_path: Path,
    max_rows: int,
    title_col_candidates: list[str],
    round_col_candidates: list[str],
) -> list[SearchDocument]:
    docs: list[SearchDocument] = []
    if df.empty:
        return docs

    capped = df.head(max_rows)
    for row_idx, row in capped.iterrows():
        title_value = _extract_first_present(row, title_col_candidates)
        title = (
            f"{source_type} | {title_value}"
            if title_value is not None
            else f"{source_type} row {row_idx}"
        )
        season = _safe_to_int(_extract_first_present(row, ["season", "year", "season_year"]))
        conference_val = _extract_first_present(row, ["conference", "conference_proxy", "conf", "league"])
        round_val = _extract_first_present(row, round_col_candidates)
        text = _row_text(row)
        if not text:
            continue

        doc_id = f"{source_type}:{file_path.name}:{row_idx}"
        docs.append(
            SearchDocument(
                doc_id=doc_id,
                source_type=source_type,
                title=title,
                text=text,
                season=season,
                team_codes=_extract_team_codes(row),
                conference=str(conference_val) if conference_val is not None else None,
                round=str(round_val) if round_val is not None else None,
                metadata_json={
                    "path": str(file_path),
                    "row_idx": int(row_idx) if isinstance(row_idx, (int, float)) else str(row_idx),
                },
                citation=f"{file_path}:row={row_idx}",
            )
        )
    return docs


def _read_elo_compact(path: Path) -> pd.DataFrame:
    # Try known compact schemas first to avoid loading the ultra-wide version.
    attempts = [
        ["code", "team", "elo", "elorank"],
        ["code", "elo", "elorank"],
        ["team", "elo", "elorank"],
        ["code", "team", "elo"],
        ["elo", "elorank"],
        ["elo"],
    ]
    for cols in attempts:
        try:
            df = pd.read_parquet(path, columns=cols)
            return _normalize_df_columns(df)
        except Exception:
            continue
    return pd.DataFrame()


def _notebook_markdown_docs(path: Path, max_cells: int = 300) -> list[SearchDocument]:
    docs: list[SearchDocument] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return docs

    cells = payload.get("cells", [])
    for idx, cell in enumerate(cells[:max_cells]):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        text = re.sub(r"\s+", " ", source).strip()
        if not text:
            continue
        title = text[:80]
        docs.append(
            SearchDocument(
                doc_id=f"notebook:{path.name}:cell={idx}",
                source_type="notebook",
                title=title,
                text=text,
                metadata_json={"path": str(path), "cell_idx": idx},
                citation=f"{path}:cell={idx}",
            )
        )
    return docs


def _doc_text_docs(path: Path, max_chars: int = 200_000) -> list[SearchDocument]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    text = re.sub(r"\s+", " ", text).strip()[:max_chars]
    if not text:
        return []

    chunk_size = 1200
    chunks: list[SearchDocument] = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        chunks.append(
            SearchDocument(
                doc_id=f"doc:{path.name}:chunk={i // chunk_size}",
                source_type="doc_text",
                title=f"{path.name} chunk {i // chunk_size}",
                text=chunk,
                metadata_json={"path": str(path), "chunk_idx": i // chunk_size},
                citation=f"{path}:chunk={i // chunk_size}",
            )
        )
    return chunks


def build_corpus(
    data_root: str | Path,
    league: str = "MBB",
    max_rows_per_table: int = 10000,
) -> list[SearchDocument]:
    root = Path(data_root).resolve()
    docs: list[SearchDocument] = []

    if league.upper() != "MBB":
        raise ValueError("Only MBB is supported in v1.")

    nat_dir = root / "NatSat_MBB_data"
    artifacts_dir = root / "artifacts_modeling"
    notebooks_dir = root / "src" / "notebooks"

    table_specs = [
        (nat_dir / "games.parquet", "games", ["home", "visitor", "id"], ["round", "round_num", "gameno"]),
        (nat_dir / "team_stats.parquet", "team_stats", ["team", "team_code", "gameid"], ["round", "round_num"]),
        (nat_dir / "teams.parquet", "teams", ["name", "full_name", "code"], ["round", "round_num"]),
    ]

    for path, source_type, title_cols, round_cols in table_specs:
        if not path.exists():
            continue
        try:
            df = _normalize_df_columns(pd.read_parquet(path))
        except Exception:
            continue
        docs.extend(
            _build_docs_from_df(
                df=df,
                source_type=source_type,
                file_path=path,
                max_rows=max_rows_per_table,
                title_col_candidates=title_cols,
                round_col_candidates=round_cols,
            )
        )

    elo_path = nat_dir / "elo.parquet"
    if elo_path.exists():
        elo_df = _read_elo_compact(elo_path)
        if not elo_df.empty:
            docs.extend(
                _build_docs_from_df(
                    df=elo_df,
                    source_type="elo",
                    file_path=elo_path,
                    max_rows=max_rows_per_table,
                    title_col_candidates=["team", "code"],
                    round_col_candidates=["round", "round_num"],
                )
            )

    if artifacts_dir.exists():
        for csv_path in sorted(artifacts_dir.glob("*.csv")):
            if league.upper() == "MBB" and "wbb" in csv_path.name.lower():
                continue
            try:
                df = _normalize_df_columns(pd.read_csv(csv_path))
            except Exception:
                continue
            if league.upper() == "MBB" and "league" in df.columns:
                league_mask = df["league"].astype(str).str.upper().isin(["MBB", "NAN", "NONE", ""])
                df = df[league_mask].copy()
            if league.upper() == "MBB" and not df.empty:
                object_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
                if object_cols:
                    wbb_mask = df[object_cols].astype(str).apply(
                        lambda col: col.str.contains(r"\bwbb\b|women", case=False, regex=True, na=False)
                    )
                    df = df[~wbb_mask.any(axis=1)].copy()
            docs.extend(
                _build_docs_from_df(
                    df=df,
                    source_type="artifact_csv",
                    file_path=csv_path,
                    max_rows=max_rows_per_table,
                    title_col_candidates=["team_name", "team", "model_family", "task", "round"],
                    round_col_candidates=["round", "round_num"],
                )
            )

        for json_path in sorted(artifacts_dir.glob("*.json")):
            if league.upper() == "MBB" and "wbb" in json_path.name.lower():
                continue
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if league.upper() == "MBB":
                payload_text = json.dumps(payload, sort_keys=True)
                if re.search(r"\bwbb\b|women", payload_text, flags=re.IGNORECASE):
                    continue
            docs.append(
                SearchDocument(
                    doc_id=f"artifact_json:{json_path.name}",
                    source_type="artifact_json",
                    title=json_path.name,
                    text=json.dumps(payload, sort_keys=True),
                    metadata_json={"path": str(json_path)},
                    citation=str(json_path),
                )
            )

    if notebooks_dir.exists():
        for ipynb in sorted(notebooks_dir.glob("*.ipynb")):
            docs.extend(_notebook_markdown_docs(ipynb))

    additional_text_files = [
        root / "project_outline",
        root / "natstat API v3.5 Documentation",
        root / "README.md",
    ]
    for p in additional_text_files:
        if p.exists():
            docs.extend(_doc_text_docs(p))

    return docs


def save_corpus(documents: list[SearchDocument], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in documents:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")


def load_corpus(path: str | Path) -> list[SearchDocument]:
    src = Path(path)
    docs: list[SearchDocument] = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            docs.append(SearchDocument.from_dict(json.loads(line)))
    return docs
