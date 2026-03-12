from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EmbeddingConfig:
    backend: str
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EmbeddingConfig":
        return cls(**payload)


class EmbeddingBackend:
    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        force_tfidf: bool = True,
    ):
        self.model_name = model_name
        self.force_tfidf = force_tfidf
        self._vectorizer: TfidfVectorizer | None = None
        self._st_model = None
        self.config = EmbeddingConfig(
            backend="tfidf",
            model_name=model_name,
            query_prefix="",
            passage_prefix="",
        )

    def _maybe_load_sentence_transformer(self) -> bool:
        if self.force_tfidf:
            return False
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return False

        self._st_model = SentenceTransformer(self.model_name)
        if "e5" in self.model_name.lower():
            self.config = EmbeddingConfig(
                backend="sentence-transformers",
                model_name=self.model_name,
                query_prefix="query: ",
                passage_prefix="passage: ",
            )
        else:
            self.config = EmbeddingConfig(
                backend="sentence-transformers",
                model_name=self.model_name,
                query_prefix="",
                passage_prefix="",
            )
        return True

    def fit_transform_passages(self, passages: list[str]) -> np.ndarray:
        if self._maybe_load_sentence_transformer():
            encoded = self._st_model.encode(
                [self.config.passage_prefix + p for p in passages],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return encoded.astype(np.float32)

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8192)
        matrix = self._vectorizer.fit_transform(passages)
        self.config = EmbeddingConfig(
            backend="tfidf",
            model_name="tfidf",
            query_prefix="",
            passage_prefix="",
        )
        return matrix.toarray().astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if self.config.backend == "sentence-transformers":
            if self._st_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except Exception as exc:
                    raise RuntimeError("sentence-transformers is required to encode query") from exc
                self._st_model = SentenceTransformer(self.config.model_name)
            vec = self._st_model.encode(
                [self.config.query_prefix + query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vec[0].astype(np.float32)

        if self._vectorizer is None:
            raise RuntimeError("TF-IDF backend is not loaded")
        return self._vectorizer.transform([query]).toarray().astype(np.float32)[0]

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "embedding_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2),
            encoding="utf-8",
        )
        if self.config.backend == "tfidf" and self._vectorizer is not None:
            with (out / "tfidf_vectorizer.pkl").open("wb") as fh:
                pickle.dump(self._vectorizer, fh)

    @classmethod
    def load(cls, index_dir: str | Path) -> "EmbeddingBackend":
        idx = Path(index_dir)
        cfg_path = idx / "embedding_config.json"
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = EmbeddingConfig.from_dict(payload)

        obj = cls(model_name=cfg.model_name, force_tfidf=(cfg.backend == "tfidf"))
        obj.config = cfg

        if cfg.backend == "tfidf":
            with (idx / "tfidf_vectorizer.pkl").open("rb") as fh:
                obj._vectorizer = pickle.load(fh)
        return obj


def cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    # matrix is expected to be float32 and either already normalized (ST) or sparse-like dense (TF-IDF).
    q = query_vec.astype(np.float32)
    q_norm = np.linalg.norm(q) + 1e-12
    m_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ q) / (m_norm * q_norm)
