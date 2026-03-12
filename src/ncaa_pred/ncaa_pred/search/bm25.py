from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: np.ndarray | None = None
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.term_freqs: list[Counter] = []

    def fit(self, tokenized_docs: list[list[str]]) -> "BM25Index":
        n_docs = len(tokenized_docs)
        self.term_freqs = [Counter(doc) for doc in tokenized_docs]
        self.doc_len = np.array([len(doc) for doc in tokenized_docs], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if n_docs else 0.0

        df = defaultdict(int)
        for doc in tokenized_docs:
            for term in set(doc):
                df[term] += 1

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        return self

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        if self.doc_len is None:
            raise RuntimeError("BM25 index not fitted")

        scores = np.zeros(len(self.term_freqs), dtype=np.float32)
        for q in query_tokens:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(q, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl + 1e-12))
                scores[i] += idf * (f * (self.k1 + 1)) / (denom + 1e-12)
        return scores
