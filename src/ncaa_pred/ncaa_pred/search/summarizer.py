from __future__ import annotations

import re

from .types import SearchResult


def _first_sentence(text: str, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    m = re.search(r"[.!?]", text)
    sentence = text[: m.end()] if m else text[:max_chars]
    return sentence[:max_chars].strip()


def summarize_results(query: str, results: list[SearchResult], max_sentences: int = 3) -> dict:
    if not results:
        return {
            "query": query,
            "answer": "No matching evidence was found in the local NCAA MBB index.",
            "citations": [],
        }

    used = results[:max_sentences]
    sentences = []
    citations = []
    for r in used:
        snippet = _first_sentence(r.snippet)
        citation = r.citation
        if not snippet:
            continue
        sentences.append(f"{snippet} [{citation}]")
        citations.append(citation)

    if not sentences:
        answer = "I found results, but none had enough extractable text to summarize."
    else:
        answer = " ".join(sentences)

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
    }
