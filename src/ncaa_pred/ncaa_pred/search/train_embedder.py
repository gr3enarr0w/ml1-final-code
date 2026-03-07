from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .corpus_builder import load_corpus
from .types import WeakLabelPair
from .weak_labels import generate_weak_labels, load_weak_labels, save_weak_labels


def _is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _q_prefix(model_name: str) -> str:
    return "query: " if _is_e5_model(model_name) else ""


def _p_prefix(model_name: str) -> str:
    return "passage: " if _is_e5_model(model_name) else ""


def _stratify_key(items: list[WeakLabelPair]) -> list[str]:
    # Keeps 80/20 split approximately balanced by source type.
    return [x.source_type for x in items]


def _mrr_at_k(
    model,
    eval_pairs: list[WeakLabelPair],
    doc_ids: list[str],
    passages: list[str],
    model_name: str,
    k: int = 10,
) -> float:
    if not eval_pairs:
        return 0.0

    pfx = _p_prefix(model_name)
    qfx = _q_prefix(model_name)

    doc_emb = model.encode(
        [pfx + p for p in passages],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    rr_total = 0.0
    for pair in eval_pairs:
        q_emb = model.encode(
            [qfx + pair.query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        sims = doc_emb @ q_emb
        order = np.argsort(-sims)[:k]
        pos_idx = id_to_idx.get(pair.positive_doc_id)
        if pos_idx is None:
            continue

        rr = 0.0
        for rank, idx in enumerate(order, start=1):
            if int(idx) == int(pos_idx):
                rr = 1.0 / rank
                break
        rr_total += rr

    return rr_total / max(1, len(eval_pairs))


def train_embedder(
    corpus_path: str | Path,
    out_dir: str | Path,
    labels_path: str | Path | None = None,
    base_model: str = "intfloat/e5-small-v2",
    fallback_model: str = "all-MiniLM-L6-v2",
    epochs: int = 2,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for embedder fine-tuning. "
            "Install it first (e.g. pip install sentence-transformers)."
        ) from exc

    rng = random.Random(seed)
    np.random.seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = load_corpus(corpus_path)
    doc_by_id = {d.doc_id: d for d in docs}

    if labels_path and Path(labels_path).exists():
        pairs = load_weak_labels(labels_path)
    else:
        pairs = generate_weak_labels(docs, max_pairs=6000, seed=seed)
        labels_path = out / "weak_labels.jsonl"
        save_weak_labels(pairs, labels_path)

    usable: list[WeakLabelPair] = [
        p for p in pairs if p.positive_doc_id in doc_by_id and p.negative_doc_id in doc_by_id
    ]
    if len(usable) < 20:
        raise RuntimeError("Not enough weak-label pairs to train embedder.")

    train_pairs, val_pairs = train_test_split(
        usable,
        test_size=0.2,
        random_state=seed,
        stratify=_stratify_key(usable),
    )

    model_name = base_model
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        model_name = fallback_model
        model = SentenceTransformer(model_name)

    qfx = _q_prefix(model_name)
    pfx = _p_prefix(model_name)

    train_examples = [
        InputExample(
            texts=[qfx + p.query, pfx + doc_by_id[p.positive_doc_id].text],
        )
        for p in train_pairs
    ]

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    best_mrr = -1.0
    best_epoch = -1
    history = []

    for epoch in range(epochs):
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            warmup_steps=max(1, int(0.1 * len(train_loader))),
            show_progress_bar=False,
        )

        sample_docs = rng.sample(docs, k=min(2000, len(docs)))
        eval_doc_ids = [d.doc_id for d in sample_docs]
        eval_passages = [d.text for d in sample_docs]

        # Ensure all positive docs are in candidate pool for valid MRR.
        must_have = {p.positive_doc_id for p in val_pairs}
        missing = [doc_by_id[d] for d in must_have if d not in set(eval_doc_ids) and d in doc_by_id]
        for d in missing:
            eval_doc_ids.append(d.doc_id)
            eval_passages.append(d.text)

        mrr10 = _mrr_at_k(
            model=model,
            eval_pairs=val_pairs,
            doc_ids=eval_doc_ids,
            passages=eval_passages,
            model_name=model_name,
            k=10,
        )

        history.append({"epoch": epoch + 1, "mrr10": mrr10})
        if mrr10 > best_mrr:
            best_mrr = mrr10
            best_epoch = epoch + 1
            model.save(str(out / "best_model"))

    metrics = {
        "base_model_requested": base_model,
        "model_used": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_epoch": best_epoch,
        "best_mrr10": best_mrr,
        "history": history,
        "labels_path": str(labels_path),
    }

    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "train_config.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "fallback_model": fallback_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "query_prefix": qfx,
                "passage_prefix": pfx,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return metrics
