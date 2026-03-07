from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus_builder import build_corpus, save_corpus
from .index_builder import build_index
from .retriever import SearchEngine
from .train_embedder import train_embedder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def cmd_build_index(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    model_name = args.model_dir if args.model_dir else args.model

    meta = build_index(
        data_root=data_root,
        out_dir=out_dir,
        league="MBB",
        model_name=model_name,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        max_rows_per_table=args.max_rows_per_table,
        max_docs=args.max_docs,
        corpus_path=args.corpus if args.corpus else None,
        force_tfidf=args.force_tfidf,
    )
    print(json.dumps(meta, indent=2))


def cmd_train_embedder(args: argparse.Namespace) -> None:
    metrics = train_embedder(
        corpus_path=args.corpus,
        out_dir=args.out_dir,
        labels_path=args.labels,
        base_model=args.base_model,
        fallback_model=args.fallback_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def cmd_build_corpus(args: argparse.Namespace) -> None:
    docs = build_corpus(
        data_root=args.data_root,
        league="MBB",
        max_rows_per_table=args.max_rows_per_table,
    )
    save_corpus(docs, args.output)
    print(json.dumps({"doc_count": len(docs), "output": str(args.output)}, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    engine = SearchEngine.load(args.index_dir)
    filters = {
        "source_type": args.source_type if args.source_type else None,
        "season": args.season if args.season else None,
        "conference": args.conference if args.conference else None,
        "team_code": args.team_code if args.team_code else None,
    }
    answer = engine.answer(query=args.q, filters=filters, top_k=args.top_k)
    print(json.dumps(answer, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NCAA MBB search tooling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_corpus = sub.add_parser("build-corpus", help="Build corpus JSONL from local data sources")
    p_corpus.add_argument("--data-root", default=str(_repo_root()))
    p_corpus.add_argument("--output", required=True)
    p_corpus.add_argument("--max-rows-per-table", type=int, default=10000)
    p_corpus.set_defaults(func=cmd_build_corpus)

    p_index = sub.add_parser("build-index", help="Build hybrid BM25+dense index")
    p_index.add_argument("--data-root", default=str(_repo_root()))
    p_index.add_argument("--out-dir", required=True)
    p_index.add_argument("--corpus", default="")
    p_index.add_argument("--model", default="intfloat/e5-small-v2")
    p_index.add_argument("--model-dir", default="")
    p_index.add_argument("--bm25-weight", type=float, default=0.45)
    p_index.add_argument("--dense-weight", type=float, default=0.55)
    p_index.add_argument("--max-rows-per-table", type=int, default=10000)
    p_index.add_argument("--max-docs", type=int, default=None)
    p_index.add_argument("--force-tfidf", action="store_true")
    p_index.set_defaults(func=cmd_build_index)

    p_train = sub.add_parser("train-embedder", help="Fine-tune sentence-transformer with weak supervision")
    p_train.add_argument("--corpus", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.add_argument("--labels", default="")
    p_train.add_argument("--base-model", default="intfloat/e5-small-v2")
    p_train.add_argument("--fallback-model", default="all-MiniLM-L6-v2")
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train_embedder)

    p_query = sub.add_parser("query", help="Run query against built index")
    p_query.add_argument("--index-dir", required=True)
    p_query.add_argument("--q", required=True)
    p_query.add_argument("--top-k", type=int, default=10)
    p_query.add_argument("--source-type", nargs="*", default=[])
    p_query.add_argument("--season", nargs="*", default=[])
    p_query.add_argument("--conference", nargs="*", default=[])
    p_query.add_argument("--team-code", nargs="*", default=[])
    p_query.set_defaults(func=cmd_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
