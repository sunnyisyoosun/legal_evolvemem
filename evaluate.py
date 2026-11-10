"""Evaluate one retrieval configuration on a legal benchmark.

Scoring goes through multilingual_bm25_recall.compute_query_metrics, the same
function the project's BM25 baseline reports, so numbers produced here sit in
the same column as the existing results. Every run also writes a run file
({query_id: [doc_id, ...]}) so any method can be re-scored later without
re-running retrieval.
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
from legal_meta import LANG_OF
from evolvemem.multi_retriever import RetrievalConfig

CASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
QRELS = "/home/yoosun/legal/output/{ds}_qrels.json"
POOL_PATH = {"LeCaRD": "/home/yoosun/legal/result/LeCaRD/candidate_pool.json"}
EVAL_K = [1, 5, 10, 20, 30]
PRIMARY = "NDCG@30"


def load_pool(dataset: str):
    """Per-query candidate pool, for benchmarks whose official task is re-ranking."""
    path = POOL_PATH.get(dataset)
    if not path:
        return None
    return {str(k): [str(x) for x in v] for k, v in json.load(open(path)).items()}


def fold_of(row: dict) -> str:
    """Which fold a query belongs to: train / val / test.

    `fold` is written by classify_queries; rows from an older build that
    predate it fall back to `split`, where everything not test is train.
    """
    return row.get("fold") or row.get("split") or "train"


def in_split(row: dict, split: str | None) -> bool:
    return split in (None, "all") or fold_of(row) == split


def weak_config() -> RetrievalConfig:
    """BM25-only start point, matching the paper's minimal theta_0."""
    return RetrievalConfig(
        fusion_mode="keyword_only", keyword_top_k=100, semantic_top_k=0,
        structured_top_k=0, max_context=100, enable_entity_swap=False,
    )


def evaluate(dataset: str, config: RetrievalConfig, split: str = "test",
             top_n: int = 30, index=None, suffix: str = "", quiet: bool = False,
             use_pool: bool = False, legal_class: str | None = None):
    """Full-corpus retrieval by default.

    LeCaRD ships a 100-candidate pool per query and its official task is to
    re-rank that pool, but every other benchmark here searches the whole
    corpus. Comparing a pooled number against an unpooled one is comparing
    two different tasks, so the project evaluates all four the same way and
    `use_pool` is opt-in for reproducing LeCaRD's official protocol.
    """
    cases = json.load(open(f"{CASE_DIR}/{dataset}_cases{suffix}.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{dataset}_queries{suffix}.json", encoding="utf-8"))
    labels = L.MBR.load_qrels(QRELS.format(ds=dataset))
    pool = load_pool(dataset) if use_pool else None

    if index is None:
        t0 = time.time()
        index = L.build_index(dataset, cases, qs)
        if not quiet:
            print(f"  index: {len(cases)} docs, {time.time() - t0:.1f}s", flush=True)

    rows = [q for q in qs if q["query_id"] in labels and in_split(q, split)
            and (legal_class is None or q.get("legal_class") == legal_class)]
    run, agg = {}, {}
    t0 = time.time()
    for q in rows:
        ranked = L.search(index, q, config,
                          pool=pool.get(q["query_id"]) if pool else None,
                          top_n=top_n)
        run[q["query_id"]] = ranked
        for k, v in L.MBR.compute_query_metrics(ranked, labels[q["query_id"]], EVAL_K).items():
            agg.setdefault(k, []).append(v)
    metrics = {k: round(st.mean(v), 4) for k, v in agg.items()}
    metrics["_n_queries"] = len(rows)
    metrics["_seconds"] = round(time.time() - t0, 1)
    return metrics, run, index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LANG_OF))
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--suffix", default="", help="memory-file variant suffix")
    ap.add_argument("--out", default=None, help="where to write the run file")
    ap.add_argument("--use-pool", action="store_true",
                    help="restrict retrieval to each query's candidate pool "
                         "(LeCaRD's official re-ranking protocol); off by default")
    args = ap.parse_args()

    print(f"[{args.dataset}] split={args.split} "
          f"mode={'subpool' if args.use_pool and args.dataset in POOL_PATH else 'full'}")
    metrics, run, _ = evaluate(args.dataset, weak_config(), args.split,
                               args.top_n, suffix=args.suffix,
                               use_pool=args.use_pool)
    print(f"  {metrics['_n_queries']} queries, {metrics['_seconds']}s")
    for m in ("Precision", "Recall", "NDCG"):
        print(f"  {m:10s} " + "  ".join(f"@{k}={metrics[f'{m}@{k}']:.4f}" for k in EVAL_K))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(run, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  run file -> {args.out}")


if __name__ == "__main__":
    main()
