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


# The test fold is scored ONCE, at the end. That is a standing instruction,
# and it was broken before this guard existed: METHODOLOGY 8 ran its seven
# settings on test across four datasets, concluded that setting 5 wins, and
# that conclusion then reached the recipes and the diagnosis prompt -- so
# those test numbers are not held-out any more. A sentence in a document did
# not stop it, because `--split test` was just another string.
#
# Now it is a wall: scoring test needs ALLOW_TEST=1 in the environment, set
# deliberately for the one final run. `split=None`/"all" would silently
# include test rows, so they are refused too; pass a fold by name.
def in_split(row: dict, split: str | None) -> bool:
    if split in (None, "all") or split == "test":
        if os.environ.get("ALLOW_TEST") != "1":
            raise RuntimeError(
                f"split={split!r} would score the test fold, which is "
                "measured once at the very end. Use 'train' or 'val'. "
                "For the final run, set ALLOW_TEST=1 explicitly."
            )
    return split in (None, "all") or fold_of(row) == split


def weak_config() -> RetrievalConfig:
    """BM25-only start point, matching the paper's minimal theta_0."""
    # `entity_swap_keyword_k` is set even though the view is off: it is the
    # depth the view GETS when something switches it on, and the engine's
    # default is 10 -- so an `enable_entity_swap: True` proposal that says
    # nothing about depth would silently land on the one value measured to
    # be worst. Every view's depth is 100 (sweep.py --grid depth).
    return RetrievalConfig(
        fusion_mode="keyword_only", keyword_top_k=100, semantic_top_k=0,
        structured_top_k=0, max_context=100, enable_entity_swap=False,
        entity_swap_keyword_k=100,
    )


def per_query(index, rows, labels, config: RetrievalConfig,
              top_n: int = 30, pool=None, metric: str = PRIMARY) -> list[float]:
    """Per-query scores for one config, in `rows` order.

    A list, not a mean: every comparison in this project is PAIRED -- the
    acceptance threshold, recipe admission, the noise floor, the val
    separation check -- and a paired bootstrap needs the per-query
    differences, which a mean has already thrown away.

    This function existed three times verbatim (learn_recipes,
    measure_recipes, select_on_val) and inline twice more (sweep, trace).
    Three copies of a scoring loop is three chances for one of them to use a
    different top_n or a different metric and for the resulting numbers to be
    compared anyway.
    """
    out = []
    for q in rows:
        ranked = L.search(index, q, config, top_n=top_n,
                          pool=pool.get(q["query_id"]) if pool else None)
        out.append(L.MBR.compute_query_metrics(
            ranked, labels[q["query_id"]], EVAL_K)[metric])
    return out


def load_fold(dataset: str, split: str | None = None, *, emb_tag: str = "m3",
              suffix: str = ""):
    """(index, rows, labels) for one dataset+split. One copy, not six.

    These four lines were duplicated in measure_recipes, learn_recipes,
    select_on_val, strip_components and sweep. Six copies of a loader is six
    chances for one of them to read a different corpus file, a different
    qrels path, or -- the one that actually bit -- a different split: the
    `--split` default was `test` in one of them while the documentation said
    train (JOURNAL 46).

    `rows` is already filtered to queries that HAVE labels and fall in the
    split, in corpus order, which is the order `per_query` returns scores in.
    """
    cases = json.load(open(f"{CASE_DIR}/{dataset}_cases{suffix}.json",
                           encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{dataset}_queries{suffix}.json",
                        encoding="utf-8"))
    labels = L.MBR.load_qrels(QRELS.format(ds=dataset))
    index = L.build_index(dataset, cases, qs, emb_tag=emb_tag)
    rows = [q for q in qs if q["query_id"] in labels and in_split(q, split)]
    return index, rows, labels


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
    # A point estimate alone hides what the fold can resolve. LeCaRD ships
    # 107 queries and our split leaves val=8 and test=20, so a LeCaRD number
    # is far less precise than a LEGAR_stricter one computed the same way --
    # and nothing in the output said so. The interval is a bootstrap over
    # QUERIES, which is the only sampling here: retrieval is deterministic,
    # so re-running changes nothing and the uncertainty is entirely "would
    # this hold on other queries".
    #
    # This is the unpaired form -- the CI of the mean score itself, not of a
    # difference. It is therefore WIDER than the paired interval used to
    # compare two configs (recipe_learning.bootstrap_ci on per-query deltas),
    # and the two must not be read against each other: use this one for "how
    # well does this config do", and the paired one for "is B better than A".
    try:
        import recipe_learning as _RL
        primary = f"NDCG@{max(EVAL_K)}" if f"NDCG@{max(EVAL_K)}" in agg else None
        if primary:
            lo, hi = _RL.bootstrap_ci(agg[primary])
            metrics["_ci95"] = [round(lo, 4), round(hi, 4)]
            metrics["_ci95_metric"] = primary
    except Exception as e:                      # never fail a run over a CI
        metrics["_ci95_error"] = str(e)[:120]
    # Cumulative retrieval budget, so a sweep's total cost and a run's total
    # cost are expressed in the same unit (see legal_retriever.budget).
    metrics["_searches"] = L.budget()
    metrics["_seconds"] = round(time.time() - t0, 1)
    return metrics, run, index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LANG_OF))
    # Was `default="test"`: a bare `evaluate.py --dataset X` scored the held
    # -out fold. That is how METHODOLOGY 8's four-dataset table came to be a
    # test-fold measurement. Default train; test needs ALLOW_TEST=1 as well.
    ap.add_argument("--split", default="train",
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
    if "_ci95" in metrics:
        lo, hi = metrics["_ci95"]
        m = metrics["_ci95_metric"]
        print(f"  {m} = {metrics[m]:.4f}  CI95 [{lo:.4f}, {hi:.4f}]  "
              f"(+/-{(hi - lo) / 2 * 100:.2f} pp over {metrics['_n_queries']} "
              f"queries -- differences smaller than that are not resolvable "
              f"on this fold)")
    for m in ("Precision", "Recall", "NDCG"):
        print(f"  {m:10s} " + "  ".join(f"@{k}={metrics[f'{m}@{k}']:.4f}" for k in EVAL_K))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(run, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  run file -> {args.out}")


if __name__ == "__main__":
    main()
