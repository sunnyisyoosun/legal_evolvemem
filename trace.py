"""Trace one query through the four views, and one evolution round through
the three layers.

Written to be read, not to compute anything new: every number it prints comes
from the same code paths `evaluate.py` and `legal_evolve.py` use. The point is
that the two loops are easy to describe and hard to picture -- a retrieval
loop that calls no LLM, inside an evolution loop that calls one per round.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
from legal_meta import LANG_OF
from evolvemem.multi_retriever import RetrievalConfig
from evaluate import CASE_DIR, QRELS, EVAL_K, in_split, weak_config

PRIMARY = "NDCG@30"


def show_query(ds: str, split: str, nth: int, top: int):
    cases = json.load(open(f"{CASE_DIR}/{ds}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{ds}_queries.json", encoding="utf-8"))
    labels = L.MBR.load_qrels(QRELS.format(ds=ds))
    rows = [q for q in qs if q["query_id"] in labels and in_split(q, split)]
    q = rows[nth]
    gold = set(labels[q["query_id"]])
    index = L.build_index(ds, cases, qs)
    title = {d["doc_id"]: (d.get("topic") or "")[:28] for d in cases}

    print(f"\n{'='*74}\nQUERY  {ds} / {split} #{nth}  id={q['query_id']}")
    print(f"{'='*74}")
    print(f"  class     {q.get('legal_class')}  (category={q.get('category')})")
    print(f"  text      {q['text'][:150]}...")
    print(f"  relevant  {len(gold)} docs")

    print(f"\n-- step 1: metadata already extracted for this query "
          f"(no LLM, done at build time)")
    print(f"  parties     {q.get('persons')}")
    print(f"  legal_refs  {(q.get('entities') or [])[:4]}")
    print(f"  location    {q.get('location')!r}")

    print(f"\n-- step 2: each view, scored independently")
    with L.query_context(q, ds):
        a = L.MR.analyze_query(q["text"])
        views = {
            "BM25 (keyword)": index.keyword_search(a["keywords"], top_k=top),
            "semantic (bge-m3)": index.semantic_search(q["text"], top_k=top),
            "metadata (legal_refs)": index.structured_search(
                persons=a["persons"] or None, entities=a["entities"] or None,
                top_k=top),
        }
        strip = list(a.get("incidentals") or a["persons"])
        stripped = q["text"]
        for sp in strip:
            stripped = stripped.replace(sp, "")
        views["party-strip (BM25)"] = index.keyword_search([stripped], top_k=top)

    for name, hits in views.items():
        ids = [index.doc_ids[i] for i, _ in hits]
        n_gold = len(gold & set(ids))
        mark = "".join("O" if d in gold else "." for d in ids[:top])
        print(f"  {name:24s} top{top}: {mark}  ({n_gold} relevant)")
    print(f"  {'party-strip deleted':24s} {strip[:6]}")

    print(f"\n-- step 3: fusion, then score. 0 LLM calls in this whole step.")
    for label, cfg in (("theta_0 (BM25 only)", weak_config()),
                       ("evolved (rrf + swap30)", RetrievalConfig(
                           fusion_mode="rrf", keyword_top_k=100,
                           semantic_top_k=100, structured_top_k=0,
                           max_context=100, weight_keyword=1.0,
                           weight_semantic=0.3, enable_entity_swap=True,
                           entity_swap_keyword_k=30,
                           entity_swap_semantic_k=30))):
        ranked = L.search(index, q, cfg, top_n=30)
        m = L.MBR.compute_query_metrics(ranked, labels[q["query_id"]], EVAL_K)
        hit = [i + 1 for i, d in enumerate(ranked) if d in gold][:5]
        print(f"  {label:24s} {PRIMARY}={m[PRIMARY]:.4f}  "
              f"relevant at ranks {hit or '-'}")


def show_evolution(run_dir: str, ds: str):
    """One run's rounds, as the loop actually recorded them."""
    import glob
    files = sorted(glob.glob(f"{run_dir}/{ds}/round_*.json"),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not files:
        print(f"no rounds under {run_dir}/{ds}")
        return
    print(f"\n{'='*74}\nEVOLUTION  {run_dir}/{ds}\n{'='*74}")
    print(f"  {'r':>2} {'score':>7} {'delta':>8}  {'verdict':8s} "
          f"{'recipe':34s} moved")
    # Elitism compares against the INCUMBENT BEST, not the previous round.
    # Reading the delta against the previous round turns a recovery from a
    # reverted round into an apparent accept: A3 round 2 is +0.2701 on round
    # 1 and still below the incumbent, so the loop rejected it.
    best, prev_cfg = None, None
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        cfg = d.get("config_scored") or {}
        moved = ([k for k in set(cfg) | set(prev_cfg) if cfg.get(k) != prev_cfg.get(k)]
                 if prev_cfg is not None else [])
        delta, verdict = "", "baseline"
        if best is not None:
            dv = d["f1"] - best
            delta = f"{dv:+.4f}"
            verdict = "ACCEPT" if dv > 0.003 else "revert"
        print(f"  {d['round']:>2} {d['f1']:7.4f} {delta:>8}  {verdict:8s} "
              f"{str(d.get('invoked_recipe') or '-'):34s} {sorted(moved)}")
        best = d["f1"] if best is None else max(best, d["f1"])
        prev_cfg = cfg
    print("\n  delta is against the incumbent best, which is what elitism "
          "compares. one LLM call per round (diagnosis);")
    print("  the retrieval inside each round makes none.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="LeCaRDv2", choices=sorted(LANG_OF))
    ap.add_argument("--split", default="val")
    ap.add_argument("--nth", type=int, default=0, help="which query of the split")
    ap.add_argument("--top", type=int, default=10, help="hits shown per view")
    ap.add_argument("--run-dir", default=None,
                    help="also trace this evolution run's rounds")
    ap.add_argument("--only-evolution", action="store_true")
    a = ap.parse_args()
    if not a.only_evolution:
        show_query(a.dataset, a.split, a.nth, a.top)
    if a.run_dir:
        show_evolution(a.run_dir, a.dataset)


if __name__ == "__main__":
    main()
