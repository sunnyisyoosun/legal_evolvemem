"""Does the best configuration actually differ by query class?

per_category_overrides is only worth having if different classes prefer
different settings. This sweeps a small set of configurations over a split and
reports the winner per class, which is the precondition for the whole
per-category approach.
"""
from __future__ import annotations
import argparse, collections, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
from evolvemem.multi_retriever import RetrievalConfig
from evaluate import CASE_DIR, QRELS, load_pool, EVAL_K, in_split

PRIMARY = "NDCG@30"


def configs() -> dict[str, RetrievalConfig]:
    """The best fused setting found so far, with and without the taxonomy filter."""
    base = dict(max_context=100, structured_top_k=0, keyword_top_k=100,
                semantic_top_k=100, weight_keyword=1.0, weight_semantic=0.3,
                fusion_mode="weighted_sum")
    swap = dict(enable_entity_swap=True, entity_swap_keyword_k=30,
                entity_swap_semantic_k=30)
    out = {
        "bm25_only": RetrievalConfig(fusion_mode="keyword_only", keyword_top_k=100,
                                     semantic_top_k=0, structured_top_k=0,
                                     max_context=100, enable_entity_swap=False),
        "best_sofar": RetrievalConfig(**base, **swap),
        "best+filter": RetrievalConfig(**base, **swap, enable_chapter_filter=True),
        "bm25+filter": RetrievalConfig(
            fusion_mode="keyword_only", keyword_top_k=100, semantic_top_k=0,
            structured_top_k=0, max_context=100, enable_entity_swap=False,
            enable_chapter_filter=True),
        "sem+filter": RetrievalConfig(**base, enable_chapter_filter=True),
    }
    return out


def fit_apply(a, qs, labels, pool, index):
    """Per-class settings are only worth it if they survive the split.

    An oracle gain measured on the fitting data is optimistic by construction,
    so the per-class choice is made on train and then scored on test, which is
    what per_category_overrides actually does at inference time.
    """
    cfgs = configs()
    names = list(cfgs)

    def key(q):
        if a.axis == "legal":
            return q["legal_class"]
        if a.axis == "difficulty":
            return q["difficulty_class"]
        return f'{q["legal_class"]} x {q["difficulty_class"]}'

    def score(split):
        rows = [q for q in qs if q["query_id"] in labels and in_split(q, split)]
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for name, cfg in cfgs.items():
            for q in rows:
                ranked = L.search(index, q, cfg,
                                  pool=pool.get(q["query_id"]) if pool else None, top_n=30)
                v = L.MBR.compute_query_metrics(ranked, labels[q["query_id"]], EVAL_K)[PRIMARY]
                per[key(q)][name].append(v)
        return per, rows

    tr, tr_rows = score("train")
    te, te_rows = score("test")
    tr_overall = {n: st.mean(v for c in tr for v in tr[c][n]) for n in names}
    gbest = max(names, key=lambda n: tr_overall[n])

    chosen = {}
    for c in tr:
        if len(tr[c][names[0]]) >= a.min_n:
            chosen[c] = max(names, key=lambda n: st.mean(tr[c][n]))
    print(f"  fitted on train: global={gbest}, "
          f"{sum(1 for c in chosen if chosen[c] != gbest)}/{len(chosen)} classes overridden")

    glob, perclass, n_cov = [], [], 0
    for c in te:
        use = chosen.get(c, gbest)
        glob += te[c][gbest]
        perclass += te[c][use]
        n_cov += len(te[c][gbest]) if c in chosen else 0
    print(f"  test n={len(te_rows)} (covered by a fitted class: {n_cov})")
    print(f"  global best        {PRIMARY}={st.mean(glob):.4f}")
    print(f"  per-class override {PRIMARY}={st.mean(perclass):.4f}  "
          f"({st.mean(perclass)-st.mean(glob):+.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--axis", default="legal", choices=("legal",))
    ap.add_argument("--min-n", type=int, default=15, help="skip classes smaller than this")
    ap.add_argument("--emb-tag", default="m3", help="which embedding set to use")
    ap.add_argument("--fit-apply", action="store_true",
                    help="pick the best config per class on train, then score it on test")
    ap.add_argument("--use-pool", action="store_true",
                    help="restrict retrieval to each query's candidate pool "
                         "(LeCaRD's official re-ranking protocol); off by default")
    a = ap.parse_args()

    cases = json.load(open(f"{CASE_DIR}/{a.dataset}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{a.dataset}_queries.json", encoding="utf-8"))
    labels = L.MBR.load_qrels(QRELS.format(ds=a.dataset))
    pool = load_pool(a.dataset) if a.use_pool else None
    index = L.build_index(a.dataset, cases, qs, emb_tag=a.emb_tag)
    rows = [q for q in qs if q["query_id"] in labels and in_split(q, a.split)]
    if a.fit_apply:
        fit_apply(a, qs, labels, pool, index)
        return
    print(f"[{a.dataset}] split={a.split} n={len(rows)}", flush=True)

    def key(q):
        if a.axis == "legal":
            return q["legal_class"]
        if a.axis == "difficulty":
            return q["difficulty_class"]
        return f'{q["legal_class"]} x {q["difficulty_class"]}'

    per: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    overall: dict[str, list[float]] = collections.defaultdict(list)
    for name, cfg in configs().items():
        for q in rows:
            ranked = L.search(index, q, cfg,
                              pool=pool.get(q["query_id"]) if pool else None, top_n=30)
            v = L.MBR.compute_query_metrics(ranked, labels[q["query_id"]], EVAL_K)[PRIMARY]
            per[key(q)][name].append(v)
            overall[name].append(v)
        print(f"  {name:12s} overall {PRIMARY}={st.mean(overall[name]):.4f}", flush=True)

    names = list(configs())
    gbest = max(names, key=lambda n: st.mean(overall[n]))
    print(f"\n  global best: {gbest} ({st.mean(overall[gbest]):.4f})")
    print(f"\n  {'class':34s} {'n':>5s}  {'best':12s} {'best':>8s} {'global':>8s} {'gain':>7s}")
    gain_total, n_total, n_diff = 0.0, 0, 0
    for k in sorted(per, key=lambda k: -len(per[k][names[0]])):
        n = len(per[k][names[0]])
        if n < a.min_n:
            continue
        best = max(names, key=lambda nm: st.mean(per[k][nm]))
        bv, gv = st.mean(per[k][best]), st.mean(per[k][gbest])
        gain_total += (bv - gv) * n
        n_total += n
        n_diff += (best != gbest)
        print(f"  {k:34s} {n:5d}  {best:12s} {bv:8.4f} {gv:8.4f} {bv-gv:+7.4f}")
    if n_total:
        print(f"\n  classes preferring a non-global config: {n_diff}")
        print(f"  oracle per-class gain over global best: {gain_total/n_total:+.4f} {PRIMARY}")


if __name__ == "__main__":
    main()
