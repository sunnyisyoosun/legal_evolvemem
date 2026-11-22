"""Score a named grid of configurations over a split.

Three grids, one runner. They were three places before: `sweep.py` (per-class
winners), `fusion_sweep.py` (the seven fusion settings of METHODOLOGY 8), and
a pair of one-off scripts in a session scratchpad that produced the
`weight_semantic` answer key of JOURNAL 28 -- numbers that are quoted all
over the docs while their generator lived in /tmp and would have vanished
with the session.

    python sweep.py --grid classes --dataset LeCaRDv2      # per-class winner
    python sweep.py --grid fusion  --split train           # all 4 datasets
    python sweep.py --grid weight  --split train           # the answer key
    python sweep.py --grid weight-edge --split train       # its boundaries

`classes` answers "is per_category_overrides worth having" -- it reports the
winner per legal class and the oracle gain over one global config, which is
the precondition for the whole per-category approach. It takes one dataset.

`fusion`, `weight` and `weight-edge` answer "what does this axis look like"
and run over every dataset, printing a delta against theta_0 per setting.

A boundary grid exists because a maximum at the edge of a grid is not a
maximum: three of the four datasets peaked at an end of the first
`weight_semantic` grid, so `weight-edge` extends past both ends.
"""
from __future__ import annotations
import argparse, collections, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
from evolvemem.multi_retriever import RetrievalConfig
from evaluate import CASE_DIR, QRELS, load_pool, EVAL_K, in_split

PRIMARY = "NDCG@30"


def grid_classes() -> dict[str, RetrievalConfig]:
    """The best fused setting found so far, with and without the taxonomy filter."""
    base = dict(max_context=100, structured_top_k=0, keyword_top_k=100,
                semantic_top_k=100, weight_keyword=1.0, weight_semantic=0.3,
                fusion_mode="weighted_sum")
    # entity_swap_semantic_k=0 is the design, not a tuning choice: deletion
    # is a lexical act, so party-strip is BM25's twin and has no semantic
    # half. `resolve_config` pins it anyway, so a 30 here never applied --
    # it just read as though the view had two halves.
    swap = dict(enable_entity_swap=True, entity_swap_keyword_k=100,
                entity_swap_semantic_k=0)
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



# ── the dataset-level grids ───────────────────────────────────────────────
# Shared by all of them: max_context=100, keyword_top_k=100,
# weight_keyword=1.0. `weight_structured` is left at the repo default (0.5)
# wherever the metadata view is on -- section 8 never set it, and reading a
# default as a deliberate choice is the mistake that broke two recipes
# (section 9).
# Per-view candidate depth is 100 everywhere. Measured on train, all
# four datasets (sweep.py --grid depth / --grid swap-depth):
#   k=30  loses most of the semantic view's gain (LeCaRDv2 +0.0140
#         -> +0.0042; LeCaRD falls back to theta_0)
#   k=100 is the knee
#   k=200 adds +0.0024 on LeCaRDv2 and ~0 elsewhere, under its floor
# metadata's 10 was the engine default and party-strip's 30 came off
# the old corpus. Depth is not an echo of top_n=30, which the metric
# fixes (NDCG@30); these are how many candidates a VIEW contributes.
_KW = dict(max_context=100, keyword_top_k=100, weight_keyword=1.0)


def grid_fusion() -> dict[str, RetrievalConfig]:
    """Every view subset, spelled out.

    BM25 is on in all of them, so the subsets are the on/off of the other
    three views: 1 + 3 + 3 + 1 = 8. Section 8 ran seven and left out the
    one with all four views on -- party-strip measured dead at the time
    (max +0.0004), so "party-strip plus everything" looked like it could
    not pay. That premise moved: `STRIP_SCOPE` now deletes orgs and places
    as well as party names, so what the view deletes is not what it was.

    This is a THIN SLICE of the real combination space, and the eighth
    setting does not change that. The weights are pinned at one value each
    (semantic 0.3, metadata 0.5, party-strip depth 30) while the optimum
    for `weight_semantic` alone spans 0.1-2.0 across datasets. Read every
    comparison here as true at these weights, not in general.
    """
    # Names carry the VIEW SET, never a depth. The old keys said `str10`
    # and `swap30` and went on saying it after the depths were unified to
    # 100 -- a name that encodes a number it no longer holds is worse than
    # no name. Every view here is at depth 100.
    sem = dict(semantic_top_k=100, weight_semantic=0.3)
    # entity_swap_semantic_k=0 is the design, not a tuning choice: deletion
    # is a lexical act, so party-strip is BM25's twin and has no semantic
    # half. `resolve_config` pins it anyway, so a 30 here never applied --
    # it just read as though the view had two halves.
    swap = dict(enable_entity_swap=True, entity_swap_keyword_k=100,
                entity_swap_semantic_k=0)
    return {
        "1_bm25":        RetrievalConfig(**_KW, fusion_mode="keyword_only",
                                              semantic_top_k=0, structured_top_k=0,
                                              enable_entity_swap=False),
        "2_sem":     RetrievalConfig(**_KW, **sem, fusion_mode="rrf",
                                              structured_top_k=0,
                                              enable_entity_swap=False),
        "3_meta":        RetrievalConfig(**_KW, fusion_mode="rrf",
                                              semantic_top_k=0, structured_top_k=100,
                                              enable_entity_swap=False),
        "4_sem_meta":    RetrievalConfig(**_KW, **sem, fusion_mode="rrf",
                                              structured_top_k=100,
                                              enable_entity_swap=False),
        "5_meta_strip": RetrievalConfig(**_KW, **swap, fusion_mode="rrf",
                                              semantic_top_k=0, structured_top_k=100),
        "6_strip":       RetrievalConfig(**_KW, **swap, fusion_mode="rrf",
                                              semantic_top_k=0, structured_top_k=0),
        "7_ws_sem_strip":  RetrievalConfig(**_KW, **sem, **swap,
                                              fusion_mode="weighted_sum",
                                              structured_top_k=0),
        # The eighth: all four views. See the docstring.
        "8_all_four": RetrievalConfig(**_KW, **sem, **swap,
                                              fusion_mode="rrf",
                                              structured_top_k=100),
    }


def grid_wstruct() -> dict[str, RetrievalConfig]:
    """`weight_structured`: the metadata view's weight, never swept until now.

    Why it became necessary. "The metadata view is harmful on all four
    datasets" was measured with `structured_top_k=10` -- the engine default --
    and with `weight_structured` left at the engine's 0.5. Both were defaults
    read as choices, the mistake that broke two recipes (section 9). At a
    common depth of 100 the picture changed: metadata alone is still negative
    on three datasets, but ALONGSIDE the semantic view it is positive --
    4_sem_meta beats 2_sem on COLIEE (+0.0061 vs +0.0041) and LEGAR (+0.0064
    vs +0.0062), and 8_all_four is first on three of four.

    So the view contributes as a complement, and the weight that governs how
    much has never been measured. The base is the arrangement where it does
    contribute: rrf + semantic + metadata, everything at depth 100. w=0 is
    the control -- it is the same as the view being off, and the deltas are
    read against it.
    """
    base = dict(max_context=100, keyword_top_k=100, weight_keyword=1.0,
                semantic_top_k=100, weight_semantic=0.3, structured_top_k=100,
                fusion_mode="rrf", enable_entity_swap=False)
    return {f"w_str={w}": RetrievalConfig(**base, weight_structured=w)
            for w in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)}


def grid_depth() -> dict[str, RetrievalConfig]:
    """One per-view candidate depth for every view, swept.

    THREE DIFFERENT NUMBERS get confused here, so they are named first:

      top_n          how many ids `search()` RETURNS. The caller sets it;
                     scoring uses 30 because the metric is NDCG@30. Not a
                     dimension -- the metric fixes it.
      max_context    how far down the FUSED ranking is kept before that
                     slice. Also not a dimension: it must simply not bind,
                     which `search()` asserts.
      *_top_k        how many candidates each VIEW contributes. These are
                     the free parameters, and this grid is about them only.

    The per-view depths have no relation to the 30 we return, and an earlier
    version of this docstring wrongly argued from one to the other.

    What is actually wrong with them today is not that they differ -- views
    may well deserve different depths -- but that the four numbers were
    fixed at four unrelated moments. BM25's 100 has a measurement behind it
    (above 100 it is exactly inert, because RRF discounts by 1/(60+rank)).
    metadata's 10 is the engine default. party-strip's 30 came off the old
    corpus, from a table whose `weight_semantic` row has since been
    overturned as a 512-token truncation artefact.

    So: a common k over the two views that measure positive on train, at
    k in {30, 100, 200}. 30 is the shallow point, not an echo of top_n; 200
    is past the known BM25 ceiling, to see whether semantic keeps paying
    where BM25 cannot. `max_context` tracks k only to keep the assert happy.
    `_KW` is not reused because its `keyword_top_k=100` is the number under
    test.
    """
    out = {}
    for k in (30, 100, 200):
        out[f"k={k}"] = RetrievalConfig(
            max_context=max(k, 100), keyword_top_k=k, weight_keyword=1.0,
            semantic_top_k=k, weight_semantic=0.3, fusion_mode="rrf",
            structured_top_k=0, enable_entity_swap=False)
    return out


def grid_swap_depth() -> dict[str, RetrievalConfig]:
    """`entity_swap_keyword_k`: how deep party-strip's second search goes.

    Why this had to be measured. Every "party-strip hurts" number in this
    project was taken at depth 30 while BM25 ran at 100, so it compared a
    shallow view with a deep one rather than comparing the views. RRF sums
    1/(60+rank), so a view that returns more candidates injects more score
    mass; depth is not a detail here.

    The 30 itself was never a choice on this corpus. The engine default is
    10, METHODOLOGY 3 records "30 best (+1.1pp), 100 is +0.4" from the OLD
    corpus -- and the same table's `weight_semantic` claim ("only near 0.3")
    was already overturned as an artefact of 512-token truncation. The code
    says it outright: the swap view's depth is UNMEASURED on this corpus
    (legal_evolve.py, the action-space notes).

    The base is `rrf + semantic`, the best setting measured on train, not
    BM25 alone: party-strip would only ever be switched on from a state the
    loop actually reaches. 0 means the view is off, which is the baseline
    the deltas are read against. 100 is the natural ceiling -- `keyword_top_k`
    is exactly inert above it, for the same RRF discount reason.
    """
    sem = dict(semantic_top_k=100, weight_semantic=0.3)
    out = {"swap_off": RetrievalConfig(**_KW, **sem, fusion_mode="rrf",
                                       structured_top_k=0,
                                       enable_entity_swap=False)}
    for k in (10, 30, 100):
        out[f"swap_k={k}"] = RetrievalConfig(
            **_KW, **sem, fusion_mode="rrf", structured_top_k=0,
            enable_entity_swap=True, entity_swap_keyword_k=k)
    return out


def _weight_grid(values) -> dict[str, RetrievalConfig]:
    """`weight_semantic` alone: rrf + the semantic view, every other view off.

    One axis at a time, because the answer key has to attribute the gain to
    this field and nothing else. The depth is 100, matching the foundation
    recipe, so the optima are valid for the state that card produces -- and
    only for it.
    """
    return {f"w_sem={w}": RetrievalConfig(
        **_KW, fusion_mode="rrf", semantic_top_k=100, weight_semantic=w,
        structured_top_k=0, enable_entity_swap=False) for w in values}


def grid_weight() -> dict[str, RetrievalConfig]:
    return _weight_grid((0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0))


def grid_weight_edge() -> dict[str, RetrievalConfig]:
    """Past both ends of grid_weight. A peak at a grid boundary is not a peak:
    COLIEE peaked at the low end (0.1) and LEGAR_stricter at the high end
    (2.0), so neither had been bracketed."""
    return _weight_grid((0.02, 0.05, 0.25, 3.0, 4.0, 5.0))


GRIDS = {"classes": grid_classes, "fusion": grid_fusion,
         "weight": grid_weight, "weight-edge": grid_weight_edge,
         "swap-depth": grid_swap_depth,
         "depth": grid_depth, "wstruct": grid_wstruct}


def run_datasets(grid, datasets, split, emb_tag, out_path):
    """Score `grid` on each dataset and print the delta against theta_0.

    The index is built and dropped per dataset: four corpora at 8192 tokens
    do not sit in memory together.
    """
    from evaluate import evaluate, weak_config
    out = {}
    for ds in datasets:
        cases = json.load(open(f"{CASE_DIR}/{ds}_cases.json", encoding="utf-8"))
        qs = json.load(open(f"{CASE_DIR}/{ds}_queries.json", encoding="utf-8"))
        index = L.build_index(ds, cases, qs, emb_tag=emb_tag)
        t0 = evaluate(ds, weak_config(), split, 30, index=index,
                      quiet=True)[0][PRIMARY]
        row = {"theta0": round(t0, 4)}
        print(f"[{ds}] split={split}  theta_0 {t0:.4f}", flush=True)
        for name, cfg in grid.items():
            v = evaluate(ds, cfg, split, 30, index=index,
                         quiet=True)[0][PRIMARY]
            row[name] = round(v, 4)
            print(f"  {name:20s} {v:.4f}  {v - t0:+.4f}", flush=True)
        out[ds] = row
        del index
        if out_path:
            json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    if out_path:
        print(f"\n  -> {out_path}")
    return out


def fit_apply(a, qs, labels, pool, index):
    """Per-class settings are only worth it if they survive the split.

    An oracle gain measured on the fitting data is optimistic by construction,
    so the per-class choice is made on train and then scored on test, which is
    what per_category_overrides actually does at inference time.
    """
    cfgs = grid_classes()
    names = list(cfgs)

    def key(q):
        if a.axis == "legal":
            return q["legal_class"]

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


DATASETS = ["LeCaRDv2", "LeCaRD", "COLIEE", "LEGAR_stricter"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="classes", choices=sorted(GRIDS),
                    help="classes: per-class winner on ONE dataset (needs "
                         "--dataset). fusion / weight / weight-edge: a delta "
                         "table per setting over --datasets")
    ap.add_argument("--dataset", help="required for --grid classes")
    ap.add_argument("--datasets", nargs="*", default=DATASETS,
                    help="for the dataset-level grids")
    ap.add_argument("--out", default=None, help="write the table as JSON")
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

    if a.grid != "classes":
        run_datasets(GRIDS[a.grid](), a.datasets, a.split, a.emb_tag, a.out)
        return
    if not a.dataset:
        ap.error("--grid classes needs --dataset")

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

    per: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    overall: dict[str, list[float]] = collections.defaultdict(list)
    for name, cfg in grid_classes().items():
        for q in rows:
            ranked = L.search(index, q, cfg,
                              pool=pool.get(q["query_id"]) if pool else None, top_n=30)
            v = L.MBR.compute_query_metrics(ranked, labels[q["query_id"]], EVAL_K)[PRIMARY]
            per[key(q)][name].append(v)
            overall[name].append(v)
        print(f"  {name:12s} overall {PRIMARY}={st.mean(overall[name]):.4f}", flush=True)

    names = list(grid_classes())
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
