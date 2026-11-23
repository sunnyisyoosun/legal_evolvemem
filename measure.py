"""Score configurations on a fold and report what separates them.

    python3 measure.py floor                      # acceptance threshold
    python3 measure.py rules  --datasets LeCaRDv2 # a rewriting rule's lift
    python3 measure.py grid   --dataset LeCaRDv2 --grid fusion
    python3 measure.py strip  --dataset LeCaRDv2  # what each strip component costs

Four jobs, one operation: score two (or N) configurations on the same
queries and read the per-query differences. They were four scripts with the
same argparse, the same per-dataset loop and the same merge-write, which is
where `recipe_learning.measure_pair` came from.

What each one answers
---------------------
floor  the smallest delta this fold can tell apart. Fixed config pairs, NOT
       recipe cards -- a card-derived threshold is a property of the pool and
       moved whenever a card was added or removed (METHODOLOGY 5).
rules  a Rule's lift. After the type split only a Rule has one: a Bundle
       carries no values, so there is no delta to take.
grid   a named grid of configurations, the hand-run answer key the loop is
       scored against. NOT an input to the loop -- putting its conclusions
       into a card is the circularity METHODOLOGY 10.1 warns about.
strip  which component of the strip list (persons / orgs / places) costs
       what, measured separately rather than as one switch.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, json, os, statistics as st, sys, time
from dataclasses import replace
import legal_retriever as L                      # noqa: E402
import recipe_learning as RL                     # noqa: E402
import registry as REG                           # noqa: E402
import query_rewrite as QR                       # noqa: E402
from evaluate import (CASE_DIR, EVAL_K, PRIMARY, QRELS, format_strata,  # noqa: E402
                      in_split, load_fold, load_pool, per_query, stratify,
                      weak_config)
from legal_meta import LANG_OF                   # noqa: E402
import argparse, collections, json, os, statistics as st, sys
import legal_retriever as L
from evolvemem.multi_retriever import RetrievalConfig
import argparse, json, os, re, statistics as st, sys
import legal_meta as LM
import recipe_learning as RL


# ======================================================================
# measure.py — Measure something on a fold: the acceptance threshold, or a rule's lift.
# ======================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR_OUT = os.path.join(HERE, "noise_floor.json")
RULE_OUT = os.path.join(HERE, "rule_lifts.json")
BUNDLE_OUT = os.path.join(HERE, "bundle_lifts.json")

# A fused state that is not theta_0. Half the floor pairs start here so the
# estimate is not only "deltas away from the starting point", and the rule
# measurement uses it because at theta_0 the semantic view is off and fusion
# does nothing -- a lexical rewrite measured there is measured against a
# configuration no tuned run would hold. Depths are all 100 (JOURNAL 47).
FUSED = {"fusion_mode": "rrf", "semantic_top_k": 100, "structured_top_k": 100,
         "enable_entity_swap": True, "entity_swap_keyword_k": 100,
         "weight_semantic": 0.3, "weight_structured": 0.1}

# (label, base, change). One field moved to another value a run could
# plausibly hold -- not a boundary, and not a value chosen to be inert. Fixed
# rather than sampled so two runs on the same corpus return the same
# threshold; a threshold that moves on its own is the problem this replaced.
PAIRS: list[tuple[str, str, dict]] = [
    ("sem_on",            "t0",    {"semantic_top_k": 100, "fusion_mode": "rrf",
                                    "weight_semantic": 0.3}),
    ("struct_on",         "t0",    {"structured_top_k": 100, "fusion_mode": "rrf",
                                    "weight_structured": 0.1}),
    ("swap_on",           "t0",    {"enable_entity_swap": True,
                                    "entity_swap_keyword_k": 100,
                                    "fusion_mode": "rrf"}),
    ("kw_depth_100_50",   "t0",    {"keyword_top_k": 50}),
    ("chapter_filter",    "t0",    {"enable_chapter_filter": True}),
    ("fusion_rrf_ws",     "fused", {"fusion_mode": "weighted_sum"}),
    ("w_sem_.3_.6",       "fused", {"weight_semantic": 0.6}),
    ("w_kw_1.0_0.8",      "fused", {"weight_keyword": 0.8}),
    ("w_struct_.1_.5",    "fused", {"weight_structured": 0.5}),
    ("swap_depth_100_30", "fused", {"entity_swap_keyword_k": 30}),
    ("sem_depth_100_30",  "fused", {"semantic_top_k": 30}),
    ("mmr_on",            "fused", {"mmr_diversity_weight": 0.3}),
    ("rewrite_on",        "fused", {"enable_query_rewrite": True}),
]


def _load_table(path: str) -> dict:
    """Merge, never overwrite: one dataset's re-measurement must not delete
    three others' rows. The loss is silent and makes `--acceptance-threshold
    auto` refuse to start for the datasets that vanished."""
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(table: dict, path: str) -> None:
    """Called after EVERY dataset, not once at the end. The first run of the
    floor estimator was killed during the third of four and lost the two that
    had finished -- hours of scoring for nothing. A long job that keeps its
    results only in memory is a job that has to be run twice."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)


def run_floor(a) -> None:
    table = _load_table(a.out)
    for ds in a.datasets:
        index, rows, labels = load_fold(ds, a.split)
        print(f"[{ds}] {a.split} n={len(rows)}", flush=True)
        t0cfg = weak_config()
        bases = {"t0": t0cfg, "fused": replace(t0cfg, **FUSED)}
        scored = {}
        for name, cfg in bases.items():
            t = time.time()
            scored[name] = per_query(index, rows, labels, cfg)
            print(f"  base {name:6s} {st.fmean(scored[name]):.4f} "
                  f"({time.time() - t:.0f}s)", flush=True)

        pairs_out, widths = {}, []
        for label, base, change in PAIRS:
            m = RL.measure_pair(index, rows, labels, bases[base], change,
                                base_scores=scored[base])
            hw = m["halfwidth"]
            # The same delta, split by how many relevant documents the
            # query has. A pair that moves only the |gold|=1 queries is
            # moving RECALL OF ONE; a pair that moves only |gold|>=30 is
            # moving ORDER. Reading one mean over both regimes hides which
            # of the two a lever does (METHODOLOGY 5).
            strata = stratify(rows, labels, m["deltas"])
            pairs_out[label] = {"base": base, "change": change,
                                "by_gold": {k: [n, round(mu, 5)]
                                            for k, (n, mu) in strata.items()},
                                "delta": round(m["delta"], 5),
                                "ci95": [round(m["ci95"][0], 5),
                                         round(m["ci95"][1], 5)],
                                "halfwidth": round(hw, 5)}
            if hw > 1e-9:
                widths.append(hw)
                print(f"  {label:20s} delta {m['delta']:+.4f}  "
                      f"halfwidth {hw:.5f}", flush=True)
            else:
                # Not a measurement of noise: nothing moved. Usually a lever
                # that cannot act on this dataset (enable_query_rewrite has no
                # cache outside LeCaRDv2), which inert_report also names.
                print(f"  {label:20s} inert here (every per-query delta 0)",
                      flush=True)

        floor = round(st.median(widths), 5) if widths else 0.0
        table[ds] = {"n": len(rows), "split": a.split, "floor": floor,
                     "usable_pairs": len(widths), "pairs": pairs_out}
        _save(table, a.out)
        print(f"  -> floor {floor:.5f} (median of {len(widths)} "
              f"non-degenerate half-widths), written\n", flush=True)
    print(f"{a.out}: {', '.join(sorted(table))}")


def _cache_provenance(ds: str) -> tuple[str, str]:
    """(stamp on disk, note). Empty stamp means the cache predates stamping."""
    try:
        with open(os.path.join(L.REWRITE_DIR, f"{ds}.stamp"), encoding="utf-8") as f:
            return (f.read().strip(), "")
    except OSError:
        return ("", f"no {ds}.stamp: cache predates stamping, assumed seed rule")


def run_rules(a) -> None:
    table = _load_table(a.out)
    for ds in a.datasets:
        if not L.load_rewrites(ds):
            print(f"[{ds}] no rewrite cache -- the rule cannot act here, so "
                  f"there is nothing to measure", flush=True)
            continue
        stamp, note = _cache_provenance(ds)
        index, rows, labels = load_fold(ds, a.split)
        floor, floor_src = RL.noise_floor(ds)
        t0cfg = weak_config()
        base_cfg = replace(t0cfg, **FUSED)
        t0v = per_query(index, rows, labels, t0cfg)
        t0mean = st.fmean(t0v)
        print(f"[{ds}] {a.split} n={len(rows)}  theta_0 {t0mean:.4f}  "
              f"floor {floor:.5f}\n  floor source: {floor_src}", flush=True)
        if note:
            print(f"  WARNING {note}", flush=True)

        rows_out, base_scores = {}, None
        for rule in REG.RULES:
            expected = QR.cache_stamp(LANG_OF[ds])
            if stamp and stamp != expected:
                print(f"  {rule.name}: SKIPPED -- cache was built by {stamp!r}, "
                      f"not {expected!r}", flush=True)
                continue
            t = time.time()
            m = RL.measure_pair(index, rows, labels, base_cfg,
                                {"enable_query_rewrite": True},
                                base_scores=base_scores)
            base_scores = m["base_scores"]
            ok = (("ci", m["ci95"][0] > 0), ("size", m["delta"] > floor),
                  ("<t0", m["after"] > t0mean))
            claim = round(m["delta"] * 100, 2) if all(v for _, v in ok) else 0.0
            why = "" if claim else " [" + ",".join(n for n, v in ok if not v) + "]"
            rows_out[rule.name] = {
                "delta": round(m["delta"], 5),
                "ci95": [round(m["ci95"][0], 5), round(m["ci95"][1], 5)],
                "base": round(m["base"], 4), "after": round(m["after"], 4),
                "theta0": round(t0mean, 4),
                "vs_theta0": round(m["after"] - t0mean, 4),
                "claim_pp": claim, "measured_from": FUSED,
                "cache_stamp": stamp or "(unstamped)",
                "zero_reason": why.strip(" []"),
            }
            print(f"  {rule.name:24s} {m['delta']:+.4f}  CI95 "
                  f"[{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]  "
                  f"claim {claim:+.2f}pp{why}  ({time.time() - t:.0f}s)",
                  flush=True)
        table[ds] = {"n": len(rows), "split": a.split, "rules": rows_out}
        _save(table, a.out)
    print(f"{a.out}: {', '.join(sorted(table))}")


# A representative move per field, for measuring bundles ONLY.
#
# A Bundle declares which fields must move together and deliberately carries
# NO values -- that is the whole point of the type (METHODOLOGY 3). But an
# interaction cannot be measured without moving something, so the
# measurement has to choose values. They are declared here, in the measuring
# script, and NOT in the bundle: a value that lives in the card is a value
# the card hands the loop, which is the circularity JOURNAL 33 found.
#
# Consequence to keep in sight: the verdict is CONDITIONAL on these values.
# A pair of fields can be additive at one operating point and interacting at
# another, so every row records the move it used.
BUNDLE_MOVES: dict[str, object] = {
    "fusion_mode": "rrf",
    "semantic_top_k": 100,
    "structured_top_k": 100,
    "keyword_top_k": 50,
    "enable_entity_swap": True,
    "entity_swap_keyword_k": 100,
    "weight_semantic": 0.3,
    "weight_structured": 0.1,
    "weight_keyword": 0.8,
    "enable_query_rewrite": True,
    "enable_chapter_filter": True,
    "mmr_diversity_weight": 0.3,
}


def run_bundles(a) -> None:
    """Is each Bundle an EDGE, or just two fields someone grouped?

    For fields A and B the loop scores four configurations on one fold --
    base, +A, +B, +AB -- and asks whether the whole differs from the sum of
    the parts. `recipe_learning.admit_bundle` does the test; this produces
    the three per-query delta lists it needs.

    A bundle of more than two fields is measured as 2x2 over a SPLIT of its
    fields (first vs rest), because the question the bundle answers is "do
    these have to move together", not "which subset interacts" -- the full
    factorial would be 2^k scorings for an answer nobody asked for.

    The base is a state where the bundle's trigger is true: measuring an
    edge from a state the bundle would never fire in says nothing about the
    edge.
    """
    import registry as REG
    table = _load_table(a.out)
    for ds in a.datasets:
        index, rows, labels = load_fold(ds, a.split)
        t0cfg = weak_config()
        print(f"[{ds}] {a.split} n={len(rows)}", flush=True)
        out = {}
        for b in REG.BUNDLES:
            fields = sorted(b.fields)
            if not b.trigger({f: getattr(t0cfg, f, None)
                              for f in RL._FIELD_TYPE}):
                print(f"  {b.name}: trigger false at theta_0 -- skipped "
                      f"(measure it from a state it fires in)", flush=True)
                continue
            missing = [f for f in fields if f not in BUNDLE_MOVES]
            if missing:
                print(f"  {b.name}: no declared move for {missing} -- "
                      f"add it to BUNDLE_MOVES", flush=True)
                continue
            a_fields, b_fields = fields[:1], fields[1:]
            mv = lambda fs: {f: BUNDLE_MOVES[f] for f in fs}
            t = time.time()
            m_a = RL.measure_pair(index, rows, labels, t0cfg, mv(a_fields))
            m_b = RL.measure_pair(index, rows, labels, t0cfg, mv(b_fields),
                                  base_scores=m_a["base_scores"])
            m_ab = RL.measure_pair(index, rows, labels, t0cfg, mv(fields),
                                   base_scores=m_a["base_scores"])
            ok, why = RL.admit_bundle(m_a["deltas"], m_b["deltas"],
                                      m_ab["deltas"])
            out[b.name] = {
                "fields": fields, "A": a_fields, "B": b_fields,
                "moves": mv(fields),
                "delta_a": round(m_a["delta"], 5),
                "delta_b": round(m_b["delta"], 5),
                "delta_ab": round(m_ab["delta"], 5),
                "interaction": round(
                    m_ab["delta"] - (m_a["delta"] + m_b["delta"]), 5),
                "admitted": ok, "why": why,
            }
            print(f"  {b.name:28s} {'EDGE' if ok else 'additive'}  {why}  "
                  f"({time.time() - t:.0f}s)", flush=True)
        table[ds] = {"n": len(rows), "split": a.split, "bundles": out}
        _save(table, a.out)
    print(f"{a.out}: {', '.join(sorted(table))}")


def _cli_measure() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="what", required=True)
    for name, out, fn in (("floor", FLOOR_OUT, run_floor),
                          ("rules", RULE_OUT, run_rules),
                          ("bundles", BUNDLE_OUT, run_bundles)):
        p = sub.add_parser(name)
        p.add_argument("--datasets", nargs="+",
                       default=["LeCaRDv2", "LeCaRD", "COLIEE", "LEGAR_stricter"])
        p.add_argument("--split", default="train",
                       help="the fold the loop optimises on")
        p.add_argument("--out", default=out)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)
    return 0


# ======================================================================
# measure.py grid — Score a named grid of configurations over a split.
# ======================================================================

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
# four datasets (measure.py grid --grid depth / --grid swap-depth):
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


def _cli_sweep():
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


# ======================================================================
# measure.py strip — Which part of the strip list costs the -0.0170 on LeCaRDv2?
# ======================================================================

def strip(text: str, spans) -> str:
    out = text
    for sp in sorted(spans, key=len, reverse=True):
        out = out.replace(sp, "")
    return re.sub(r"\s+", " ", out).strip(" ?.,")


def components(row, text, lang):
    """The four sources, kept apart instead of unioned.

    org/place come from `legal_meta.extract_incidentals(by_kind=True)` -- the
    real code path. An earlier version of this script re-ran the regexes
    itself and the copy kept a first-character deictic test after the module
    had moved to a whole-span one, so a fix to the extractor did not reach
    the measurement at all (JOURNAL 41).
    """
    parties = [p for p in L.parties_of(row) if L.is_party_name(p, lang)]
    kinds = LM.extract_incidentals(text, lang, by_kind=True)
    loc = str(row.get("location") or "").strip()
    court = [loc] if len(loc) >= 2 else []
    keep = lambda xs: [x.strip() for x in xs if len(x.strip()) >= 2]
    return {"parties": keep(parties), "org": keep(kinds["org"]),
            "place": keep(kinds["place"]), "court": court}


def _cli_strip_components():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+",
                    help="one or more datasets. Absorbed strip_primary.py on "
                         "2026-11-12: its two variants ('parties only',"
                         " 'parties+org+place') are the first and last rows "
                         "of the component table below, measured from the "
                         "same theta_0, so it was a strict subset")
    ap.add_argument("--spans", type=int, default=12,
                    help="how many worst-hurt queries to print spans for")
    a = ap.parse_args()
    for ds in a.datasets:
        _one(ds, a.spans)


def _one(ds: str, n_spans: int) -> None:
    lang = LANG_OF[ds]

    index, rows, labels = load_fold(ds, "train")
    cfg = weak_config()
    comp = {q["query_id"]: components(q, q["text"], lang) for q in rows}

    def run(pick):
        """pick: which component names to delete."""
        out, used = [], 0
        for q in rows:
            sp = [s for k in pick for s in comp[q["query_id"]][k]]
            used += bool(sp)
            r = {**q, "text": strip(q["text"], sp)} if sp else q
            ranked = L.search(index, r, cfg, top_n=30)
            out.append(L.MBR.compute_query_metrics(
                ranked, labels[q["query_id"]], EVAL_K)[PRIMARY])
        return out, used

    base, _ = run([])
    thr, _ = RL.noise_floor(ds)
    print(f"\n[{ds}] train n={len(rows)}  theta_0 = {st.fmean(base):.4f}  "
          f"noise floor {thr:.4f}")
    print(f"  {'삭제 대상':28s} {'점수':>7s} {'Δ':>9s}  {'CI95':>22s}  적용률")
    results = {}
    for name, pick in [
        ("parties", ["parties"]),
        ("parties+court", ["parties", "court"]),
        ("parties+org", ["parties", "org"]),
        ("parties+place", ["parties", "place"]),
        ("parties+org+place+court", ["parties", "org", "place", "court"]),
    ]:
        v, used = run(pick)
        d = [y - x for x, y in zip(base, v)]
        lo, hi = RL.bootstrap_ci(d)
        results[name] = (v, d)
        mark = "악화" if hi < 0 else ("개선" if lo > 0 else "구분 불가")
        print(f"  {name:28s} {st.fmean(v):7.4f} {st.fmean(d):+9.4f}  "
              f"[{lo:+.4f},{hi:+.4f}]  {used}/{len(rows)}  {mark}")

    if n_spans:
        # the place-only variant is the one under suspicion
        _, d = results["parties+place"]
        order = sorted(range(len(rows)), key=lambda i: d[i])[:n_spans]
        print(f"\n  가장 많이 떨어진 {n_spans}건 — place로 지워진 span")
        for i in order:
            q = rows[i]
            pl = comp[q["query_id"]]["place"]
            if not pl:
                continue
            print(f"   Δ={d[i]:+.4f}  place={pl}")
            before = q["text"][:90].replace("\n", " ")
            after = strip(q["text"], comp[q["query_id"]]["parties"] + pl)[:90]
            print(f"      전: {before}")
            print(f"      후: {after}")
    del index


# ── one entry point ───────────────────────────────────────────────────────
STAGES = {"floor": None, "rules": None, "bundles": None,
          "grid": _cli_sweep, "strip": _cli_strip_components}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__)
        return 2
    stage, rest = sys.argv[1], sys.argv[2:]
    if stage in ("floor", "rules", "bundles"):
        sys.argv = [f"measure.py {stage}", stage] + rest
        return _cli_measure()
    sys.argv = [f"measure.py {stage}"] + rest
    STAGES[stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())