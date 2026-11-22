"""Estimate each dataset's noise floor from CONFIG PAIRS, not from recipes.

    python3 noise_floor.py                      # four datasets, train
    python3 noise_floor.py --datasets LeCaRD

Why this exists
---------------
The acceptance threshold has to clear the uncertainty of the fold, or the
loop accepts rounds that cannot be told apart from doing nothing and elitism
carries that state forward as the incumbent best (METHODOLOGY 5).

Until 2026-11-22 that number came out of `recipe_lifts.json`: the median
CI95 half-width over the cards `measure_recipes.py` happened to measure. Two
things are wrong with that.

1. It is a property of the POOL, not of the dataset. Measured cards drift in
   and out -- of the eight that produced the live thresholds, three
   (escape_plateau_shallower_swap, undo_auxiliary_overweight,
   undo_query_decomposition) no longer exist and three more were removed on
   2026-11-22 -- so the threshold moved whenever the pool did. LeCaRD's
   0.0228 is the median of two cards, one of which is the misread
   undo_chapter_filter (JOURNAL 48).
2. Cards are a BIASED sample of deltas. A card is a move someone thought
   would help; they cluster on a few fields. The spread of the deltas our
   cards produce is not the noise of the metric on this fold.

And it was load-bearing by accident: with recipes splitting into Bundles
(which carry no values and so have no lift at all) and Rules, a card-derived
floor would go to 0.0 on three of four datasets -- silently admitting
everything.

What this measures instead
--------------------------
Retrieval here is deterministic: the same (query, config) always scores the
same. The uncertainty is that the fold is ONE SAMPLE OF QUERIES, so a delta
measured on it may not hold on another. For a set of config pairs someone
could actually stand on, this scores both sides on the same queries, takes
the paired bootstrap CI95 of the per-query differences, and reports the
MEDIAN half-width. Zero-width intervals are dropped: a pair whose every
per-query difference is exactly 0 measures the noise of nothing.

The estimate improves with more pairs, where the old one degraded as the
pool shrank.
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys, time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L                      # noqa: E402
import recipe_learning as RL                     # noqa: E402
from evaluate import load_fold, per_query, weak_config  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "noise_floor.json")

# A fused state that is not theta_0: half the pairs are measured from here so
# the estimate is not only "deltas away from the starting point". Depths are
# all 100 (JOURNAL 47) and the weight is the one value measured to work on
# more than one dataset -- this is a place someone could stand, which is the
# same requirement recipe_learning.REPRESENTATIVE exists for.
FUSED = {"fusion_mode": "rrf", "semantic_top_k": 100, "structured_top_k": 100,
         "enable_entity_swap": True, "entity_swap_keyword_k": 100,
         "weight_semantic": 0.3, "weight_structured": 0.1}

# (label, base, change). Each is one field moved to another value a run could
# plausibly hold -- not a boundary, and not a value chosen to be inert. The
# list is fixed rather than sampled so two runs of this script on the same
# corpus return the same threshold; a threshold that moves on its own is the
# problem this file was written to end.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["LeCaRDv2", "LeCaRD", "COLIEE", "LEGAR_stricter"])
    ap.add_argument("--split", default="train",
                    help="the fold the loop optimises on; the floor has to "
                         "describe THAT fold's sampling noise")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    # Merge, never overwrite: one dataset's re-measurement must not delete
    # three others' rows. measure_recipes.py learned this the hard way --
    # the loss is silent and makes `--acceptance-threshold auto` refuse to
    # start for the datasets that vanished.
    try:
        table = json.load(open(a.out, encoding="utf-8"))
    except (OSError, ValueError):
        table = {}

    for ds in a.datasets:
        index, rows, labels = load_fold(ds, a.split)
        print(f"[{ds}] {a.split} n={len(rows)}", flush=True)
        t0cfg = weak_config()
        bases = {"t0": t0cfg, "fused": replace(t0cfg, **FUSED)}
        scored: dict[str, list[float]] = {}
        for name, cfg in bases.items():
            t = time.time()
            scored[name] = per_query(index, rows, labels, cfg)
            print(f"  base {name:6s} {st.fmean(scored[name]):.4f} "
                  f"({time.time() - t:.0f}s)", flush=True)

        pairs_out, widths = {}, []
        for label, base, change in PAIRS:
            bv = scored[base]
            av = per_query(index, rows, labels, replace(bases[base], **change))
            d = [y - x for x, y in zip(bv, av)]
            lo, hi = RL.bootstrap_ci(d, alpha=RL.ALPHA)
            hw = (hi - lo) / 2.0
            pairs_out[label] = {"base": base, "change": change,
                                "delta": round(st.fmean(d), 5),
                                "ci95": [round(lo, 5), round(hi, 5)],
                                "halfwidth": round(hw, 5)}
            if hw > 1e-9:
                widths.append(hw)
            else:
                # Not a measurement of noise: nothing moved. Usually a lever
                # that cannot act on this dataset (enable_query_rewrite has
                # no cache outside LeCaRDv2), which inert_report already
                # names during a run.
                print(f"  {label:20s} inert here (every per-query delta 0)",
                      flush=True)
                continue
            print(f"  {label:20s} delta {st.fmean(d):+.4f}  "
                  f"halfwidth {hw:.5f}", flush=True)

        floor = round(st.median(widths), 5) if widths else 0.0
        table[ds] = {"n": len(rows), "split": a.split,
                     "floor": floor, "usable_pairs": len(widths),
                     "pairs": pairs_out}
        # Write after EVERY dataset, not once at the end. The first run of
        # this script was killed during the third of four and lost the two
        # that had already finished -- hours of scoring for nothing. A long
        # job that keeps its results only in memory is a job that has to be
        # run twice.
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=1)
        print(f"  -> floor {floor:.5f} (median of {len(widths)} "
              f"non-degenerate half-widths), written to {a.out}\n", flush=True)

    print(f"{a.out}: {', '.join(sorted(table))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
