"""Measure each RULE's lift on train, with a paired bootstrap CI.

    python3 measure_recipes.py                       # every dataset with a cache
    python3 measure_recipes.py --datasets LeCaRDv2

Writes rule_lifts.json, which `recipe_learning.apply_measured_lifts` renders
into the diagnosis prompt.

Why this file is a third of what it was
---------------------------------------
It used to measure every card in RECIPES_LEGAL. After the recipe type split
(METHODOLOGY 3) only one of the three types has a lift to measure:

    Bundle     -- carries no values. There is nothing to apply, so there is
                  no delta. Its admission test is a 2x2 interaction, not an
                  effect, and that is a different script.
    Constraint -- declares a forbidden state. Not a move.
    Rule       -- rewrites the query. It HAS an effect on the ranking, and
                  its admission test IS that effect (CI lower bound > 0,
                  mean > the dataset's noise floor, endpoint > theta_0).

The other half of what this file used to do was supply the acceptance
threshold: `noise_floor()` read the median CI half-width out of the table
written here. That was never this file's job -- it made the threshold a
property of the recipe pool -- and it now lives in `noise_floor.py`, which
samples fixed config pairs instead. `recipe_lifts.json` stays on disk as the
read-only fallback for a checkout that has not run the estimator yet.

What measuring a rule means
---------------------------
A rule is text; what it produces is the rewrite cache. So "measure rule R on
dataset D" is: with cache/rewrite/D.json built FROM R, compare
enable_query_rewrite off vs on. The cache is keyed by query id and says
nothing about which prompt made it, so a cache built from one rule and
scored as another would be unattributable -- `query_rewrite.cache_stamp()`
is written beside it and checked here.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L                      # noqa: E402
import recipe_learning as RL                     # noqa: E402
import registry as REG                           # noqa: E402
import query_rewrite as QR                       # noqa: E402
from evaluate import load_fold, per_query, weak_config   # noqa: E402
from legal_meta import LANG_OF                   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "rule_lifts.json")

# A rule is measured from a state someone could stand on, not from theta_0:
# at theta_0 the semantic view is off and fusion does nothing, so a lexical
# rewrite is measured against a configuration no tuned run would hold. Same
# requirement as recipe_learning.REPRESENTATIVE, and the same depths as
# noise_floor.py's fused base so the two are comparable.
BASE = {"fusion_mode": "rrf", "semantic_top_k": 100, "structured_top_k": 100,
        "enable_entity_swap": True, "entity_swap_keyword_k": 100,
        "weight_semantic": 0.3, "weight_structured": 0.1}


def cache_provenance(ds: str) -> tuple[str, str]:
    """(stamp on disk, note). Empty stamp means the cache predates stamping."""
    path = os.path.join(L.REWRITE_DIR, f"{ds}.stamp")
    try:
        with open(path, encoding="utf-8") as f:
            return (f.read().strip(), "")
    except OSError:
        return ("", f"no {ds}.stamp: cache predates stamping, assumed to be "
                    f"the seed rule")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["LeCaRDv2", "LeCaRD", "COLIEE", "LEGAR_stricter"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    # Merge, never overwrite: one dataset's re-measurement must not delete
    # three others' rows.
    try:
        table = json.load(open(a.out, encoding="utf-8"))
    except (OSError, ValueError):
        table = {}

    for ds in a.datasets:
        if not L.load_rewrites(ds):
            print(f"[{ds}] no rewrite cache -- the rule cannot act here, "
                  f"so there is nothing to measure", flush=True)
            continue
        stamp, note = cache_provenance(ds)
        index, rows, labels = load_fold(ds, a.split)
        floor, floor_src = RL.noise_floor(ds)
        t0cfg = weak_config()
        base_cfg = replace(t0cfg, **BASE)
        t0v = per_query(index, rows, labels, t0cfg)
        t0mean = sum(t0v) / len(t0v)
        print(f"[{ds}] {a.split} n={len(rows)}  theta_0 {t0mean:.4f}  "
              f"floor {floor:.5f}", flush=True)
        print(f"  floor source: {floor_src}", flush=True)
        if note:
            print(f"  WARNING {note}", flush=True)

        rows_out, base_scores = {}, None
        for rule in REG.RULES:
            expected = QR.cache_stamp(LANG_OF[ds])
            if stamp and stamp != expected:
                print(f"  {rule.name}: SKIPPED -- cache was built by "
                      f"{stamp!r}, not {expected!r}. Rebuild the cache for "
                      f"this rule before measuring it.", flush=True)
                continue
            t = time.time()
            m = RL.measure_pair(index, rows, labels, base_cfg,
                                {"enable_query_rewrite": True},
                                base_scores=base_scores)
            base_scores = m["base_scores"]          # one base, every rule
            ok_ci = m["ci95"][0] > 0
            ok_size = m["delta"] > floor
            ok_abs = m["after"] > t0mean
            claim = round(m["delta"] * 100, 2) if (ok_ci and ok_size and ok_abs) else 0.0
            why = "" if claim else " [" + ",".join(
                n for n, k in (("ci", ok_ci), ("size", ok_size),
                               ("<t0", ok_abs)) if not k) + "]"
            rows_out[rule.name] = {
                "delta": round(m["delta"], 5),
                "ci95": [round(m["ci95"][0], 5), round(m["ci95"][1], 5)],
                "base": round(m["base"], 4), "after": round(m["after"], 4),
                "theta0": round(t0mean, 4),
                "vs_theta0": round(m["after"] - t0mean, 4),
                "claim_pp": claim, "measured_from": BASE,
                "cache_stamp": stamp or "(unstamped)",
                "zero_reason": why.strip(" []") or "",
            }
            print(f"  {rule.name:24s} {m['delta']:+.4f}  CI95 "
                  f"[{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]  "
                  f"claim {claim:+.2f}pp{why}  ({time.time() - t:.0f}s)",
                  flush=True)

        table[ds] = {"n": len(rows), "split": a.split, "rules": rows_out}
        # Written per dataset: see noise_floor.py for why.
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=1)

    print(f"{a.out}: {', '.join(sorted(table))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
