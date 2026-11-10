"""Measure the hand-written recipes with the same instrument the learned ones face.

Why this exists
---------------
`RECIPES_LEGAL`'s `expected_lift_pp` values were typed in by hand from the
sweep in METHODOLOGY 8 -- and that sweep was run on TEST. The numbers are
test leakage, and they are not inert data: `format_recipes_for_prompt`
renders them into the diagnosis prompt as fact, so every evolution round has
been steered by them. Replacing them is the precondition for re-running any
evolution.

Two things had to change, not one.

1. The fold. The loop's fitness is train (METHODOLOGY 5), so a lift shown to
   the diagnosis LLM has to be a train measurement.
2. The baseline. A card's lift only means something measured from a state
   where its trigger FIRES. The old numbers came from a 7-setting sweep whose
   configurations do not line up with the cards one-to-one: `undo_structured_view`
   is "turn the metadata view off", and no sweep row isolates that.

So each card is measured the way a learned candidate is (recipe_learning):
build a config that satisfies the card's own trigger, score it, apply the
card's proposal, score again, and take a paired bootstrap CI over the
per-query differences. A card whose CI includes zero gets 0.0 -- not because
the card is useless, but because we cannot claim a number for it.

The hand-written triggers are Python lambdas, not the declarative form a
learned candidate uses, so the trigger state cannot be derived from them
automatically. TRIGGER_STATES below spells out, per card, a config the card's
own trigger accepts; each one is asserted against the real trigger before
anything is measured, so a wrong guess fails loudly instead of producing a
number from the wrong state.
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys, time
from dataclasses import asdict, replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
import recipe_learning as RL
from legal_meta import LANG_OF
from evaluate import CASE_DIR, QRELS, EVAL_K, in_split, weak_config
from evolvemem.evolution_cookbook import RECIPES_LEGAL

PRIMARY = "NDCG@30"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "recipe_lifts.json")

# A state each card's trigger accepts, as a diff from theta_0. Asserted below.
TRIGGER_STATES: dict[str, dict] = {
    # fusion off -> the foundation card fires at theta_0 itself
    "lexical_first_light_semantic": {},
    # needs score fusion on and swap off/shallow
    "entity_swap_shallow": {"fusion_mode": "weighted_sum",
                            "semantic_top_k": 100, "weight_semantic": 0.3},
    # needs fusion on, swap off, rewrite off
    "query_rewrite_append_once": {"fusion_mode": "weighted_sum",
                                  "semantic_top_k": 100, "weight_semantic": 0.3},
    # weighted-sum with an overweighted auxiliary view
    "undo_auxiliary_overweight": {"fusion_mode": "weighted_sum",
                                  "semantic_top_k": 100, "weight_semantic": 1.0,
                                  "structured_top_k": 10, "weight_structured": 0.5},
    "undo_structured_view": {"fusion_mode": "weighted_sum",
                             "semantic_top_k": 100, "weight_semantic": 0.3,
                             "structured_top_k": 10, "weight_structured": 0.5},
    # the per-query LLM levers turned on
    "undo_query_decomposition": {"fusion_mode": "weighted_sum",
                                 "semantic_top_k": 100, "weight_semantic": 0.3,
                                 "enable_intent_planning": True},
    "undo_chapter_filter": {"fusion_mode": "weighted_sum",
                            "semantic_top_k": 100, "weight_semantic": 0.3,
                            "enable_chapter_filter": True},
    # stalled, with a deep swap already on -- the history argument is what
    # `_consec_noaccept` reads, so a stalled history is supplied too
    "escape_plateau_shallower_swap": {"fusion_mode": "rrf",
                                      "semantic_top_k": 100, "weight_semantic": 0.3,
                                      "enable_entity_swap": True,
                                      "entity_swap_keyword_k": 30,
                                      "entity_swap_semantic_k": 30},
}
STALLED = [{"accepted": False, "llm_status": "ok", "f1_after": 0.0}] * 3


def per_query(index, rows, labels, cfg) -> list[float]:
    out = []
    for q in rows:
        r = L.search(index, q, cfg, top_n=30)
        out.append(L.MBR.compute_query_metrics(
            r, labels[q["query_id"]], EVAL_K)[PRIMARY])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=sorted(LANG_OF))
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    # Fail before measuring anything if a declared state does not satisfy the
    # card's real trigger: a number from the wrong state is worse than none.
    base0 = asdict(weak_config())
    for r in RECIPES_LEGAL:
        if r.name not in TRIGGER_STATES:
            raise SystemExit(f"no trigger state declared for {r.name}")
        cfg = dict(base0, **TRIGGER_STATES[r.name])
        hist = STALLED if "escape" in r.name else []
        if not r.trigger({}, cfg, hist):
            raise SystemExit(f"declared state does not fire {r.name}: "
                             f"{TRIGGER_STATES[r.name]}")
        changed = [k for k, v in r.proposal.items() if cfg.get(k) != v]
        if not changed:
            raise SystemExit(f"{r.name} is a no-op in its declared state")
    print(f"{len(RECIPES_LEGAL)} cards: trigger states verified\n", flush=True)

    table = {}
    for ds in a.datasets:
        cases = json.load(open(f"{CASE_DIR}/{ds}_cases.json", encoding="utf-8"))
        qs = json.load(open(f"{CASE_DIR}/{ds}_queries.json", encoding="utf-8"))
        labels = L.MBR.load_qrels(QRELS.format(ds=ds))
        rows = [q for q in qs if q["query_id"] in labels and in_split(q, a.split)]
        index = L.build_index(ds, cases, qs)
        print(f"[{ds}] {a.split} n={len(rows)}", flush=True)
        t0cfg = weak_config()
        cache: dict = {}
        table[ds] = {"n": len(rows), "split": a.split, "cards": {}}
        for r in RECIPES_LEGAL:
            t = time.time()
            state = TRIGGER_STATES[r.name]
            key = json.dumps(state, sort_keys=True)
            bcfg = replace(t0cfg, **state)
            if key not in cache:
                cache[key] = per_query(index, rows, labels, bcfg)
            bv = cache[key]
            av = per_query(index, rows, labels, replace(bcfg, **r.proposal))
            d = [y - x for x, y in zip(bv, av)]
            lo, hi = RL.bootstrap_ci(d, alpha=RL.ALPHA)
            mean = st.fmean(d)
            # Shown to the LLM only when the interval clears zero.
            claim = round(mean * 100, 2) if lo > 0 else 0.0
            table[ds]["cards"][r.name] = {
                "delta": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "base": round(st.fmean(bv), 4), "after": round(st.fmean(av), 4),
                "claim_pp": claim, "old_pp": r.expected_lift_pp,
                "measured_from": state,
            }
            print(f"  {r.name:34s} {st.fmean(bv):.4f} -> {st.fmean(av):.4f}  "
                  f"{mean:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
                  f"claim {claim:+.2f}pp (was {r.expected_lift_pp:+.1f})  "
                  f"({time.time()-t:.0f}s)", flush=True)
        del index
        json.dump(table, open(a.out, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
