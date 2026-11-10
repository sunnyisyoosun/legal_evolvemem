"""Propose recipes with the LLM, admit them by measurement on train-dev.

Run this BEFORE an evolution run, not inside one. Keeping the two separate
means an evolution result can be attributed to the pool it had, and a pool
can be compared against another pool on the same loop. Folding recipe
learning into the loop would make a round's outcome depend on a pool that
changed during the same run.

    python3 learn_recipes.py --dataset LeCaRDv2 --proposals 6 \
        --base-url https://openrouter.ai/api/v1 --model qwen/...

Each proposal costs one LLM call plus two scoring passes over train-dev
(base and candidate). Nothing else calls the LLM.
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys, time
from dataclasses import asdict, replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
import recipe_learning as RL
from legal_meta import LANG_OF
from evaluate import CASE_DIR, QRELS, EVAL_K, in_split, weak_config
from legal_evolve import make_llm_call

PRIMARY = "NDCG@30"


def per_query(index, rows, labels, cfg) -> list[float]:
    """Per-query scores, kept as a list because admission needs the paired
    differences, not just their mean (recipe_learning.admit)."""
    out = []
    for q in rows:
        ranked = L.search(index, q, cfg, top_n=30)
        out.append(L.MBR.compute_query_metrics(
            ranked, labels[q["query_id"]], EVAL_K)[PRIMARY])
    return out


def score(index, rows, labels, cfg) -> float:
    v = per_query(index, rows, labels, cfg)
    return st.mean(v) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="LeCaRDv2", choices=sorted(LANG_OF))
    ap.add_argument("--proposals", type=int, default=5)
    ap.add_argument("--base-url", default=os.environ.get(
        "OPENAI_API_BASE", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("EVOLVE_MODEL", ""))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="proposal sampling temperature. 0.3 collapsed onto "
                         "one idea (~90%% near-duplicates); raise it to widen "
                         "the sample")
    ap.add_argument("--rotate-crowded", type=int, default=0,
                    help="withhold this many of the most-proposed fields each "
                         "round, computed from the store's own counts")
    ap.add_argument("--from-config", default=None,
                    help="JSON config to propose against; default theta_0")
    ap.add_argument("--no-cot", action="store_true",
                    help="drop the six reasoning steps from the prompt; "
                         "everything else (fields, limits, coverage counts, "
                         "rejected list) is identical, so the arms differ "
                         "only in the reasoning instruction")
    ap.add_argument("--store", default=None,
                    help="store file to use. Two prompt arms MUST use "
                         "separate stores: the dedup check compares against "
                         "whatever the store holds, so a shared file gives "
                         "the later arm a bigger collision set")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt and exit, no LLM call")
    a = ap.parse_args()

    if a.store:
        RL.set_store(a.store)
    cases = json.load(open(f"{CASE_DIR}/{a.dataset}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{a.dataset}_queries.json", encoding="utf-8"))
    labels = L.MBR.load_qrels(QRELS.format(ds=a.dataset))
    train = [q for q in qs if q["query_id"] in labels and in_split(q, "train")]
    fit, dev = RL.split_train(train)

    base_cfg = weak_config()
    if a.from_config:
        base_cfg = replace(base_cfg, **{
            k: v for k, v in json.load(open(a.from_config)).items()
            if k in RL.ALLOWED_FIELDS})

    store = RL.load_store()
    print(f"[{a.dataset}] train={len(train)}  fit={len(fit)}  "
          f"dev={len(dev)}  (dev is {RL.DEV_FRACTION:.0%}, hashed by query id)")
    print(f"store: {len(store['admitted'])} admitted, "
          f"{len(store['rejected'])} rejected  "
          f"[{'CoT' if not a.no_cot else 'no-CoT'}] {RL.STORE}")

    if a.dry_run:
        print("\n" + RL.build_prompt(asdict(base_cfg), store,
                                      cot=not a.no_cot))
        return

    index = L.build_index(a.dataset, cases, qs)
    theta0 = score(index, dev, labels, base_cfg)
    print(f"theta_0 {PRIMARY} on train-dev = {theta0:.4f}")
    print("each candidate is measured from a config that SATISFIES its own "
          "trigger, not from theta_0")
    print(f"temperature={a.temperature}"
          + (f", withholding {a.rotate_crowded} crowded field(s)/round"
             if a.rotate_crowded else ""))
    print(f"admission: paired-bootstrap CI lower bound > 0 at "
          f"alpha={RL.ALPHA}/k, mean > +{RL.ADMIT_THRESHOLD}, and "
          f"after > theta_0\n")
    _base_cache: dict = {}
    # Every candidate is tested against the same dev fold, so the confidence
    # level tightens as the count grows (recipe_learning.admit).
    n_tested = 0

    llm = make_llm_call(a.base_url, a.model, api_key=a.api_key,
                        seed=None if a.seed < 0 else a.seed)

    for i in range(a.proposals):
        banned = (RL.crowded_fields(store, k=a.rotate_crowded)
                  if a.rotate_crowded else [])
        prompt = RL.build_prompt(asdict(base_cfg), store, cot=not a.no_cot,
                                 banned=banned)
        t0 = time.time()
        try:
            # Room for the five reasoning steps plus the object. At 1024 the
            # JSON was getting cut off mid-object once reasoning came first.
            raw = llm([{"role": "user", "content": prompt}],
                      max_tokens=2048, temperature=a.temperature)
        except Exception as e:                       # noqa: BLE001
            print(f"[{i+1}/{a.proposals}] LLM call failed: "
                  f"{type(e).__name__}: {e}")
            continue

        reasoning, _ = RL.split_reasoning(raw)
        try:
            cand = RL.validate(RL.parse_response(raw),
                               store["admitted"] + store["rejected"],
                               banned=banned)
        except ValueError as e:
            # A malformed candidate is recorded too: without it the same
            # unusable shape comes back next round and nothing says why.
            print(f"[{i+1}/{a.proposals}] invalid: {e}")
            store.setdefault("rejected", []).append({
                "name": f"invalid_{int(time.time())}", "proposal": {},
                "trigger": [], "note": f"invalid: {e}",
                "reasoning": reasoning[:4000], "model": a.model,
                "raw": raw[:1200]})
            RL.save_store(store)
            continue

        # The delta has to come from the state the trigger describes.
        tb = RL.trigger_base(asdict(base_cfg), cand["trigger"])
        if tb is None:
            print(f"[{i+1}/{a.proposals}] {cand['name']}: unmeasurable -- "
                  f"no config satisfies its trigger")
            RL.record(store, cand, {}, admitted=False,
                      note="unmeasurable: trigger unsatisfiable",
                      reasoning=reasoning, model=a.model)
            RL.save_store(store)
            continue
        state = {k: v for k, v in tb.items()
                 if asdict(base_cfg).get(k) != v}
        key = json.dumps(state, sort_keys=True)
        base_cfg_t = replace(base_cfg, **{k: v for k, v in tb.items()
                                          if k in RL.ALLOWED_FIELDS})
        if key not in _base_cache:
            _base_cache[key] = per_query(index, dev, labels, base_cfg_t)
        base_v = _base_cache[key]
        base = st.mean(base_v)

        cfg = replace(base_cfg_t, **cand["proposal"])
        after_v = per_query(index, dev, labels, cfg)
        after = st.mean(after_v)
        deltas = [b - a_ for a_, b in zip(base_v, after_v)]
        delta = after - base
        n_tested += 1
        ok, why = RL.admit(deltas, after, theta0, n_tested=n_tested)
        full = RL.crowding(store)
        if ok and full:
            ok, why = False, full
        pred = RL.score_prediction(cand.get("predicted") or {}, delta)
        RL.record(store, cand,
                  RL.measured_block(delta, base, after, len(dev), a.dataset,
                                    trigger_state=state,
                                    ci=RL.bootstrap_ci(
                                        deltas, alpha=RL.ALPHA / n_tested),
                                    k_tested=n_tested),
                  admitted=ok, note=why, reasoning=reasoning,
                  model=a.model, prediction_score=pred)
        RL.save_store(store)
        print(f"[{i+1}/{a.proposals}] {cand['name']:34s} "
              f"{base:.4f} -> {after:.4f}  {delta:+.4f}  "
              f"{'ADMIT' if ok else 'reject'}   ({time.time()-t0:.0f}s)")
        print(f"      {why}")
        first = next((ln for ln in reasoning.splitlines() if ln.strip()), "")
        print(f"      reasoning {len(reasoning)}자: {first[:96]}")
        if cand.get("predicted"):
            hits = ", ".join(f"{k}={v}" for k, v in pred.items())
            print(f"      predicted {cand['predicted']}  ->  {hits}")
        print(f"      base     {state or 'theta_0'}"
              + ("" if after > theta0 else
                 f"   [after {after:.4f} <= theta_0 {theta0:.4f}]"))
        print(f"      proposal {json.dumps(cand['proposal'], ensure_ascii=False)}")
        print(f"      trigger  {json.dumps(cand['trigger'], ensure_ascii=False)}")

    store = RL.load_store()
    print(f"\nstore -> {RL.STORE}")
    print(f"  admitted {len(store['admitted'])}: "
          f"{[e['name'] for e in store['admitted']]}")
    print(f"  rejected {len(store['rejected'])}")


if __name__ == "__main__":
    main()
