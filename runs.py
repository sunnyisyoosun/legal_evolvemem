"""Read what an evolution run produced.

    python3 runs.py trace  --dataset LeCaRDv2 --qid 1       # one query through four views
    python3 runs.py select --dataset LeCaRDv2 --dir evolution_official --write
    python3 runs.py percat --dataset LeCaRDv2               # assemble per-class configs

Three scripts became one because all three read `evolution_*/<ds>/round_N.json`
and nothing else writes them. They also carried two implementations of the
same four lines -- dict to RetrievalConfig, unknown keys dropped -- which is
now one function.

trace   why a query scored what it did: which view surfaced which gold
        document, at which rank, under the round's config.
select  the loop maximises TRAIN; this re-scores every round's saved config
        on VAL and picks by the rule in METHODOLOGY 5 (beat theta_0 on both,
        then highest val). No re-search: the configs are on disk.
percat  per-class configs assembled into one `per_category_overrides` map.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, json, os, sys
import legal_retriever as L
from legal_meta import LANG_OF
from evolvemem.multi_retriever import RetrievalConfig
from evaluate import CASE_DIR, QRELS, EVAL_K, in_split, weak_config
import argparse, glob, json, os, sys
from evaluate import EVAL_K, evaluate, load_fold, weak_config
import argparse, json, os
from evaluate import CASE_DIR, evaluate, weak_config
from dataclasses import asdict


# ======================================================================
# runs.py trace — Trace one query through the four views, and one evolution round through
# ======================================================================

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
                           entity_swap_keyword_k=100,
                           entity_swap_semantic_k=0))):
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


def _cli_trace():
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


# ======================================================================
# runs.py select — Re-score every evolution round's config on the validation fold.
# ======================================================================

FIELDS = set(RetrievalConfig.__dataclass_fields__)


def as_config(d: dict) -> RetrievalConfig:
    return RetrievalConfig(**{k: v for k, v in d.items() if k in FIELDS})


def round_configs(run_dir: str) -> list[tuple[int, dict]]:
    """(round, config) pairs. round_N.json holds the config proposed FOR the
    next round, so round N was evaluated with round N-1's file; round 0 ran
    theta_0."""
    out = []
    for p in sorted(glob.glob(f"{run_dir}/round_*.json"),
                    key=lambda x: int(x.split("_")[-1].split(".")[0])):
        n = int(p.split("_")[-1].split(".")[0])
        out.append((n, json.load(open(p, encoding="utf-8"))["config"]))
    return out


def _cli_select_on_val():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dir", default="evolution_global")
    ap.add_argument("--legal-class", default=None,
                    help="score only this Axis A class (per-class runs)")
    ap.add_argument("--write", action="store_true",
                    help="write the winner to best_config_val.json")
    a = ap.parse_args()

    run_dir = f"{a.dir}/{a.dataset}"
    # `None` keeps every labelled query: this script re-scores saved run
    # files per fold, so it needs the whole set, not one split.
    index, _rows, labels = load_fold(a.dataset, None)

    def scores_from_run(run: dict) -> dict:
        """Per-query NDCG@30 from a run file -- no re-searching.

        Not `evaluate.per_query`, which re-searches: this reads the ranking
        `evaluate` already returned. Same metric, different input, so it
        keeps its own name.

        `evaluate` already returns the ranked list per query, and the metric
        is a pure function of (ranking, qrels), so the per-query vector the
        paired bootstrap needs costs nothing beyond the pass already made.
        """
        return {qid: L.MBR.compute_query_metrics(r, labels[qid], EVAL_K)["NDCG@30"]
                for qid, r in run.items() if qid in labels}

    def score(cfg):
        out = []
        for sp in ("train", "val"):
            m, run, _ = evaluate(a.dataset, cfg, sp, 30, index=index, quiet=True,
                                 legal_class=a.legal_class)
            out.append((m["NDCG@30"], scores_from_run(run)))
        return out[0][0], out[1][0], out[0][1], out[1][1]

    rows = [(-1, "theta_0", *score(weak_config()))]
    seen = set()
    for n, cfg in round_configs(run_dir):
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key in seen:                      # a rejected round reverts; skip duplicates
            continue
        seen.add(key)
        rows.append((n, f"round_{n}", *score(as_config(cfg)), cfg))

    import recipe_learning as RL
    t0, v0, t0q, v0q = rows[0][2], rows[0][3], rows[0][4], rows[0][5]

    def val_ci(vq: dict) -> tuple[float, float, float]:
        """Paired CI of the val difference against theta_0, and its mean."""
        qids = [q for q in v0q if q in vq]
        d = [vq[q] - v0q[q] for q in qids]
        lo, hi = RL.bootstrap_ci(d)
        return lo, hi, (sum(d) / len(d) if d else 0.0)

    print(f"[{a.dataset}]  theta_0  train={t0:.4f}  val={v0:.4f}  "
          f"(train n={len(t0q)}, val n={len(v0q)})")
    print(f"  {'round':>8s} {'train':>8s} {'val':>8s}  {'Δtrain':>8s} "
          f"{'Δval':>8s}  {'Δval CI95':>20s}  sep")
    for r in rows[1:]:
        n, name, tr, va = r[0], r[1], r[2], r[3]
        lo, hi, _ = val_ci(r[5])
        sep = "yes" if lo > 0 else ("no" if hi >= 0 else "worse")
        print(f"  {name:>8s} {tr:8.4f} {va:8.4f}  {tr-t0:+8.4f} {va-v0:+8.4f}"
              f"  [{lo:+7.4f},{hi:+7.4f}]  {sep}")

    ok = [r for r in rows[1:] if r[2] > t0 and r[3] > v0]
    if not ok:
        print("  → no round beats theta_0 on both folds; keeping theta_0")
        return
    win = max(ok, key=lambda r: r[3])
    wlo, whi, _ = val_ci(win[5])
    print(f"  → selected {win[1]}: train={win[2]:.4f} val={win[3]:.4f}")
    # Say which kind of selection this is. `sep=no` does not invalidate the
    # choice -- something has to be chosen -- but a number reported from it
    # must not be presented as a measured improvement.
    if wlo > 0:
        print(f"    val difference is separated from theta_0: "
              f"CI95 [{wlo:+.4f}, {whi:+.4f}]")
    else:
        print(f"    !! NOT separated from theta_0 on val: "
              f"CI95 [{wlo:+.4f}, {whi:+.4f}] includes 0 (val n={len(v0q)}). "
              f"This is a ranking, not a measured gain -- report it as such.")
    if a.write:
        path = f"{run_dir}/best_config_val.json"
        json.dump(win[6], open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  → {path}")


# ======================================================================
# runs.py percat — #!/usr/bin/env python3
# ======================================================================

PERCAT = "evolution_percat"


# `as_config` is defined once, above: select_on_val filtered by the dataclass
# field list and assemble_percat by hasattr, which is the same filter written
# twice.


def class_to_category(dataset: str) -> dict[str, int]:
    """Map Axis A class name -> the integer the overrides are keyed on."""
    qs = json.load(open(f"{CASE_DIR}/{dataset}_queries.json", encoding="utf-8"))
    out: dict[str, int] = {}
    for q in qs:
        cl, cat = q.get("legal_class"), q.get("category")
        if cl is not None and cat is not None:
            out.setdefault(cl, int(cat))
    return out


def pick(run_dir: str) -> dict | None:
    for name in ("best_config_val.json", "best_config.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    return None


def _cli_assemble_percat():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default=None,
                    help="directory holding the global best config to use as "
                         "the base; default is theta_0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--evaluate", default=None, choices=["train", "val", "test"],
                    help="also score base vs assembled on this split")
    a = ap.parse_args()

    base_cfg = weak_config()
    base_name = "theta_0"
    if a.base:
        d = pick(f"{a.base}/{a.dataset}")
        if d is None:
            raise SystemExit(f"base config not found under {a.base}/{a.dataset}")
        base_cfg, base_name = as_config(d), a.base
    base = asdict(base_cfg)

    cat_of = class_to_category(a.dataset)
    overrides: dict[str, dict] = {}
    skipped: list[str] = []
    for cl in sorted(os.listdir(PERCAT)) if os.path.isdir(PERCAT) else []:
        run_dir = f"{PERCAT}/{cl}/{a.dataset}"
        if not os.path.isdir(run_dir):
            continue
        cfg = pick(run_dir)
        if cfg is None:
            skipped.append(f"{cl} (no selected config -- kept theta_0)")
            continue
        if cl not in cat_of:
            skipped.append(f"{cl} (no category id)")
            continue
        diff = {k: v for k, v in cfg.items()
                if k in base and base[k] != v and k != "per_category_overrides"
                and not isinstance(v, (dict, list))}
        if not diff:
            skipped.append(f"{cl} (selected config equals the base)")
            continue
        overrides[str(cat_of[cl])] = diff
        print(f"  {cl:20s} -> category {cat_of[cl]}: {diff}")
    for s in skipped:
        print(f"  {s}")

    assembled = RetrievalConfig(**{k: v for k, v in base.items()
                                   if k != "per_category_overrides"})
    assembled.per_category_overrides = overrides
    out = a.out or f"{PERCAT}/{a.dataset}_assembled.json"
    json.dump(asdict(assembled), open(out, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nbase = {base_name};  {len(overrides)} class override(s)  ->  {out}")

    if a.evaluate:
        cases = json.load(open(f"{CASE_DIR}/{a.dataset}_cases.json", encoding="utf-8"))
        qs = json.load(open(f"{CASE_DIR}/{a.dataset}_queries.json", encoding="utf-8"))
        index = L.build_index(a.dataset, cases, qs)
        b = evaluate(a.dataset, base_cfg, a.evaluate, 30, index=index, quiet=True)[0]
        m = evaluate(a.dataset, assembled, a.evaluate, 30, index=index, quiet=True)[0]
        print(f"\n{a.dataset} on {a.evaluate} ({b['_n_queries']} queries)")
        print(f"  base ({base_name:18s}) NDCG@30 = {b['NDCG@30']:.4f}")
        print(f"  assembled per-class     NDCG@30 = {m['NDCG@30']:.4f}"
              f"   ({m['NDCG@30'] - b['NDCG@30']:+.4f})")



# ── one entry point ───────────────────────────────────────────────────────
STAGES = {"trace": _cli_trace, "select": _cli_select_on_val,
          "percat": _cli_assemble_percat}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__)
        return 2
    stage, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [f"runs.py {stage}"] + rest
    STAGES[stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
