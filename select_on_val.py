"""Re-score every evolution round's config on the validation fold.

Evolution runs on `train` and the engine keeps the round with the best train
score. That is the right loop to leave alone -- val must stay out of it -- but
the round that wins on train is not necessarily the one that transfers. Every
round's config is already on disk, so the check costs one pass over val per
round and no re-running of the loop.

Selection rule: among rounds that beat theta_0 on BOTH folds, take the highest
val. Requiring train as well keeps a round that merely got lucky on a small
val from being picked.

That guard is not enough on LeCaRD. Its val fold is **8 queries** (107 total,
split 79/8/20 -- the figure of 17 in an earlier version of this docstring was
wrong). With n=8 the paired interval around a val difference is wide enough
that "highest val" among several candidates is close to a coin flip, so every
Dval is printed WITH its interval and the selection line says plainly whether
the winner is separated from theta_0 or merely ranked above it. The data is
what it is -- LeCaRD ships 107 queries -- so the fold is not changed; what
changes is that the output no longer presents an unresolvable ordering as a
result.

    python select_on_val.py --dataset LeCaRDv2 --dir evolution_global
"""
from __future__ import annotations

import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
from evaluate import EVAL_K, evaluate, load_fold, weak_config
from evolvemem.multi_retriever import RetrievalConfig

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


def main():
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


if __name__ == "__main__":
    main()
