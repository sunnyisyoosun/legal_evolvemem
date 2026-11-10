"""Re-score every evolution round's config on the validation fold.

Evolution runs on `train` and the engine keeps the round with the best train
score. That is the right loop to leave alone -- val must stay out of it -- but
the round that wins on train is not necessarily the one that transfers. Every
round's config is already on disk, so the check costs one pass over val per
round and no re-running of the loop.

Selection rule: among rounds that beat theta_0 on BOTH folds, take the highest
val. Requiring train as well keeps a round that merely got lucky on a small
val (LeCaRD's is 17 queries) from being picked.

    python select_on_val.py --dataset LeCaRDv2 --dir evolution_global
"""
from __future__ import annotations

import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
from evaluate import CASE_DIR, evaluate, weak_config
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
    cases = json.load(open(f"{CASE_DIR}/{a.dataset}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{a.dataset}_queries.json", encoding="utf-8"))
    index = L.build_index(a.dataset, cases, qs)

    def score(cfg):
        return tuple(evaluate(a.dataset, cfg, sp, 30, index=index, quiet=True,
                              legal_class=a.legal_class)[0]["NDCG@30"]
                     for sp in ("train", "val"))

    rows = [(-1, "theta_0", *score(weak_config()))]
    seen = set()
    for n, cfg in round_configs(run_dir):
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key in seen:                      # a rejected round reverts; skip duplicates
            continue
        seen.add(key)
        rows.append((n, f"round_{n}", *score(as_config(cfg)), cfg))

    t0, v0 = rows[0][2], rows[0][3]
    print(f"[{a.dataset}]  theta_0  train={t0:.4f}  val={v0:.4f}")
    print(f"  {'round':>8s} {'train':>8s} {'val':>8s}  {'Δtrain':>8s} {'Δval':>8s}")
    for r in rows[1:]:
        n, name, tr, va = r[0], r[1], r[2], r[3]
        print(f"  {name:>8s} {tr:8.4f} {va:8.4f}  {tr-t0:+8.4f} {va-v0:+8.4f}")

    ok = [r for r in rows[1:] if r[2] > t0 and r[3] > v0]
    if not ok:
        print("  → no round beats theta_0 on both folds; keeping theta_0")
        return
    win = max(ok, key=lambda r: r[3])
    print(f"  → selected {win[1]}: train={win[2]:.4f} val={win[3]:.4f}")
    if a.write:
        path = f"{run_dir}/best_config_val.json"
        json.dump(win[4], open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  → {path}")


if __name__ == "__main__":
    main()
