#!/usr/bin/env python3
"""Assemble the per-class evolved settings into ONE deployable config.

Evolving per class is only half of it: the settings have to be *applied* per
class at retrieval time too, or they sit unused in separate files. The engine
already supports that -- `resolve_config()` swaps in
`per_category_overrides[category]` for each query -- so the missing piece is
turning `evolution_percat/<class>/<dataset>/best_config_val.json` into those
overrides.

A class that kept theta_0 contributes no override: it falls through to the
base config, which is what "no change was better" means.

    python assemble_percat.py --dataset LeCaRDv2 --base evolution_official
    python assemble_percat.py --dataset LeCaRDv2 --evaluate val
"""
from __future__ import annotations
import argparse, json, os
import legal_retriever as L
from evaluate import CASE_DIR, evaluate, weak_config
from evolvemem.multi_retriever import RetrievalConfig
from dataclasses import asdict

PERCAT = "evolution_percat"


def as_config(d: dict) -> RetrievalConfig:
    c = RetrievalConfig()
    for k, v in (d or {}).items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c


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


def main():
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


if __name__ == "__main__":
    main()
