#!/usr/bin/env python3
"""The seven fusion settings of METHODOLOGY 8, on a chosen split.

Section 8 was originally run ad hoc on **test**, and its numbers were then
written into RECIPES_LEGAL -- which the evolution prompt shows the LLM. That
is test leakage into the method, so the settings live here now and the
recipe numbers are re-derived on **val**.

    python fusion_sweep.py --split val
"""
from __future__ import annotations
import argparse, json
import legal_retriever as L
from evaluate import CASE_DIR, evaluate, weak_config
from evolvemem.multi_retriever import RetrievalConfig

DATASETS = ["LeCaRDv2", "LeCaRD", "COLIEE", "LEGAR_stricter"]


def configs() -> dict[str, RetrievalConfig]:
    """The seven settings, spelled out.

    Shared by all: max_context=100, keyword_top_k=100, weight_keyword=1.0.
    `weight_structured` is left at the repo default (0.5) wherever the
    metadata view is on -- Section 8 never set it, and reading a default as a
    deliberate choice is the mistake that broke two recipes (Section 9).
    """
    kw = dict(max_context=100, keyword_top_k=100, weight_keyword=1.0)
    sem = dict(semantic_top_k=100, weight_semantic=0.3)
    swap = dict(enable_entity_swap=True, entity_swap_keyword_k=30,
                entity_swap_semantic_k=30)
    return {
        "1_bm25_only":        RetrievalConfig(**kw, fusion_mode="keyword_only",
                                              semantic_top_k=0, structured_top_k=0,
                                              enable_entity_swap=False),
        "2_rrf_semantic":     RetrievalConfig(**kw, **sem, fusion_mode="rrf",
                                              structured_top_k=0,
                                              enable_entity_swap=False),
        "3_rrf_str10":        RetrievalConfig(**kw, fusion_mode="rrf",
                                              semantic_top_k=0, structured_top_k=10,
                                              enable_entity_swap=False),
        "4_rrf_sem_str10":    RetrievalConfig(**kw, **sem, fusion_mode="rrf",
                                              structured_top_k=10,
                                              enable_entity_swap=False),
        "5_rrf_swap30_str10": RetrievalConfig(**kw, **swap, fusion_mode="rrf",
                                              semantic_top_k=0, structured_top_k=10),
        "6_rrf_swap30":       RetrievalConfig(**kw, **swap, fusion_mode="rrf",
                                              semantic_top_k=0, structured_top_k=0),
        "7_ws_sem03_swap30":  RetrievalConfig(**kw, **sem, **swap,
                                              fusion_mode="weighted_sum",
                                              structured_top_k=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfgs = configs()
    table: dict[str, dict[str, float]] = {}
    for ds in a.datasets:
        cases = json.load(open(f"{CASE_DIR}/{ds}_cases.json", encoding="utf-8"))
        qs = json.load(open(f"{CASE_DIR}/{ds}_queries.json", encoding="utf-8"))
        index = L.build_index(ds, cases, qs)
        row = {}
        for name, cfg in cfgs.items():
            m = evaluate(ds, cfg, a.split, 30, index=index, quiet=True)[0]
            row[name] = m["NDCG@30"]
            print(f"  {ds:16s} {name:20s} {m['NDCG@30']:.4f}", flush=True)
        table[ds] = row
        del index

    t0key = "1_bm25_only"
    print(f"\nNDCG@30 on {a.split}; delta vs theta_0 (= {t0key})\n")
    hdr = f"{'setting':22s}" + "".join(f"{d:>17s}" for d in a.datasets)
    print(hdr); print("-" * len(hdr))
    for name in cfgs:
        line = f"{name:22s}"
        for ds in a.datasets:
            v, t0 = table[ds][name], table[ds][t0key]
            line += f"{v:8.4f}{v - t0:+9.4f}" if name != t0key else f"{v:8.4f}{'':>9s}"
        print(line)

    if a.out:
        json.dump({"split": a.split, "scores": table}, open(a.out, "w"), indent=2)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
