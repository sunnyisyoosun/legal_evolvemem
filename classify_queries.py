"""Assign each query a legal class, a difficulty class, and a composite id.

Writes the labels back into corpus/{ds}_queries.json and records the id
registry, so the retrieval wrapper can pass a category straight into
EvolveMem's per_category_overrides.

Difficulty cuts are fitted on the training split only. The score profile that
feeds them uses no relevance labels, so the same classification is available at
query time on an unseen query.
"""
from __future__ import annotations
import argparse, json, os, sys, time, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import legal_retriever as L
from legal_meta import LANG_OF
from legal_taxonomy import legal_class, build_charge_to_article, composite

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.join(HERE, "corpus")
POOL = {"LeCaRD": "/home/yoosun/legal/result/LeCaRD/candidate_pool.json"}


def lexical_scores(index, query_text: str, pool_ids=None) -> list[float]:
    tokens = index.tokenize(query_text)
    if not tokens:
        return []
    if pool_ids:
        scores, _ = index.bm25.get_scores_subpool(tokens, pool_ids)
        return list(scores)
    scores = index.bm25.get_scores(tokens)
    top = np.argpartition(-scores, min(30, len(scores) - 1))[:30]
    return [float(scores[i]) for i in top]


# Every benchmark ships a train/val/test split; the files under result/ are
# the one source that carries all three for all four. `output/` has a val for
# LEGAR only, and it is a DIFFERENT re-split: its 476 val rows and result's
# 429 are disjoint, while test is identical in both. Using one source keeps
# the four datasets comparable.
OFFICIAL_SPLIT = {
    "COLIEE": "COLIEE",
    "LeCaRD": "LeCaRD",
    "LeCaRDv2": "LeCaRDv2",
    "LEGAR_stricter": "LEGAR_BENCH",
}
SPLIT_PATH = "/home/yoosun/legal/result/{dir}/query_structured.json"


def assign_fold(qs: list[dict], dataset: str) -> dict:
    """Attach the benchmark's official train / val / test split as `fold`.

    Evolution runs on `train`, every check and repair happens on `val`, and
    `test` is scored once at the end. Reading `test` between runs would turn
    it into a validation set, which is what `val` is for.

    Raises when the official file does not cover every query: a silent
    fallback would mean two runs disagreeing about which queries are held
    out, and their numbers would not be comparable.
    """
    path = SPLIT_PATH.format(dir=OFFICIAL_SPLIT[dataset])
    with open(path, encoding="utf-8") as f:
        official = json.load(f)
    missing = [q["query_id"] for q in qs if str(q["query_id"]) not in official]
    if missing:
        raise KeyError(f"{dataset}: {len(missing)} queries absent from {path} "
                       f"(e.g. {missing[:3]})")
    for q in qs:
        q["fold"] = official[str(q["query_id"])]["split"]
    return collections.Counter(q["fold"] for q in qs)


def run(dataset: str) -> dict:
    """Label every query with its Axis-A class and a stable integer id.

    The retrieval-difficulty axis that used to sit alongside this was dropped:
    it was derived from BM25's own score spread, and measurement showed it
    predicted how hard a query is (weakly) without predicting that a different
    configuration would serve it better -- the per-class gain fell to zero once
    a globally good lever was available.
    """
    cases = json.load(open(f"{CASE_DIR}/{dataset}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{dataset}_queries.json", encoding="utf-8"))
    c2a = build_charge_to_article(cases) if dataset.startswith("LeCaRD") else None

    registry: dict[str, int] = {}
    for q in qs:
        q["legal_class"] = legal_class(q, dataset, c2a)
        q["category"] = composite(q["legal_class"], "", registry)
        q.pop("difficulty_class", None)
        q.pop("profile_gap", None)
        q.pop("profile_entropy", None)

    assign_fold(qs, dataset)

    json.dump(qs, open(f"{CASE_DIR}/{dataset}_queries.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    meta = {"dataset": dataset, "registry": registry,
            "counts": {k: v for k, v in
                       collections.Counter(q["legal_class"] for q in qs).most_common()}}
    json.dump(meta, open(f"{CASE_DIR}/{dataset}_categories.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LANG_OF))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for ds in (sorted(LANG_OF) if a.all else [a.dataset]):
        meta = run(ds)
        qs = json.load(open(f"{CASE_DIR}/{ds}_queries.json", encoding="utf-8"))
        test = collections.Counter(q["legal_class"] for q in qs if q["split"] == "test")
        folds = collections.Counter(q.get("fold") for q in qs)
        print(f"[{ds}] classes={len(meta['registry'])}  folds="
              f"train {folds.get('train', 0)} / val {folds.get('val', 0)} / "
              f"test {folds.get('test', 0)}")
        for k, v in meta["counts"].items():
            print(f"     all {v:6d}  test {test.get(k, 0):5d}  {k}")


if __name__ == "__main__":
    main()
