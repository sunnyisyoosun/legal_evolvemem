"""Rewrite a case fact pattern into a retrieval query, and merge with the original.

A LeCaRD query is a narrative: names, addresses, plate numbers, dates and exact
amounts, interleaved with the conduct that actually decides which precedents are
alike. BM25 spends its weight on the incidental half. This is the same failure
entity-swap exploits by deleting party names, generalised: the model is asked to
keep the conduct and the legal issue and drop the rest.

EvolveMem's own intent_planning was measured at -0.088 here -- decomposing a fact
pattern into LoCoMo-style multi-hop sub-questions does not fit case law -- so the
rewrite is a single normalised query rather than a set of sub-questions, and it is
fused with the original by RRF so a bad rewrite cannot cost more than it gains.
"""
from __future__ import annotations
import argparse, json, os, sys, time, collections, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
import registry as REG
from evolvemem.multi_retriever import RetrievalConfig
from evaluate import CASE_DIR, QRELS, EVAL_K, weak_config

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "rewrite")

# The rewrite prompt is no longer a constant: it is assembled from the Rule
# pool (recipes.RULES), the third recipe type. A Rule is text, not a field --
# `query_rewrite_repeat` is the only trace rewriting had in RetrievalConfig,
# and an integer cannot say WHAT to rewrite. METHODOLOGY 3 ("세 타입").
#
# With the single seed rule the assembled string is byte-identical to the
# constant that used to live here, which is what keeps cache/rewrite/*.json
# attributable: the cache is keyed by query id, so a changed prompt would
# silently leave old rewrites in place under a new rule. verify.py asserts
# the identity, and `cache_stamp()` is what a new pool must write beside the
# cache so the two cannot drift unnoticed.
PROMPTS = {lang: REG.assemble_rewrite_prompt(lang) for lang in ("zh", "ko", "en")}


def cache_stamp(lang: str) -> str:
    """Identity of the prompt that produced a cache, for the sidecar."""
    import hashlib
    names = ",".join(r.name for r in REG.RULES)
    digest = hashlib.sha256(PROMPTS[lang].encode("utf-8")).hexdigest()[:12]
    return f"{names}|{lang}|{digest}"


def rewrite_all(llm, dataset: str, rows: list[dict], lang: str,
                max_chars: int = 2500, cache_path: str | None = None) -> dict:
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    todo = [r for r in rows if r["query_id"] not in cache]
    print(f"  cached {len(cache)}, to rewrite {len(todo)}", flush=True)
    for i, r in enumerate(todo):
        msg = [{"role": "user",
                "content": PROMPTS[lang].format(text=r["text"][:max_chars])}]
        out = (llm(msg, 300, 0.0) or "").strip()
        cache[r["query_id"]] = out
        if cache_path and (i + 1) % 50 == 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"    {i+1}/{len(todo)}", flush=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
        # Which rule produced this cache. The cache is keyed by query id, so
        # a pool change leaves every old rewrite in place under a new rule
        # and nothing notices; measure.py (rules) refuses to attribute a
        # measurement to a rule the stamp does not name.
        with open(os.path.splitext(cache_path)[0] + ".stamp", "w",
                  encoding="utf-8") as f:
            f.write(cache_stamp(lang) + "\n")
    return cache


def rrf(lists: list[list[str]], k: int = 60, top_n: int = 30) -> list[str]:
    score: dict[str, float] = collections.defaultdict(float)
    for lst in lists:
        for rank, d in enumerate(lst):
            score[d] += 1.0 / (k + rank)
    return [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])][:top_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen")
    a = ap.parse_args()

    from legal_evolve import make_llm_call
    from legal_meta import LANG_OF
    lang = LANG_OF[a.dataset]
    cases = json.load(open(f"{CASE_DIR}/{a.dataset}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{a.dataset}_queries.json", encoding="utf-8"))
    labels = L.MBR.load_qrels(QRELS.format(ds=a.dataset))
    rows = [q for q in qs if q["query_id"] in labels
            and q["split"] == a.split][:a.limit]

    t0 = time.time()
    rw = rewrite_all(make_llm_call(a.base_url, a.model), a.dataset, rows, lang,
                     cache_path=f"{CACHE}/{a.dataset}.json")
    print(f"  rewrite {time.time()-t0:.0f}s", flush=True)
    shown = next((r for r in rows if rw.get(r["query_id"])), None)
    if shown:
        print(f"\n  원쿼리({len(shown['text'])}자): {shown['text'][:110]}")
        print(f"  재작성({len(rw[shown['query_id']])}자): {rw[shown['query_id']][:200]}\n")

    index = L.build_index(a.dataset, cases, qs)
    cfg = weak_config()
    def run_text(q, text):
        row = dict(q); row["text"] = text
        return L.search(index, row, cfg, top_n=30)

    res = collections.defaultdict(list)
    for q in rows:
        gold = labels[q["query_id"]]
        r = rw.get(q["query_id"]) or ""
        runs = {"원쿼리": L.search(index, q, cfg, top_n=30)}
        if r:
            runs["재작성 단독"] = run_text(q, r)
            # Expansion: the rewrite is appended to the original rather than
            # replacing it, repeated to weight its terms more heavily.
            for n in (1, 2, 4):
                runs[f"확장 x{n}"] = run_text(q, q["text"] + (" " + r) * n)
            runs["RRF(원+재작성)"] = rrf([runs["원쿼리"], runs["재작성 단독"]])
            runs["RRF(원+확장x2)"] = rrf([runs["원쿼리"], runs["확장 x2"]])
        for name, lst in runs.items():
            res[name].append(L.MBR.compute_query_metrics(lst, gold, EVAL_K)["NDCG@30"])
    print(f"[{a.dataset}] split={a.split} n={len(rows)}")
    base = st.mean(res["원쿼리"])
    for k, v in sorted(res.items(), key=lambda kv: -st.mean(kv[1])):
        m = st.mean(v)
        print(f"  {k:16s} NDCG@30={m:.4f}  ({m-base:+.4f})")


if __name__ == "__main__":
    main()
