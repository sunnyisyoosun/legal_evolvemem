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
from evolvemem.multi_retriever import RetrievalConfig
from evaluate import CASE_DIR, QRELS, EVAL_K, weak_config

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "rewrite")

PROMPTS = {
    "zh": """你是法律检索助手。下面是一份刑事案件的事实描述。

请改写成一个用于检索相似判例的查询，要求：
1. 删除所有当事人姓名、地名、车牌号、身份证号、具体日期和具体金额数字
2. 保留犯罪手段、行为方式、情节特征、法律争点
3. 用简洁的法律术语表述，不要叙事

事实描述：
{text}

只输出改写后的查询，不要解释。""",
    "ko": """다음은 형사사건 사실관계입니다.

유사 판례 검색용 질의로 바꾸세요:
1. 당사자 이름, 지명, 차량번호, 구체적 날짜·금액은 제거
2. 범행 수법, 행위 유형, 법적 쟁점은 유지
3. 서사가 아닌 간결한 법률 용어로

사실관계:
{text}

변환된 질의만 출력하세요.""",
    "en": """Rewrite the following case facts into a query for retrieving similar cases.

Drop party names, places, dates and exact amounts. Keep the legal issue, the
area of law, and the conduct at stake. Use concise legal terms, not narrative.

Case facts:
{text}

Output only the rewritten query.""",
}


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
