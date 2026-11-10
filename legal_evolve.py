"""Run EvolveMem's self-evolution loop on legal case retrieval.

The engine is built for QA: it retrieves, generates an answer, and scores the
answer with token-F1. Legal case retrieval has no answer to generate -- the
ranking itself is the output, scored with NDCG@30. Three seams are used so the
engine's own loop (diagnose -> propose -> guard -> revert) stays untouched:

1. `evolution.MultiViewIndex` is replaced by a factory returning the prebuilt
   LegalIndex, which carries the language-aware BM25, the precomputed
   embeddings and the pool mask.
2. `_generate_answer` is overridden to emit the ranked doc ids instead of
   calling an LLM. This removes answer generation from the loop entirely,
   which is most of its cost.
3. The adapter scores those ids against qrels with the project's own metric
   function, so evolution optimises the same number the baselines report.

Usage:
    python legal_evolve.py --dataset LeCaRDv2 --rounds 7 \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B-Instruct
"""
from __future__ import annotations

import argparse
import logging
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L                      # patches analyze_query on import
import evolvemem.diagnosis as DIAG
import evolvemem.evolution as EV
from evolvemem.benchmarks import BenchmarkAdapter, BenchmarkSample, register_adapter
from evolvemem.evolution import EvolutionConfig, EvolutionEngine
from evolvemem.multi_retriever import RetrievalConfig
from evaluate import CASE_DIR, QRELS, load_pool, EVAL_K, weak_config, in_split

PRIMARY = "NDCG@30"
TOP_N = 30


# ── Adapter ───────────────────────────────────────────────────────────────
class LegalRetrievalAdapter(BenchmarkAdapter):
    """Scores a ranked list of doc ids against graded relevance judgements."""

    name = "legal_retrieval"
    primary_metric = PRIMARY
    subcategory_keys = ("category",)

    def load(self, path: str, **kw) -> list[BenchmarkSample]:  # pragma: no cover
        raise NotImplementedError("queries are built by build_qa_pairs()")

    def score(self, prediction: str, reference: str, qa: dict) -> float:
        return self.score_all(prediction, reference, qa)[self.primary_metric]

    def score_all(self, prediction: str, reference: str, qa: dict) -> dict:
        qrels = (qa.get("meta") or {}).get("extras", {}).get("qrels") or {}
        ranked = [d for d in str(prediction).split() if d]
        if not qrels:
            return {self.primary_metric: 0.0}
        return L.MBR.compute_query_metrics(ranked, qrels, EVAL_K)


def build_qa_pairs(dataset: str, split: str, use_pool: bool = False,
                   legal_class: str | None = None) -> list[dict]:
    """QA pairs in the engine's shape, carrying qrels and the query row.

    `answer` is unused -- nothing is generated -- but the engine expects the
    key to exist.
    """
    queries = json.load(open(f"{CASE_DIR}/{dataset}_queries.json", encoding="utf-8"))
    labels = L.MBR.load_qrels(QRELS.format(ds=dataset))
    cases = json.load(open(f"{CASE_DIR}/{dataset}_cases.json", encoding="utf-8"))
    docmeta = {m["doc_id"]: m for m in cases}
    pool = load_pool(dataset) if use_pool else None
    out = []
    for q in queries:
        if q["query_id"] not in labels:
            continue
        if not in_split(q, split):
            continue
        if legal_class is not None and q.get("legal_class") != legal_class:
            continue
        gold = [d for d, rel in labels[q["query_id"]].items() if rel > 0][:3]
        out.append({
            "question": q["text"],
            # Unused for scoring -- that comes from qrels -- but it is the
            # field the failure log prints as the reference, so it carries a
            # digest of what a correct case looks like for this query.
            "answer": "GOLD " + " ; ".join(f"{d}[{_describe(docmeta.get(d, {}))}]"
                                           for d in gold),
            "category": int(q.get("category", 0)),
            "meta": {"qid": q["query_id"],
                     "qtype": f'{q.get("legal_class", "")} x {q.get("difficulty_class", "")}'.strip(" x"),
                     "extras": {"qrels": labels[q["query_id"]],
                                "row": q,
                                "pool": pool.get(q["query_id"]) if pool else None}},
        })
    return out


# ── Engine ────────────────────────────────────────────────────────────────
class LegalEvolutionEngine(EvolutionEngine):
    """EvolutionEngine whose 'answer' is the ranking itself."""

    def __init__(self, *a, index: L.LegalIndex, **kw):
        super().__init__(*a, **kw)
        self.index = index

    def _generate_answer(self, question, retrieved, category, qa=None) -> str:
        """Return the retrieved doc ids, space-separated. No LLM call."""
        return " ".join(self.index.resolve(retrieved)[:TOP_N])

    def _verify_answer(self, question, retrieved, candidate, category, qa=None,
                       style: str = "strict") -> str:
        """No-op. The second pass rewrites a natural-language answer, which
        here is a ranked list of document ids -- rewriting it destroys the
        ranking. Left reachable so the engine may still toggle the dimension;
        it simply has no effect in a retrieval task."""
        return candidate


# Dimensions that exist in RetrievalConfig but cannot affect a retrieval task:
# answer-generation levers (nothing is generated) and LoCoMo's per-split prompt
# flags. Left in the config so the engine stays intact, but hidden from the
# diagnosis prompt -- otherwise the LLM spends every round proposing them and
# every round is rejected with delta +0.0000.
INERT_DIMENSIONS = (
    "locomo_cat", "answer_style", "enable_answer_verification",
    "verification_style", "answer_model",
)


def _install_diagnosis_context(qa_pairs: list[dict]) -> None:
    """Tell the diagnosis LLM what our categories actually are.

    The engine labels categories with LoCoMo's splits (1=SingleHop,
    5=Adversarial, ...). Ours are penal-code chapter x retrieval difficulty,
    so without this the LLM reads a Chinese case-law run as a conversational
    QA one and proposes accordingly.
    """
    names = {}
    for qa in qa_pairs:
        cat = int(qa.get("category", 0))
        label = (qa.get("meta") or {}).get("qtype") or ""
        if cat and cat not in names:
            names[cat] = label or f"class{cat}"
    DIAG.set_category_names(names)
    DIAG.set_prompt_exclude(INERT_DIMENSIONS)
    # Restrict the action space to the fields MEASURED to change the ranking
    # in this pipeline (recipe_learning.EFFECTIVE_FIELDS: 12 of 45; see that
    # module for the probe). Before this, 37% of the field changes the loop
    # made over 16 runs landed on fields that cannot move the metric --
    # `enable_intent_planning` 28 times, `entity_swap_semantic_k` 18,
    # `intent_final_context` 12 -- each one a step-cap slot spent on a round
    # that could not teach anything. The remaining 33 fields stay in
    # RetrievalConfig at their defaults; they are simply not offered.
    import recipe_learning as _RLF
    DIAG.set_action_fields(_RLF.EFFECTIVE_FIELDS)
    print(f"action space: {len(_RLF.EFFECTIVE_FIELDS)} measured-effective "
          f"field(s) + per_category_overrides "
          f"(of {len(_RLF._FIELD_TYPE)} in RetrievalConfig)", flush=True)
    # Legal fact patterns and case digests do not fit the defaults sized for
    # conversational utterances.
    DIAG.set_sample_widths(question=320, field=200, retrieved_shown=5)
    # Drop the built-in TODO checklist. Its three generic items are LoCoMo
    # numbers ("fusion_mode -> rrf: +5-10 pp, NEAR-UNIVERSAL WIN") that are
    # wrong here -- switching fusion while every other view is off measured
    # exactly +0.0000 -- and the prompt tells the LLM to take ONE item per
    # round, which is the fragmentation RECIPES_LEGAL exists to avoid. The
    # recipes carry our own measured numbers for the same levers.
    DIAG.set_todo_items([])
    # Rendered into the prompt once per round, then cleared so the next
    # round counts its own retrievals.
    def _block():
        body = view_report()
        inert = inert_report()
        _reset_view_stats()
        _reset_lever_stats()
        if inert:
            body += ("\n\n### Levers that were ON but never acted\n" + inert)
        return body
    DIAG.set_extra_context(_block)
    # Lets the diagnosis layer name WHY a round has no proposal instead of
    # recording it as a round that proposed nothing.
    DIAG.set_llm_status_fn(lambda: LLM_STATUS[-1] if LLM_STATUS else "empty")
    # Names the config fields behind each inert lever, so a +0.0000 from a
    # lever that could not act does not strike its recipe off the pool.
    _INERT_FIELDS = {
        "query_rewrite": ["enable_query_rewrite", "query_rewrite_repeat"],
        "chapter_filter": ["enable_chapter_filter"],
        "view:semantic": ["weight_semantic", "semantic_top_k"],
        "view:metadata": ["weight_structured", "structured_top_k"],
        "view:party-strip": ["enable_entity_swap", "entity_swap_keyword_k",
                             "entity_swap_semantic_k"],
        "fusion_mode": ["fusion_mode"],
    }

    def _inert():
        out = []
        for name, (acted, on) in _LEVER_STATS.items():
            if on and acted == 0:
                out.extend(_INERT_FIELDS.get(name, []))
        return out
    DIAG.set_inert_fields_fn(_inert)


# Levers that were switched ON this round, and how often they actually acted.
# Measured in the retrieve seam rather than inferred from the config, because
# the question is not "is the flag set" but "did anything happen".
_LEVER_STATS: dict[str, list[int]] = {}      # name -> [acted, on]


def _note_lever(name: str, acted: bool) -> None:
    c = _LEVER_STATS.setdefault(name, [0, 0])
    c[1] += 1
    if acted:
        c[0] += 1


def inert_report() -> str:
    """Prompt block naming levers that were on but never acted.

    A lever that cannot act still scores, and it scores exactly +0.0000. The
    round then enters attempt_history as "tried, rejected", which the next
    diagnosis reads as a measured refutation. It is not one -- nothing was
    exercised. Saying so is what stops the LLM concluding something the run
    never supported (METHODOLOGY 13).
    """
    REASON = {
        "query_rewrite": "no rewrite exists for these queries "
                         "(cache/rewrite/<dataset>.json missing or empty)",
        "chapter_filter": "no query carries a legal_class, so the filter fell "
                          "back to the unrestricted corpus every time",
        "view:semantic": "semantic_top_k=0, so the semantic view returns no "
                         "candidates. weight_semantic cannot change any score "
                         "while top_k is 0 -- raise semantic_top_k first",
        "view:metadata": "structured_top_k=0, so the metadata view returns no "
                         "candidates. weight_structured cannot change any "
                         "score while top_k is 0",
        "view:party-strip": "enable_entity_swap is on but its top_k is 0, so "
                            "the party-strip view returns no candidates",
        "fusion_mode": "only ONE view is active, and every fusion mode "
                       "reduces to that view's own ranking when there is "
                       "nothing to fuse. rrf with a single view is "
                       "arithmetically identical to keyword_only. Activate a "
                       "second view before changing fusion_mode",
    }
    lines = []
    for name, (acted, on) in sorted(_LEVER_STATS.items()):
        if on and acted == 0:
            lines.append(
                f"- {name}: ON for {on} queries, acted on 0. "
                f"{REASON.get(name, 'it could not take effect')}. This field "
                f"cannot change any score, so a delta of +0.0000 is NOT "
                f"evidence against it. Do not conclude it is ineffective.")
    return "\n".join(lines)


def _note_view_levers(cfg) -> None:
    """A weight on a view whose top_k is 0, or a fusion mode with one view.

    These cost a whole round and score exactly +0.0000, because nothing they
    touch can act. Observed directly: a round set fusion_mode=rrf with
    semantic_top_k=0 and scored theta_0 to four decimals, because rrf over a
    single view just reproduces that view's ranking (METHODOLOGY 14).
    """
    active = 0
    for name, k, w in (("semantic", "semantic_top_k", "weight_semantic"),
                       ("metadata", "structured_top_k", "weight_structured"),
                       ("party-strip", "entity_swap_keyword_k", None)):
        on = getattr(cfg, k, 0) > 0
        if name == "party-strip":
            on = on and getattr(cfg, "enable_entity_swap", False)
        if on:
            active += 1
        # Only report it as inert when something is asking it to contribute:
        # a weight above zero, or the swap switch flipped on.
        asked = (getattr(cfg, w, 0) > 0 if w else
                 getattr(cfg, "enable_entity_swap", False))
        if asked:
            _note_lever(f"view:{name}", on)
    if getattr(cfg, "keyword_top_k", 0) > 0:
        active += 1
    _note_lever("fusion_mode", active >= 2)


def _reset_lever_stats() -> None:
    _LEVER_STATS.clear()


def _describe(meta: dict) -> str:
    """One-line digest of a case: the fields that decide legal similarity."""
    parts = []
    for key, cap in (("charges", 2), ("articles", 3)):
        vals = meta.get(key) or []
        if vals:
            parts.append(",".join(str(v) for v in vals[:cap]))
    if meta.get("location"):
        parts.append(str(meta["location"]))
    if meta.get("timestamp"):
        parts.append(str(meta["timestamp"])[:7])
    return "|".join(parts) if parts else "-"


# ── Per-view contribution stats ───────────────────────────────────────────
# The diagnosis prompt describes failures but says nothing about which
# retrieval view produced the ranking, so the LLM has no grounds to reason
# about view weights or top-k. These counters are filled by the retrieval
# seam and rendered into the prompt once per round.
_VIEW_STATS: dict = {}


def _reset_view_stats() -> None:
    _VIEW_STATS.clear()


VIEW_ORDER = ("keyword", "semantic", "structured", "swap")


def _enabled_views(cfg) -> set:
    """Which views this config actually runs (mirrors retrieve_multiview)."""
    mode = getattr(cfg, "fusion_mode", "")
    on = set()
    if mode not in ("semantic_only", "structured_only") and getattr(cfg, "keyword_top_k", 0) > 0:
        on.add("keyword")
    if mode not in ("keyword_only", "structured_only") and getattr(cfg, "semantic_top_k", 0) > 0:
        on.add("semantic")
    if mode not in ("semantic_only", "keyword_only") and getattr(cfg, "structured_top_k", 0) > 0:
        on.add("structured")
    if getattr(cfg, "enable_entity_swap", False):
        on.add("swap")
    return on


def _note_views(hits, qrels: dict) -> None:
    """Record, for one query's top-N, what each view contributed."""
    for h in hits[:TOP_N]:
        views = [v for v in str(h.source).split(":")[0].split("+") if v]
        gold = float((qrels or {}).get(h.doc_id, 0)) > 0
        for v in views:
            st = _VIEW_STATS.setdefault(v, {"hits": 0, "gold": 0, "sole_gold": 0})
            st["hits"] += 1
            if gold:
                st["gold"] += 1
                if len(views) == 1:
                    st["sole_gold"] += 1
    _VIEW_STATS.setdefault("_n", {"hits": 0, "gold": 0, "sole_gold": 0})
    _VIEW_STATS["_n"]["hits"] += 1


def view_report() -> str:
    """Prompt block: which view actually produced the retrieved ranking.

    Every view is listed, including the ones this config does not run. An
    earlier version printed only the views that returned something, which
    made a disabled view indistinguishable from a useless one: at theta_0
    only BM25 runs, so the block read "keyword 100%" and the diagnosis LLM
    took that as evidence that the other views were not worth enabling --
    the opposite of what the numbers meant.
    """
    n = (_VIEW_STATS.get("_n") or {}).get("hits", 0)
    if not n:
        return ""
    on = _VIEW_STATS.get("_enabled") or set()
    stats = {v: st for v, st in _VIEW_STATS.items() if not v.startswith("_")}
    total_hits = sum(st["hits"] for st in stats.values()) or 1
    lines = [f"Across {n} queries, of the top-{TOP_N} documents each view helped "
             f"rank (a document can be credited to several views):", ""]
    lines.append(f"  {'view':10s} {'status':>8s} {'share':>7s} {'relevant':>9s} {'sole':>6s}")
    for v in VIEW_ORDER:
        if v not in on:
            lines.append(f"  {v:10s} {'OFF':>8s} {'-':>7s} {'-':>9s} {'-':>6s}")
            continue
        st = stats.get(v) or {"hits": 0, "gold": 0, "sole_gold": 0}
        lines.append(f"  {v:10s} {'on':>8s} {100*st['hits']/total_hits:6.1f}% "
                     f"{st['gold']:9d} {st['sole_gold']:6d}")
    lines.append("")
    lines.append("OFF means the view did not run under the current config -- it has "
                 "no numbers here, which is NOT evidence that enabling it would not "
                 "help.")
    lines.append("`sole` = relevant documents this view alone surfaced, i.e. documents "
                 "no other active view found. An active view with many hits but sole=0 "
                 "is only re-ranking what another view already found; raising its "
                 "weight or top_k will not reach new documents.")
    return "\n".join(lines)


def _install_seams(index: L.LegalIndex, qa_pairs: list[dict]) -> None:
    """Point the engine at our index, and set the pool/metadata per question.

    The engine calls retrieve_multiview directly, so the per-query state it
    cannot pass -- the candidate pool and the precomputed query metadata --
    is attached to the index before each retrieval.
    """
    EV.MultiViewIndex = lambda cases, embedder=None: index

    by_text = {qa["question"]: qa["meta"]["extras"] for qa in qa_pairs}
    original = L.MR.retrieve_multiview

    def retrieve(idx, question, config=None, category=None, **kw):
        extras = by_text.get(question) or {}
        idx.set_pool(extras.get("pool"))
        # Resolve the per-class override here: expand_query below needs the
        # resolved config. resolve_config clears the map so the engine does
        # not apply it twice.
        cfg = L.resolve_config(config, category) if config is not None else config
        row = extras.get("row") or {}
        text = L.expand_query(question, row, cfg, idx.dataset) if cfg else question
        if cfg is not None and getattr(cfg, "enable_query_rewrite", False):
            _note_lever("query_rewrite", text != question)
        with L.query_context(row):
            hits = original(idx, text, config=cfg, category=category, **kw)
        if cfg is not None and getattr(cfg, "enable_chapter_filter", False):
            # set_chapter ran inside retrieve_multiview; a mask equal to the
            # pool means it fell back and restricted nothing.
            _note_lever("chapter_filter", idx._mask != idx._pool_mask)
        if cfg is not None:
            _note_view_levers(cfg)
        # Attach each hit's identity and legal metadata to the field the
        # engine already forwards into the failure log, so the diagnosis LLM
        # sees what was retrieved rather than a bare document id.
        _VIEW_STATS.setdefault("_enabled", set()).update(_enabled_views(cfg))
        _note_views(hits, extras.get("qrels") or {})
        for h in hits:
            i = idx.docid2idx.get(h.doc_id)
            m = idx.memories[i] if i is not None else {}
            h.source = f"{h.source}:{h.doc_id}[{_describe(m)}]"
        return hits

    L.MR.retrieve_multiview = retrieve
    EV.retrieve_multiview = retrieve


# ── LLM ───────────────────────────────────────────────────────────────────
# A reasoning model spends its budget on the <think> block before writing the
# answer, and the engine asks for 2048 tokens -- enough for a plain model's JSON
# but not for one that reasons first. The floor keeps the answer from being cut
# off mid-object; vLLM's reasoning parser keeps the thinking out of `content`.
MIN_COMPLETION_TOKENS = 8192


def _qid_of(pair: dict):
    """Query id out of a QA pair, whichever shape build_qa_pairs produced."""
    meta = pair.get("meta") or {}
    return meta.get("qid") or pair.get("query_id") or pair.get("id")


def make_llm_call(base_url: str, model: str, api_key: str = "EMPTY",
                  seed: int | None = 0):
    """OpenAI-compatible call in the (messages, max_tokens, temperature) shape.

    `seed` is passed through because temperature=0 alone did not reproduce:
    two runs whose round-1 state was identical to six decimals produced
    different round-2 proposals. Continuous batching and MoE routing make a
    greedy decode order-dependent, so a seed narrows but does not close the
    gap -- replicate runs are still needed before calling a difference an
    effect.
    """
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    def llm_call(messages: list[dict], max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        # The floor exists so a diagnosis JSON is not cut off mid-object. It
        # must not apply to the per-query calls intent planning makes, which
        # ask for 512 and are issued once per query -- inflating those to 8192
        # buys nothing and slows a round that already runs thousands of them.
        if max_tokens >= 1024:
            max_tokens = max(max_tokens, MIN_COMPLETION_TOKENS)
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                **({"seed": seed} if seed is not None else {}))
            text = r.choices[0].message.content or ""
            LLM_STATUS.append("ok" if text else "empty_content")
            return text
        except Exception as e:
            LLM_STATUS.append(_classify_llm_error(e))
            print(f"  [llm] {LLM_STATUS[-1]}: {str(e)[:600]}", flush=True)
            return ""                   # engine reads the status, not the ""

    return llm_call


# Every LLM outcome, newest last. The engine is handed "" on failure because
# that is the interface it has, but "" alone made a dead round look like a
# round that proposed nothing -- and that reading then entered attempt_history
# as evidence (METHODOLOGY 13). The status is what tells the two apart.
LLM_STATUS: list[str] = []


def _classify_llm_error(e: Exception) -> str:
    """Name the failure so the round record says WHY it has no diagnosis."""
    msg = str(e).lower()
    if "maximum context length" in msg or "context_length" in msg:
        return "context_overflow"
    if "rate limit" in msg or "429" in msg:
        return "rate_limited"
    if "api key" in msg or "401" in msg or "403" in msg:
        return "auth"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    return f"error:{type(e).__name__}"


def llm_integrity_report(rounds: int) -> str:
    """One line stating whether the run's rounds are data points at all.

    Printed with the trajectory because the absence of this line is what let a
    run with two dead rounds read as a complete, honest exploration.
    """
    diag = list(DIAG.DIAGNOSIS_OUTCOMES)
    bad = [s for s in diag if s != "ok"]
    other = len(LLM_STATUS) - len(diag)     # per-query calls (intent planning)
    tail = (f"  ({other} further per-query LLM calls were made, e.g. intent "
            f"planning; those are not diagnosis calls)" if other > 0 else "")
    if not bad:
        return f"Evolution integrity: all {len(diag)} diagnosis calls OK{tail}"
    from collections import Counter
    detail = ", ".join(f"{k}x{v}" for k, v in Counter(bad).most_common())
    return (f"Evolution integrity: {len(bad)}/{len(diag)} diagnosis "
            f"calls FAILED ({detail}) -- the rounds that follow each failure "
            f"tested nothing. Treat this run as {rounds - len(bad)} "
            f"rounds.{tail}")


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "all"],
                    help="evolve on train; check on val; score test once")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE",
                                                         "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("EVOLVE_MODEL", ""))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                    help="EMPTY for a local vLLM; an OpenRouter key when "
                         "--base-url points at openrouter.ai")
    ap.add_argument("--emb-tag", default="m3")
    ap.add_argument("--use-pool", action="store_true",
                    help="restrict retrieval to each query's candidate pool "
                         "(LeCaRD's official protocol); off by default")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="cap queries per round (a round scores every query)")
    ap.add_argument("--max-changes", type=int, default=2,
                    help="step cap: how many RetrievalConfig fields one round "
                         "may move. A recipe counts as 1 slot however many "
                         "fields it carries, so this interacts with --no-recipes")
    ap.add_argument("--recipe-pool", default="legacy8",
                    choices=["legacy8", "ladder"],
                    help="legacy8: the hand-written RECIPES_LEGAL cards. "
                         "ladder: legal_recipes.py, generated as axes x both "
                         "directions so every move has its inverse")
    ap.add_argument("--learned-recipes", action="store_true",
                    help="add the recipes in learned_recipes.json, which the "
                         "LLM proposed and train-dev admitted (learn_recipes.py)")
    ap.add_argument("--exclude-recipe-dev", action="store_true",
                    help="drop the train-dev 20%% from the fitness set. Use it "
                         "with --learned-recipes: otherwise the fold that "
                         "admitted a recipe is also the fold it is scored on")
    ap.add_argument("--no-recipes", action="store_true",
                    help="empty the recipe pool, so the LLM must assemble the "
                         "same configuration out of raw fields and pay one "
                         "step-cap slot per field")
    ap.add_argument("--out", default="evolution_results")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed forwarded to the diagnosis LLM; -1 disables")
    ap.add_argument("--legal-class", default=None,
                    help="evolve on one Axis A legal class only (e.g. "
                         "A5_재산). Default: all classes, one global config")
    ap.add_argument("--allow-tokenizer-fallback", action="store_true",
                    help="accept the regex tokenizer when mecab/spaCy fail to "
                         "import; off by default so a broken dependency stops "
                         "the run instead of silently changing every score")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from the last completed "
                         "round (reads <out>/<dataset>/resume_state.json)")
    a = ap.parse_args()

    # Without a handler the engine's per-round logger.info and the
    # logger.error that names a dead diagnosis (METHODOLOGY 13) reach nothing,
    # so a failed round was invisible in the log as well as in the record.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s %(message)s")
    # One line per HTTP call buried the per-round progress once intent
    # planning started issuing thousands of them. Naming the loggers does not
    # work: the client here logs under "httpx2", not "httpx", and the logger
    # does not exist yet when main() runs, so setLevel on a guessed name is
    # both wrong and too early. Filter on the handler by name prefix instead.
    _NOISY = ("httpx", "httpcore", "openai", "urllib3")

    class _DropHTTPChatter(logging.Filter):
        def filter(self, record):
            return not (record.name.startswith(_NOISY)
                        and record.levelno < logging.WARNING)

    for h in logging.getLogger().handlers:
        h.addFilter(_DropHTTPChatter())
    L.ALLOW_TOKENIZER_FALLBACK = a.allow_tokenizer_fallback
    cases = json.load(open(f"{CASE_DIR}/{a.dataset}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{a.dataset}_queries.json", encoding="utf-8"))
    t0 = time.time()
    index = L.build_index(a.dataset, cases, qs, emb_tag=a.emb_tag)
    print(f"index: {len(cases)} docs, {time.time() - t0:.1f}s", flush=True)

    qa_pairs = build_qa_pairs(a.dataset, a.split, use_pool=a.use_pool,
                              legal_class=a.legal_class)
    if a.legal_class is not None and not qa_pairs:
        raise SystemExit(f"{a.dataset}: no {a.split} query in class "
                         f"{a.legal_class!r}")
    if a.max_queries:
        qa_pairs = qa_pairs[:a.max_queries]
    if a.exclude_recipe_dev:
        import recipe_learning
        before = len(qa_pairs)
        qa_pairs = [p for p in qa_pairs
                    if not recipe_learning.dev_bucket(_qid_of(p))]
        print(f"fitness set: {before} -> {len(qa_pairs)} "
              f"(train-dev held out for recipe admission)", flush=True)
    print(f"{len(qa_pairs)} queries, split={a.split}", flush=True)
    _install_seams(index, qa_pairs)
    _install_diagnosis_context(qa_pairs)

    register_adapter("legal_retrieval", LegalRetrievalAdapter)
    cfg = EvolutionConfig(
        max_rounds=a.rounds,
        initial_retrieval_config=weak_config(),   # same theta_0 as the baseline
        cache_dir=f"{a.out}/{a.dataset}/cache",
        results_dir=f"{a.out}/{a.dataset}",
        resume=a.resume,
        max_changes_per_round=a.max_changes,
    )
    if not a.no_recipes:
        # The hand-written lifts came from TEST. Overwrite them with the
        # train measurements for THIS dataset before the pool is ever
        # rendered into a prompt (recipe_learning.apply_measured_lifts).
        import recipe_learning as _RL
        n_set, n_zero = _RL.apply_measured_lifts(a.dataset)
        if n_set:
            print(f"recipe lifts: {n_set} card(s) re-measured on "
                  f"{a.dataset} train, {n_zero} with CI including 0 "
                  f"(no gain claimed)", flush=True)
        else:
            print("recipe lifts: WARNING no recipe_lifts.json for "
                  f"{a.dataset} -- the pool still carries the "
                  "TEST-derived numbers", flush=True)

    if a.recipe_pool == "ladder" and not a.no_recipes:
        import legal_recipes
        print(f"recipes: ladder pool, {legal_recipes.install()} cards "
              f"(axes x both directions)", flush=True)
    if a.learned_recipes and not a.no_recipes:
        import recipe_learning
        print(f"recipes: +{recipe_learning.install()} learned "
              f"(LLM-proposed, train-dev admitted)", flush=True)
    if a.no_recipes:
        # Emptying the pool disables both paths the LLM can reach a recipe
        # by: the trigger match and `use_recipe` by name.
        from evolvemem import evolution_cookbook as CB
        CB.RECIPE_POOLS["legal_retrieval"] = []
        print("recipes: pool emptied (--no-recipes)", flush=True)
    print(f"step cap: {a.max_changes} field(s)/round", flush=True)
    engine = LegalEvolutionEngine(
        make_llm_call(a.base_url, a.model, api_key=a.api_key,
                      seed=None if a.seed < 0 else a.seed), embedder=None,
        config=cfg, adapter=LegalRetrievalAdapter(), index=index,
    )
    result = engine.evolve(sessions=[], qa_pairs=qa_pairs, initial_memories=cases)

    print("\n" + result.trajectory())
    print(llm_integrity_report(a.rounds))
    path = f"{a.out}/{a.dataset}/best_config.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(result.final_config, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"best config -> {path}")


if __name__ == "__main__":
    main()
