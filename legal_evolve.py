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
import legal_retriever as L               # patches analyze_query on import
import registry as REG                    # levers + Bundle/Constraint/Rule
import evolvemem.diagnosis as DIAG
import evolvemem.evolution as EV
from evolvemem.benchmarks import BenchmarkAdapter, BenchmarkSample, register_adapter
from evolvemem.evolution import EvolutionConfig, EvolutionEngine
from evolvemem.multi_retriever import RetrievalConfig
from evaluate import CASE_DIR, QRELS, load_pool, EVAL_K, weak_config, in_split
from legal_meta import LANG_OF           # dataset -> corpus language

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
                     "qtype": q.get("legal_class", ""),
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
        self._install_bundle_cap_seam()

    def _install_bundle_cap_seam(self) -> None:
        """A Bundle widens this round's step cap instead of spending a slot.

        The vendor path for an invoked recipe does two things: it merges the
        recipe's `proposal` and it charges one slot for the whole card. A
        Bundle has no proposal -- that is the point of the type -- so under
        the vendor rule it would cost a slot and set nothing.

        What a Bundle means is narrower: *these fields are not independent,
        so moving them together must not cost one slot each*. So the seam
        adds, to this round's cap, the number of BUNDLE fields the LLM
        actually proposed. Fields outside the bundle keep competing for the
        original cap, which is what stops an invoked bundle from becoming a
        free-for-all round that no later diagnosis can attribute.

        Wrapping the bound method on our own diagnostics object rather than
        editing `evolvemem.diagnosis`: same seam policy as the rest of this
        module (METHODOLOGY 2 "seam"). `invoked_recipe` is cleared only for
        the duration of the vendor call -- evolution.py records it in the
        round history afterwards, and a bundle that vanished from the record
        would make the run unreadable.
        """
        inner = self.diagnostics.suggest_config_update

        def wrapper(report, current_config, benchmark="locomo",
                    max_changes=None):
            name = getattr(report, "invoked_recipe", None)
            bundle = REG.bundle_by_name(name)
            if bundle is None:
                return inner(report, current_config, benchmark=benchmark,
                             max_changes=max_changes)
            proposed = set(report.parameter_suggestions or {})
            free = proposed & set(bundle.fields)
            widened = (None if max_changes is None
                       else max_changes + len(free))
            print(f"  bundle {name}: cap {max_changes} -> {widened} "
                  f"(free: {', '.join(sorted(free)) or 'none proposed'})",
                  flush=True)
            try:
                report.invoked_recipe = None   # no slot for a card that sets nothing
                return inner(report, current_config, benchmark=benchmark,
                             max_changes=widened)
            finally:
                report.invoked_recipe = name   # keep it in the round history

        self.diagnostics.suggest_config_update = wrapper

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
# Step-cap contention order, and the order the prompt states. ONE object for
# both: `DIAG.set_impact_tiers` makes the engine sort proposals by it when the
# cap truncates, and `diagnosis_prompts.render_priority` renders the PRIORITY
# ORDER section from it. Written out twice, the stated priority and the actual
# truncation order would drift and only the engine's would be real.
IMPACT_TIERS = {
    "fusion_mode": 1, "semantic_top_k": 1, "weight_semantic": 1,
    "structured_top_k": 2, "enable_chapter_filter": 2,
    "enable_entity_swap": 2, "entity_swap_keyword_k": 2,
    "keyword_top_k": 3, "weight_keyword": 3, "weight_structured": 3,
    "enable_query_rewrite": 3, "mmr_diversity_weight": 4,
}

INERT_DIMENSIONS = (
    "locomo_cat", "answer_style", "enable_answer_verification",
    "verification_style", "answer_model",
)


def _gold_ids(reference: str) -> set[str]:
    """Gold doc ids out of the reference string build_qa_pairs writes.

    That string is "GOLD id[digest] ; id[digest] ; ...". Parsing it couples
    this to that format, so every caller treats an empty result as "unknown"
    and drops the line rather than reporting a zero it did not measure.
    """
    ref = (reference or "").strip()
    if not ref.startswith("GOLD"):
        return set()
    out = set()
    for part in ref[4:].split(";"):
        tok = part.strip().split("[", 1)[0].strip()
        if tok:
            out.add(tok)
    return out


def _failure_summary(report, results, cat_total, cat_zeros) -> list[str]:
    """Failure summary for a ranking task.

    Replaces the engine's answer-generation summary, whose every line read
    `prediction` as prose (diagnosis.set_failure_summary_fn has the audit of
    what each one reported on a list of document ids). The lines here are
    the two things the rubric asks the LLM to tell apart -- a document that
    never entered the candidate set versus one that was found and outranked
    -- plus the window that distinction is visible within.
    """
    import diagnosis_prompts as _DP
    import statistics as _st
    names = _CAT_NAMES or {}
    lines = []
    if cat_zeros:
        lines.append(_DP.t("fail.zero_by_cat"))
        for cat in sorted(cat_total):
            lines.append(_DP.t("fail.zero_row").format(
                label=names.get(cat, f"class{cat}"),
                zero=cat_zeros.get(cat, 0), total=cat_total[cat]))
    low = sum(1 for r in results if 0 < r.f1 < 0.3)
    if low:
        lines.append(_DP.t("fail.low_band").format(n=low, total=len(results)))

    zeros = [r for r in results if r.f1 == 0]
    if zeros:
        absent = below = unknown = contradiction = 0
        for r in zeros:
            gold = _gold_ids(r.reference)
            if not gold:
                unknown += 1
                continue
            ranked = str(r.prediction or "").split()
            best = next((i for i, d in enumerate(ranked) if d in gold), None)
            if best is None:
                absent += 1
            elif best >= TOP_N:
                below += 1
            else:
                # A relevant document inside the scored window cannot give
                # NDCG@K = 0. Reaching this means `reference` and the qrels
                # disagree -- most likely because _gold_ids is parsing a
                # format build_qa_pairs no longer writes. Counting it as
                # either bucket would publish a number from a broken parse.
                contradiction += 1
        # Claim the split only when it was measurable for EVERY zero query.
        if not unknown and not contradiction:
            lines.append(_DP.t("fail.split").format(
                z=len(zeros), absent=absent, below=below, k=TOP_N))
        elif contradiction:
            print(f"  [diag] failure split suppressed: {contradiction} "
                  f"zero-score quer(ies) have a relevant document inside "
                  f"top-{TOP_N}. _gold_ids may be parsing a stale reference "
                  f"format -- check build_qa_pairs's `answer` field.",
                  flush=True)
    lens = [len(str(r.prediction or "").split()) for r in results]
    if lens:
        lines.append(_DP.t("fail.ranklen").format(
            med=int(_st.median(lens)), lo=min(lens), hi=max(lens)))
    return lines


def _category_breakdown(results, sub_breakdown) -> str:
    """{category_breakdown} for a ranking task.

    The engine keys this block on `"|".join(subcategory)`, which for our
    adapter (`subcategory_keys = ("category",)`) is the bare integer -- so it
    printed "1: mean=0.512 (n=45)" while the failure summary beside it called
    the same class by its chapter name. Both are needed: the integer is the
    key `per_category_overrides` must use, and the name is the only thing
    that says what the class IS. This prints both and says which is which.
    """
    import diagnosis_prompts as _DP
    names = _CAT_NAMES or {}
    rows = []
    for key, vals in sorted(sub_breakdown.items(), key=lambda kv: _cat_key(kv[0])):
        cat = key.split("|")[0]
        rows.append(_DP.t("cat.row").format(
            cat=cat, label=names.get(_as_int(cat), f"class{cat}"),
            metric=PRIMARY, mean=sum(vals) / len(vals), n=len(vals)))
    if not rows:
        return _DP.t("cat.none")
    return "\n".join(rows) + "\n" + _DP.t("cat.key_note")


# How many samples of each kind to show per category. The built-in block
# took 3 zero-score queries per category and nothing else, which on 7-8
# categories is ~24 samples and makes {sample_failures} the largest block in
# the prompt by far -- larger than the template and the recipe block
# together. Adding a second stratum on top of that would have doubled it, so
# the budget is REALLOCATED instead: fewer of each kind, both kinds present.
# A zero-score sample where the relevant document is simply absent looks the
# same every time; an ordering sample shows what outranked it, which is the
# thing the rubric asks the LLM to read.
SAMPLES_PER_CAT = {"zero": 1, "low": 1}
LOW_BAND = 0.3


def _rank_note(r, caps) -> str:
    """Header line: where the relevant documents actually landed.

    This is the per-query form of the recall/ordering split in the failure
    summary. Without it an ordering failure and a total miss render as the
    same thing -- a list of document ids the LLM cannot locate the gold in.
    """
    import diagnosis_prompts as _DP
    names = _CAT_NAMES or {}
    label = names.get(r.category, r.qtype or "?")
    gold = _gold_ids(r.reference)
    ranked = str(r.prediction or "").split()
    hits = [i + 1 for i, d in enumerate(ranked) if d in gold]
    if gold and hits:
        return _DP.t("sample.head_rank").format(
            cat=r.category, label=label, score=r.f1,
            ranks=", ".join(str(h) for h in hits[:5]), n=len(ranked))
    return _DP.t("sample.head_miss").format(
        cat=r.category, label=label, score=r.f1)


def _sample_failures(by_cat, caps, results) -> str:
    """{sample_failures}, sampling BOTH failure kinds.

    The engine samples `zeros` only, so the rubric's second case -- score
    above zero, relevant document retrieved and then outranked -- was asked
    about but never shown. `by_cat` is still the engine's zero grouping; the
    low band is taken from `results`, which the seam now also passes.
    """
    import diagnosis_prompts as _DP
    from collections import defaultdict
    low_by_cat = defaultdict(list)
    for r in results or []:
        if 0 < r.f1 < LOW_BAND:
            low_by_cat[r.category].append(r)

    lines = []
    for cat in sorted(set(by_cat) | set(low_by_cat)):
        picks = (list(by_cat.get(cat, []))[:SAMPLES_PER_CAT["zero"]]
                 + list(low_by_cat.get(cat, []))[:SAMPLES_PER_CAT["low"]])
        for r in picks:
            lines.append(_rank_note(r, caps))
            lines.append(_DP.t("sample.body").format(
                q=r.question[:caps["question"]],
                # The "GOLD " prefix is build_qa_pairs's marker and is what
                # _gold_ids keys on, but the label already says these are the
                # relevant documents, so it is noise in front of the LLM.
                ref=_strip_gold(r.reference)[:caps["field"]]))
            if r.retrieved_sources:
                shown = r.retrieved_sources[:caps["retrieved_shown"]]
                lines.append(_DP.t("sample.retrieved").format(
                    shown=" ; ".join(str(x) for x in shown)))
    if not lines:
        return _DP.t("sample.none")
    return _DP.t("sample.strata") + "\n\n" + "\n".join(lines)


def _strip_gold(reference: str) -> str:
    ref = str(reference or "").strip()
    return ref[5:].lstrip() if ref.startswith("GOLD ") else ref


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _cat_key(key: str):
    """Sort "10" after "9". The engine sorts these keys as strings."""
    v = _as_int(key.split("|")[0])
    return (0, v) if isinstance(v, int) else (1, str(v))


_CAT_NAMES: dict = {}


def _install_diagnosis_context(qa_pairs: list[dict], dataset: str) -> None:
    """Tell the diagnosis LLM what our categories actually are.

    The engine labels categories with LoCoMo's splits (1=SingleHop,
    5=Adversarial, ...). Ours are penal-code chapters,
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
    global _CAT_NAMES
    _CAT_NAMES = names
    # Our own failure summary. The engine's reads `prediction` as free text
    # an answer LLM wrote; ours is a ranked list of document ids, so its
    # abstention line printed 0 every round, its wrong-answer line restated
    # zero_f1_count, and its "paraphrase symptom" line fired on
    # `len(pred.split()) >= 4 and len(ref.split()) <= 3` -- true for most of
    # our queries, reported as a symptom of prose this system never writes.
    DIAG.set_failure_summary_fn(_failure_summary)
    # Same reason, for the other two report-built blocks. The breakdown also
    # gains the chapter names it never had, beside the integer keys that
    # per_category_overrides needs.
    DIAG.set_category_breakdown_fn(_category_breakdown)
    DIAG.set_sample_failures_fn(_sample_failures)
    # And no invented move when the diagnosis fails. Both fallback branches
    # tune max_context, which is not in our action space, and the branch that
    # would fire is gated on `wrong_answer_count > zero_f1_count * 0.6` --
    # always true once abstention_count is structurally 0.
    DIAG.set_heuristic_fallback(False)
    DIAG.set_prompt_exclude(INERT_DIMENSIONS)
    # Drop `first_found` from the values the diagnosis LLM may propose.
    #
    # `set_action_fields` below restricts WHICH fields are proposable; it says
    # nothing about which VALUES. `fusion_mode` is the one field where that
    # gap matters, because its enum is not four levels of one axis: `*_only`
    # gates which views even RUN (multi_retriever.retrieve_multiview checks
    # `fusion_mode not in ("keyword_only", ...)` before each search) and
    # `first_found` is a concatenation in a fixed view order with no sort, so
    # with semantic_top_k >= max_context the first view takes every slot.
    # Measured on LeCaRDv2 test: first_found (semantic 100 + BM25 100) 0.2797,
    # identical to semantic_only, against keyword_only 0.3948. A round that
    # proposes it spends a step-cap slot to measure a different mode's score.
    #
    # The class attribute is overridden rather than the vendor file edited:
    # same seam policy as the rest of this module (METHODOLOGY 2 "seam").
    # `*_only` stays for now -- theta_0 IS `keyword_only`, and whether it is
    # equivalent to rrf with every auxiliary depth at 0 is an open measurement
    # (METHODOLOGY 2, the -0.010 left unexplained).
    _EV = DIAG.MemoryDiagnostics.ENUM_VALUES
    DIAG.MemoryDiagnostics.ENUM_VALUES = {
        **_EV, "fusion_mode": set(_EV["fusion_mode"]) - {"first_found"},
    }
    # Restrict the action space to the fields MEASURED to change the ranking
    # in this pipeline (recipe_learning.EFFECTIVE_FIELDS: 12 of 45; see that
    # module for the probe). Before this, 37% of the field changes the loop
    # made over 16 runs landed on fields that cannot move the metric --
    # `enable_intent_planning` 28 times, `entity_swap_semantic_k` 18,
    # `intent_final_context` 12 -- each one a step-cap slot spent on a round
    # that could not teach anything. The remaining 33 fields stay in
    # RetrievalConfig at their defaults; they are simply not offered.
    import recipe_learning as _RLF
    # Per-dataset, not global. `enable_query_rewrite` reads a precomputed
    # rewrite per query from cache/rewrite/<dataset>.json; where that file
    # does not exist, `legal_retriever.expand_query` returns the query
    # unchanged, so the field cannot move the ranking at all. Only
    # LeCaRDv2.json exists -- the lever is dead on LeCaRD, COLIEE and
    # LEGAR_stricter. Offering a dead field is what section 26 removed 33
    # other fields for: every proposal of it spends a step-cap slot on a
    # round that can only measure +0.0000, and `inert_report` can then only
    # explain the waste after the fact.
    fields = list(_RLF.EFFECTIVE_FIELDS)
    dropped = []
    # The predicate is `load_rewrites(dataset)` itself, not "does the file
    # exist": `expand_query` returns the query unchanged when the dict is
    # empty, so an existing but empty cache/rewrite/<ds>.json is exactly as
    # inert as a missing one and a file-existence proxy would pass it.
    if not L.load_rewrites(dataset):
        if "enable_query_rewrite" in fields:
            fields.remove("enable_query_rewrite")
            dropped.append(f"enable_query_rewrite (no rewrites for {dataset} "
                           f"in {L.REWRITE_DIR})")
    DIAG.set_action_fields(fields)
    print(f"action space: {len(fields)} measured-effective "
          f"field(s) + per_category_overrides "
          f"(of {len(_RLF._FIELD_TYPE)} in RetrievalConfig)", flush=True)
    for d in dropped:
        print(f"  dropped for this dataset: {d}", flush=True)

    # Prompt in the corpus language: Chinese for LeCaRD/LeCaRDv2, Korean for
    # LEGAR_stricter, English for COLIEE. The LLM reads judgments, category
    # names and retrieval snippets in that language; until 2026-11-11 the
    # instructions around them were English on all four, so on three of the
    # four it reasoned across a language boundary between the task and the
    # data. The three variants carry the same sections, the same action
    # space and the same output schema -- `check_prompt_variants()` asserts
    # the placeholder sets match -- so language is the only variable.
    #
    # They also replace the vendored LoCoMo prompt rather than filtering it,
    # so COLIEE's prompt is NOT what ran before either: see the module
    # docstring in diagnosis_prompts.py for what that costs us in
    # comparability against the pre-2026-11-11 baseline.
    import diagnosis_prompts as _DP
    # `fields` first: the prompt must describe the action space this run
    # actually has, not EFFECTIVE_FIELDS in the abstract.
    _lang = _DP.install(dataset, DIAG, fields, IMPACT_TIERS)
    print(f"diagnosis prompt: {_lang} ({dataset})", flush=True)
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
    # Step-cap contention order. The engine's default ranks scalar weights
    # LAST (its tiers come from LoCoMo, where flags were worth +5-10 pp and
    # weights were tweaks). Here the ordering would defeat the foundation
    # card: that card sets fusion_mode and semantic_top_k and tells the LLM
    # to choose weight_semantic in the SAME round, because theta_0 leaves the
    # weight at 1.0 and switching the semantic view on at 1.0 is worse than
    # theta_0 (-0.0059 LeCaRDv2, -0.1067 LeCaRD on train). With one slot left
    # after the card, a tier-1 flag proposed alongside would take it and the
    # weight would be dropped without a word.
    #
    # Measured priority instead: the fields that decide whether the semantic
    # view helps or hurts come first, swap depth next (its only live field),
    # the rest keep the default.
    DIAG.set_impact_tiers(IMPACT_TIERS)
    # Rendered into the prompt once per round, then cleared so the next
    # round counts its own retrievals.
    def _block():
        body = view_report()
        inert = inert_report()
        _reset_view_stats()
        _reset_lever_stats()
        L.reset_semantic_misses()
        import diagnosis_prompts as _DP
        bud = budget_block()
        if bud:
            body += "\n\n" + bud
        if inert:
            body += "\n\n" + _DP.t("inert.heading") + "\n" + inert
        notes = state_notes()
        if notes:
            body += "\n\n" + notes
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
    import diagnosis_prompts as _DP
    REASON = {k: _DP.t("inert." + k) for k in
              ("query_rewrite", "chapter_filter", "view:semantic",
               "view:metadata", "view:party-strip", "fusion_mode")}
    lines = []
    for name, (acted, on) in sorted(_LEVER_STATS.items()):
        if on and acted == 0:
            lines.append(_DP.t("inert.line").format(
                name=name, on=on,
                reason=REASON.get(name, _DP.t("inert.fallback"))))
    return "\n".join(lines)


# Library defaults for the fields whose value decides a view's contribution.
# A config sitting on one of these is not a choice anybody made: theta_0
# inherits them, and nothing in the loop sets them unless the LLM does.
_UNCHOSEN = (
    # field, default, which view it weights, the field that activates it
    ("weight_semantic",   1.0, "state.view_semantic",   "semantic_top_k"),
    # weight_structured REMOVED 2026-11-11. Two reasons, either sufficient:
    #
    # (1) It could not fire. The note needs structured_top_k > 0, theta_0
    #     sets it to 0, and the only surviving card that touches the field
    #     sets it to 0 as well. Nothing in the pool raises it.
    # (2) In the one state where it COULD fire -- a bundled round accepted
    #     for another field's sake, leaving the view on -- the note points at
    #     tuning the weight of a view that measured negative on all four
    #     datasets, shallow and under RRF included: RRF+sem+str10 is BELOW
    #     RRF+sem on every one of them (METHODOLOGY 8). The right move from
    #     that state is to switch the view off, which undo_structured_view
    #     already offers. The note would have competed with it for the round.
    #
    # The field stays in the action space. That is not a contradiction: a
    # view whose sign flips between classes of ONE dataset (section 15: the
    # semantic view, +0.0119 on A2 and -0.3421 on A3 of LeCaRDv2) cannot be
    # ruled out per-category from a global measurement, and per-category is
    # still unmeasured. What is removed is the nudge, not the option.
    #
    # entity_swap_keyword_k STAYS, and it is now MEASURED (measure.py grid --grid
    # swap-depth, train, base rrf+sem): off / 10 / 30 / 100 gives
    #   LeCaRDv2 +0.0140 / +0.0125 / +0.0118 / +0.0161
    #   LeCaRD   +0.0134 / +0.0139 / +0.0172 / +0.0191
    #   COLIEE   +0.0041 / +0.0041 / +0.0030 / +0.0036
    #   LEGAR    +0.0062 / +0.0057 / +0.0065 / +0.0064
    # Every earlier "party-strip is dead / harmful" verdict was taken at
    # depth 30, and at 100 two datasets come out above the off state. The
    # deltas are still inside their noise floors, so the honest reading is
    # that the depth-30 verdict is VOID and 100 is neutral -- not that the
    # view pays. The default here is 100 like every other view's depth.
    ("entity_swap_keyword_k", 100, "state.view_swap",   None),
)

_LAST_CFG = None

# Set once per run so the prompt can state the two constraints the LLM was
# being judged against without being told: what a round has to clear, and
# how many rounds remain. Both were decided outside the prompt and never
# shown -- the LLM was asked to propose moves while unable to tell whether a
# move it believed in was big enough to survive, or whether it had six
# rounds left to explore or one round to consolidate.
_BUDGET_INFO = {"threshold": None, "total_rounds": None, "round": 0}


def set_budget_info(threshold: float, total_rounds: int) -> None:
    _BUDGET_INFO.update(threshold=threshold, total_rounds=total_rounds,
                        round=0)


def budget_block() -> str:
    """Prompt block naming the acceptance threshold and the rounds left.

    The round number is counted here rather than read from the engine: this
    is called exactly once per diagnosis, which is once per round. If that
    ever stops being true the count drifts, so it is reported in the block
    itself where a wrong number is visible rather than silent.
    """
    import diagnosis_prompts as _DP
    thr, total = _BUDGET_INFO["threshold"], _BUDGET_INFO["total_rounds"]
    if thr is None or not total:
        return ""
    _BUDGET_INFO["round"] += 1
    r = _BUDGET_INFO["round"]
    return _DP.t("budget.line").format(
        thr=thr, thr_pp=thr * 100, round=r, total=total,
        left=max(0, total - r))


def state_notes() -> str:
    """Prompt block: fields left at a default while the view they weight is on.

    This is what is left of four recipes removed on 2026-11-11
    (undo_semantic_overweight and friends). Their proposals were pure scalar
    magnitudes on fields already in the action space, so the card carried no
    structure and `undo_semantic_overweight`'s 0.3 was LeCaRDv2's exact train
    optimum -- the answer key, handed over. But their TRIGGERS were doing real
    work: they named the round in which the symptom was present.

    So the trigger survives and the number does not. The note says what state
    the config is in and that nobody chose it. It gives no value and no
    DIRECTION, because the removed cards could only ever propose going down
    and on LEGAR_stricter the semantic weight's optimum is above the default --
    one dataset in four where the card pointed the wrong way.
    """
    import diagnosis_prompts as _DP
    cfg = _LAST_CFG
    if cfg is None:
        return ""
    lines = []
    for field, default, view_key, gate in _UNCHOSEN:
        val = getattr(cfg, field, None)
        if val is None or val != default:
            continue
        if gate is not None:
            # Which fusion modes a weight actually reaches. NOT just
            # weighted_sum: multi_retriever._accumulate multiplies every
            # contribution by the view's weight, and `rrf` goes through
            # _accumulate too -- what RRF replaces is the SCORE (1/(60+rank)
            # instead of the normalised score), not the weighting. The four
            # cards removed today all carried "UNDER WEIGHTED-SUM FUSION
            # ONLY" in their descriptions and their triggers, and that was
            # wrong: under rrf the weight still decides the ranking, so the
            # note has to fire there as well. `first_found` and the
            # `*_only` modes never call _accumulate, so weights are inert
            # there and the note would be the false symptom it exists to
            # avoid.
            if getattr(cfg, "fusion_mode", "") not in ("weighted_sum", "rrf"):
                continue
            if getattr(cfg, gate, 0) <= 0:
                continue
        else:
            if not getattr(cfg, "enable_entity_swap", False):
                continue
        lines.append(_DP.t("state.unchosen").format(
            field=field, value=val, view=_DP.t(view_key)))
    if not lines:
        return ""
    return _DP.t("state.heading") + "\n" + "\n".join(lines)


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
    import diagnosis_prompts as _DP
    lines = [_DP.t("view.header").format(n=n, k=TOP_N), ""]
    c = _DP.t("view.cols")
    lines.append("  " + _DP.pad(c[0], 10) + " " + _DP.pad(c[1], 8, True)
                 + " " + _DP.pad(c[2], 7, True) + " " + _DP.pad(c[3], 9, True)
                 + " " + _DP.pad(c[4], 6, True))
    for v in VIEW_ORDER:
        if v not in on:
            lines.append(f"  {v:10s} {'OFF':>8s} {'-':>7s} {'-':>9s} {'-':>6s}")
            continue
        st = stats.get(v) or {"hits": 0, "gold": 0, "sole_gold": 0}
        lines.append(f"  {v:10s} {'on':>8s} {100*st['hits']/total_hits:6.1f}% "
                     f"{st['gold']:9d} {st['sole_gold']:6d}")
    lines.append("")
    lines.append(_DP.t("view.off_note"))
    lines.append(_DP.t("view.sole_note"))
    # A semantic view that is ON and empty because the query text changed. No
    # other block can say this: `inert_report` watches levers (the rewrite
    # lever DID act) and the table above would just show a 0% share.
    sm = L.semantic_misses()
    if sm["miss"]:
        lines.append(_DP.t("view.sem_miss").format(
            miss=sm["miss"], total=sm["miss"] + sm["hit"],
            sample=sm["sample"] or "?"))
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
            global _LAST_CFG
            _LAST_CFG = cfg
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
    ap.add_argument("--acceptance-threshold", default="auto",
                    help="delta a round must beat the incumbent best by. "
                         "'auto' (default) uses the measured noise floor for "
                         "this dataset (recipe_learning.noise_floor); a float "
                         "pins it; 'engine' keeps the engine's 0.003, which "
                         "is a LoCoMo-era default BELOW our noise floor on "
                         "three of four datasets -- use it only to reproduce "
                         "a pre-2026-11-11 run")
    ap.add_argument("--update-rule", default="elitist",
                    choices=["elitist", "paper"],
                    help="elitist (default, this repo's): keep a round only "
                         "if it beats the incumbent BEST by more than the "
                         "acceptance threshold. paper (EvolveMem Eq. 4): "
                         "revert only when the score DROPS from the PREVIOUS "
                         "round by more than --tau-rev; a flat or slightly "
                         "worse round is kept and the search continues from "
                         "it. Eq. 4's random-perturbation branch is not "
                         "implemented -- the anti-plateau recipe rule stands "
                         "in for it")
    ap.add_argument("--tau-rev", type=float, default=0.01,
                    help="regression tolerance for --update-rule paper; "
                         "0.01 is the value the paper's case study reports")
    ap.add_argument("--max-changes", type=int, default=2,
                    help="step cap: how many RetrievalConfig fields one round "
                         "may move. A recipe counts as 1 slot however many "
                         "fields it carries, so this interacts with --recipes")
    # One flag for the four arms, because the two booleans it replaces could
    # not express the one this project now needs: `--learned-recipes` was
    # guarded by `and not a.no_recipes`, so "learned cards only" was
    # unreachable -- the hand-written pool always came along.
    ap.add_argument("--recipes", default="hand",
                    # "bundles" is the third type's arm: cards that carry no
                    # values and only widen the cap (METHODOLOGY 3).
                    choices=["hand", "learned", "both", "none", "bundles"],
                    help="which recipe pool the loop may use. hand: the "
                         "written RECIPES_LEGAL cards. learned: only the "
                         "LLM-proposed, train-dev-admitted cards for THIS "
                         "dataset (learn_recipes.py). both: hand + learned. "
                         "bundles: recipes.BUNDLES, cards that set NOTHING "
                         "and only widen this round's step cap over fields "
                         "that are not independent. none: empty pool, so the "
                         "LLM must assemble the same configuration out of raw "
                         "fields and pay one step-cap slot per field")
    ap.add_argument("--exclude-recipe-dev", action="store_true",
                    help="drop the train-dev 20%% from the fitness set. Use it "
                         "with --recipes learned/both: otherwise the fold that "
                         "admitted a recipe is also the fold it is scored on")

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

    # Acceptance threshold. The engine's default 0.003 is an absolute delta
    # from LoCoMo's F1 scale; on our folds it sits BELOW the noise floor on
    # LeCaRD (7.7x), LeCaRDv2 and COLIEE (1.7x each), so the loop was
    # accepting rounds it could not distinguish from doing nothing and
    # elitism then carried them forward as the incumbent best.
    import recipe_learning as _RLT
    if a.acceptance_threshold == "auto":
        accept_thr, thr_why = _RLT.noise_floor(a.dataset)
        if accept_thr <= 0:
            print(f"!! acceptance threshold: {thr_why}. Cannot derive a "
                  f"noise floor, so the run would fall back to the engine's "
                  f"0.003 -- a number measured on a different benchmark. "
                  f"Run measure.py floor --datasets {a.dataset} first, or "
                  f"pass --acceptance-threshold explicitly.", flush=True)
            sys.exit(2)
    elif a.acceptance_threshold == "engine":
        accept_thr, thr_why = 0.003, "engine default (LoCoMo-era, unverified here)"
    else:
        accept_thr, thr_why = float(a.acceptance_threshold), "set on the command line"
    print(f"acceptance threshold: {accept_thr:.5f} -- {thr_why}", flush=True)
    if a.split == "train":
        # Said out loud because it decides what the run can possibly show.
        print(f"  a round must beat the incumbent best by more than "
              f"{accept_thr * 100:.2f} pp to be accepted", flush=True)
    _install_seams(index, qa_pairs)
    _install_diagnosis_context(qa_pairs, a.dataset)

    register_adapter("legal_retrieval", LegalRetrievalAdapter)
    cfg = EvolutionConfig(
        max_rounds=a.rounds,
        initial_retrieval_config=weak_config(),   # same theta_0 as the baseline
        cache_dir=f"{a.out}/{a.dataset}/cache",
        results_dir=f"{a.out}/{a.dataset}",
        resume=a.resume,
        max_changes_per_round=a.max_changes,
        acceptance_threshold=accept_thr,
        update_rule=a.update_rule,
        tau_rev=a.tau_rev,
    )
    # No lift figures are written into the pool any more. `apply_measured_lifts`
    # existed to overwrite hand-typed lifts (which came from TEST) with train
    # measurements, and it matched cards by name -- but after the type split
    # the only measurable thing is a Rule, whose names do not intersect the
    # hand pool's at all, so the function could never match. The renderer's
    # `set_show_unmeasured_lift(False)` already does what it was protecting:
    # a card nobody measured on this dataset prints no number.

    # Order matters: empty first, then add, or "learned" would keep the hand
    # pool underneath it.
    from evolvemem import evolution_cookbook as CB
    if a.recipes == "bundles":
        # Bundles replace the hand pool rather than joining it: the two are
        # different types answering the same prompt slot, and mixing them
        # makes "did the bundle help" unanswerable (METHODOLOGY 3, the 2x2).
        CB.RECIPE_POOLS["legal_retrieval"] = REG.as_vendor_shims()
        print(f"recipes: {len(CB.RECIPE_POOLS['legal_retrieval'])} bundle "
              f"shim(s) -- they set nothing and widen the step cap",
              flush=True)
    if a.recipes in ("learned", "none"):
        # Emptying the pool disables both paths the LLM can reach a recipe
        # by: the trigger match and `use_recipe` by name.
        CB.RECIPE_POOLS["legal_retrieval"] = []
        print(f"recipes: hand-written pool emptied (--recipes {a.recipes})",
              flush=True)
    if a.recipes in ("learned", "both"):
        import recipe_learning
        n = recipe_learning.install(dataset=a.dataset)
        print(f"recipes: +{n} learned (LLM-proposed, train-dev admitted "
              f"on {a.dataset})", flush=True)
        if not n:
            print(f"  none admitted for {a.dataset} yet -- run "
                  f"`learn_recipes.py --dataset {a.dataset}` first",
                  flush=True)
    # Card descriptions in the corpus language, AFTER every pool mutation
    # above, because this replaces the pool list. The recipe block is the
    # largest static text in the prompt -- larger than the template itself --
    # so without this a "Chinese prompt" was 57% English by character, and
    # the English part was the half that says what each move does.
    if a.recipes != "none":
        import diagnosis_prompts as _DP2
        from evolvemem import evolution_cookbook as _CB
        _gone = _DP2.drop_dead_cards(_CB, a.dataset)
        if _gone:
            print(f"recipes: dropped {len(_gone)} card(s) that cannot act on "
                  f"{a.dataset}: {', '.join(_gone)}", flush=True)
        print(_DP2.install_recipe_text(LANG_OF[a.dataset], _CB), flush=True)
    set_budget_info(accept_thr, a.rounds)
    print(f"step cap: {a.max_changes} field(s)/round  |  update rule: "
          f"{a.update_rule}" + (f" (tau_rev={a.tau_rev})"
                                if a.update_rule == "paper" else ""), flush=True)
    engine = LegalEvolutionEngine(
        make_llm_call(a.base_url, a.model, api_key=a.api_key,
                      seed=None if a.seed < 0 else a.seed), embedder=None,
        config=cfg, adapter=LegalRetrievalAdapter(), index=index,
    )
    # Silence the meta-analyzer's LLM call. Phase 7b runs
    # MetaEvolutionAnalyzer.analyze() EVERY round, and analyze() issues a
    # second LLM call (meta_analysis.META_PROMPT, temperature 0.2) whose
    # output elitist mode then discards entirely: evolution.py gates the
    # revert on `not self.config.elitist`, and both proposal merges on
    # `not self.config.elitist` as well. Ours is elitist (the engine default
    # and the only mode our acceptance rule is defined for), so the call was
    # pure cost -- it doubled the per-round LLM count that METHODOLOGY
    # reports as 1, and it was the one call in the loop still reading
    # LoCoMo's "self-evolving memory system" framing at a non-zero
    # temperature.
    #
    # Setting llm_call to None is the documented off switch: analyze() guards
    # the call with `if self.llm_call is not None`, so its deterministic
    # stagnation/worst-subcategory logic still runs and still produces a
    # MetaPlan -- which elitist ignores exactly as before. Nothing about the
    # trajectory changes; one call per round stops being made.
    engine.meta_analyzer.llm_call = None
    print("meta-analyzer: LLM call disabled (elitist discards its plan); "
          "1 diagnosis call per round", flush=True)
    result = engine.evolve(sessions=[], qa_pairs=qa_pairs, initial_memories=cases)

    sm = L.semantic_misses()
    if sm["miss"]:
        print(f"!! semantic view integrity: {sm['miss']} of "
              f"{sm['miss'] + sm['hit']} semantic searches in the LAST round "
              f"used a MODIFIED query text and returned nothing (no "
              f"precomputed vector). enable_query_rewrite and the "
              f"party-strip view both modify it. Any semantic-view number in "
              f"this run was measured with that view partly dark "
              f"(JOURNAL 43).", flush=True)
    print("\n" + result.trajectory())
    print(llm_integrity_report(a.rounds))
    path = f"{a.out}/{a.dataset}/best_config.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(result.final_config, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"best config -> {path}")


if __name__ == "__main__":
    main()
