"""The registry: what a lever is, and the three kinds of recipe.

METHODOLOGY 3. One module because the two halves share one invariant -- a
Rule declares the lever coordinate it creates, and that coordinate has to
exist here. Split across two files that invariant was a convention; together
it is a lookup.

Part 1, LEVERS: every retrieval lever, declared by its coordinates. Five
separate-looking faults found on 2026-11-16 -- `fusion_mode` was not an axis,
`first_found` hid in a vendor default, `Recipe.proposal` had the same type as
the diagnosis LLM's `parameter_suggestions`, query-rewrite turned out to be
party-strip by another implementation, and the fusion delta came from a
single slice -- are one defect: nothing declared what a lever IS, so a
non-axis could enter as an axis and the same operation could enter twice.

A lever is a cell of a grid, not a name:

    (input, scorer) + combine

    input   -- what the query is turned into before searching
    scorer  -- what produces the score
    combine -- how the result joins the ranking: a separate candidate list
               fused with the others ("view"), or merged into the query text
               itself ("in-place"). RECORDED, NOT IDENTIFYING: see
               `coordinate()`.

Same (input, scorer) == same lever, whatever it is called and however it is
implemented. A regex and an LLM that both delete party names are ONE lever
(R2); that is how `enable_query_rewrite` and `enable_entity_swap` turned out
to be the same move, which section 5 had already measured (corr +0.524,
Jaccard 0.463) without drawing the conclusion.

Part 2, the recipe types: the vendor's `Recipe` is one dataclass carrying
`proposal: dict`, the same type the diagnosis LLM already emits as
`parameter_suggestions`. Two channels with the same type, the same budget and
the same prompt menu are not two layers. And that one dataclass was being
asked to carry three different jobs:

    (2)  what value a field takes            -> the action space. Stays a dict.
    (3)  which fields have to move TOGETHER  -> Bundle. An edge, not a value.
    (3') how the query is rewritten          -> Rule. Text. No field exists.

The third has no home in RetrievalConfig at all, so it was flattened into the
integer `query_rewrite_repeat` -- and it is exactly what the comparison paper
("When Rules Learn", Tao et al. 2026) evolves: a rewriting rule, seeded with
one human-written rule.

Why the split is in the types and not in a convention: a Bundle that cannot
hold a value cannot hand the loop a dataset's own optimum (JOURNAL 33 found
`0.3`, LeCaRDv2's train optimum, sitting inside a card and pointing the wrong
way for LEGAR). It is not a rule someone has to remember; it is a field that
does not exist.

Admission differs per type, which is why one `admit()` could not serve both:

    Bundle -- 2x2 interaction term != 0. Defined only because a bundle has
              two or more fields to factor.
    Rule   -- the effect itself (paired bootstrap CI lower bound > 0, mean >
              that dataset's measured noise floor, endpoint > theta_0). A
              rule has no fields, so no interaction is definable.

Nothing in part 1 runs during retrieval; it is data, so verify.py can check
the rules mechanically instead of a human re-reading a markdown table.
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ── Coordinate vocabularies ────────────────────────────────────────────────
# Closed sets on purpose. A new value is a claim that the system gained an
# axis, and that claim should be visible in a diff rather than arriving as a
# free-form string.
# There is deliberately no "strip+compress" value. Query-rewrite deletes the
# same spans party-strip deletes and then restates what is left in legal
# terms; giving the pair of operations its own coordinate would let a lever
# escape R2 by naming its side effect, and the compression half has never been
# measured apart from the deletion. It stays `strip` -- colliding, waived and
# printed on every gate run -- until someone measures compression alone.
INPUTS = ("original", "strip", "expand")
SCORERS = ("bm25", "embedding", "metadata")
COMBINES = ("view", "in-place")

# Lifecycle. `vacant` rows carry no implementation: they are cells of the grid
# that nothing fills, kept here because an empty cell is a finding (the
# expansion row below), not an omission.
STATUSES = ("active", "retired", "vacant")


@dataclass(frozen=True)
class Lever:
    name: str
    input: str
    scorer: str
    combine: str
    status: str
    # RetrievalConfig fields this lever OWNS. A field is owned once; see
    # `shares` for the one field two levers read.
    fields: tuple[str, ...] = ()
    # Fields this lever reads but does not own. `weight_semantic` is here for
    # party-strip because multi_retriever.retrieve_multiview hardcodes
    # `"swap": cfg.weight_semantic` -- the strip view has NO weight of its
    # own, so every depth measurement for it (JOURNAL 47) was taken with its
    # weight tied to the semantic view's.
    shares: tuple[str, ...] = ()
    # Non-empty: this lever duplicates another lever's coordinate and we know
    # it. The gate reports it instead of failing, because the resolution is a
    # decision about what the lever should BE, not a typo to fix. An empty
    # waiver on a duplicate coordinate is a hard failure.
    coord_waiver: str = ""
    note: str = ""


LEVERS: tuple[Lever, ...] = (
    Lever(
        name="keyword", input="original", scorer="bm25", combine="view",
        status="active",
        fields=("keyword_top_k", "weight_keyword"),
        note="theta_0 and the strongest single view in this domain "
             "(LeCaRDv2 test 0.3948 vs semantic 0.2797). Never swept: both "
             "fields sit at 100/1.0 on no evidence (JOURNAL 25).",
    ),
    Lever(
        name="semantic", input="original", scorer="embedding", combine="view",
        status="active",
        fields=("semantic_top_k", "weight_semantic"),
        note="The only consistently positive lever across all four datasets. "
             "Optimal weight spans 0.1~2.0 -- a 20x spread -- which is the "
             "per-dataset value the loop is supposed to reach on its own. "
             "Its input is pinned to the row's original text "
             "(SEMANTIC_USES_ROW_TEXT), so no query transform reaches it.",
    ),
    Lever(
        name="metadata", input="original", scorer="metadata", combine="view",
        status="active",
        fields=("structured_top_k", "weight_structured"),
        note="Scores overlap of `legal_refs`. Harmful alone, contributing as "
             "a complement (JOURNAL 49). NOTE: a query's legal_refs include "
             "the annotated `Crime` field (build_cases.build_queries), not "
             "only terms extracted from the query text -- so this lever reads "
             "an annotation, which is a leakage question nobody has written "
             "down. METHODOLOGY 3.",
    ),
    Lever(
        name="party-strip", input="strip", scorer="bm25", combine="view",
        status="active",
        fields=("enable_entity_swap", "entity_swap_keyword_k"),
        shares=("weight_semantic",),
        note="Deletes person/org/place spans and searches the remainder. "
             "Neutral at depth 100; the earlier 'dead' verdict was taken at "
             "depth 30, an old-corpus leftover (R3, JOURNAL 47).",
    ),
    Lever(
        name="rewrite", input="strip", scorer="bm25", combine="in-place",
        status="active",
        fields=("enable_query_rewrite",),
        coord_waiver=(
            "R2: same deletion as party-strip. query_rewrite.PROMPTS line 1 "
            "is literally the strip list (names, places, plate numbers, "
            "dates, amounts); what remains that party-strip cannot do is the "
            "compression into legal terms, and that has never been measured "
            "apart from the deletion. §5 measured the overlap (corr +0.524, "
            "Jaccard 0.463, 90 queries harmed by both) and kept both levers. "
            "Resolution is a decision, not a fix: either retire this lever, "
            "or move its input coordinate to `expand` by deleting the "
            "deletion instruction from the prompt. Until then the +0.0009 "
            "CI[-0.0021,+0.0037] is a joint measurement, not this lever's."),
        note="Appended to the original query, so with a deletion-style "
             "rewrite it can introduce no term the original lacks: it is a "
             "term-frequency reweighting of a subset. Cached for LeCaRDv2 "
             "only; dropped from the action space elsewhere.",
    ),
    Lever(
        name="strip-semantic", input="strip", scorer="embedding",
        combine="view",
        status="retired",
        note="entity_swap_semantic_k pinned to 0. Retired because the main "
             "semantic view reads the same original text, so this counted one "
             "ranking twice (JOURNAL 44). Second reason found in code on "
             "2026-11-16: the swap view is built as `sw_sem + sw_kw` with no "
             "sort, so under RRF the BM25 half starts at rank "
             "entity_swap_semantic_k -- switching this on silently penalises "
             "the lexical half it exists to serve.",
    ),
    Lever(
        name="legal-element expansion", input="expand", scorer="bm25",
        combine="view",
        status="vacant",
        note="EMPTY CELL, and the largest finding of the registry. Both "
             "query-transforming levers DELETE; nothing adds. BM25 can only "
             "match terms present in the document, so deletion alone cannot "
             "raise recall -- which is why rewrite measures +0.0009 CI∋0. "
             "'When Rules Learn' (Tao et al. 2026) rewrites by EXPANSION "
             "(synonyms, terminology variants) and concatenates, which is why "
             "concatenation works there. Filling this cell must use terms "
             "taken from the query TEXT: expanding from the annotated "
             "`Crime` field would both double-count the metadata lever and "
             "raise the leakage question noted there.",
    ),
)


# Proposable fields that are not levers. They configure the combine stage or
# run after fusion, so they have no (input, scorer) coordinate -- listing them
# here is what lets the gate demand that EVERY proposable field is accounted
# for exactly once. Three of twelve being non-levers is itself worth seeing.
NON_LEVER_FIELDS: dict[str, str] = {
    "fusion_mode": (
        "The combine axis itself, not a lever. Its enum held three different "
        "kinds of operation, so the values were not levels of one axis (R1, "
        "METHODOLOGY 2). `first_found` -- concatenation in a fixed view order "
        "with no sort -- is REMOVED (R7, four places plus the gate). "
        "`*_only` is still accepted and is a separate question from fusion: "
        "it gates which views even RUN, so it belongs to the view-mask axis, "
        "and whether it duplicates a depth-0 mask (R6) is settled by "
        "comparing run files, not by scoring."),
    "enable_chapter_filter": (
        "A candidate-set filter applied before scoring, not a view. Measured "
        "harmful (-8.3 pp); the default was already False, which JOURNAL 48 "
        "misread as a finding (R3)."),
    "mmr_diversity_weight": (
        "Post-fusion diversity rerank. Operates on the fused ranking, so it "
        "has no scorer of its own. Unexplored."),
}


def coordinate(lever: Lever) -> tuple[str, str]:
    """Identity is (input, scorer). `combine` is recorded, not identifying.

    Two levers that feed the same scorer with the same transformed query
    carry the same information; whether one fuses a separate candidate list
    and the other merges terms into the query is HOW that information enters,
    which is the "implementation is not a coordinate" clause of R2. Including
    `combine` in the identity would have let party-strip (strip, bm25, view)
    and rewrite (strip, bm25, in-place) pass as distinct levers -- the exact
    pair the rule exists to catch.
    """
    return (lever.input, lever.scorer)


def coordinates() -> set[tuple[str, str]]:
    """Every coordinate the registry knows, vacant cells included.

    A Rule (recipes.Rule) declares the coordinate it creates; a rule pointing
    at a cell nobody registered would widen tier (1) without the registry
    recording it, which is the one thing the rule type is allowed to do and
    therefore the one thing that has to stay visible.
    """
    return {coordinate(l) for l in LEVERS}


def active() -> tuple[Lever, ...]:
    return tuple(l for l in LEVERS if l.status == "active")


def owned_fields() -> dict[str, str]:
    """field -> owning lever name, for levers that own fields."""
    out: dict[str, str] = {}
    for l in LEVERS:
        for f in l.fields:
            out[f] = l.name
    return out


# ── (3) Bundle: an edge over fields. No values. ────────────────────────────
@dataclass(frozen=True)
class Bundle:
    name: str
    # Two or more, always: one field is not an edge, and "a card whose
    # proposal is a single field" is something the diagnosis LLM can write
    # itself. JOURNAL 33 removed four cards for being scalar-only and left
    # the boolean-only ones because the criterion was written as "scalar";
    # `len(fields) >= 2` is that criterion with no wording to get wrong.
    fields: frozenset[str]
    rationale: str
    # Predicate over the current config. Unlike the vendor trigger this never
    # sees a proposal, because a Bundle has nothing to propose.
    trigger: Callable[[dict], bool] = lambda cfg: True

    def __post_init__(self):
        if len(self.fields) < 2:
            raise ValueError(
                f"Bundle {self.name!r} has {len(self.fields)} field(s): an "
                f"edge needs two. A single-field move is an action-space "
                f"proposal, not a bundle (METHODOLOGY 3).")


# ── (3) Constraint: a forbidden state. ─────────────────────────────────────
@dataclass(frozen=True)
class Constraint:
    name: str
    forbids: Callable[[object], bool]   # (resolved cfg) -> True when illegal
    message: str                        # why, with the measurement
    repair_hint: str                    # which fields to move together

    def check(self, cfg) -> str:
        """Empty string when legal, else the full refusal text."""
        try:
            bad = bool(self.forbids(cfg))
        except Exception:                # a predicate that cannot evaluate
            return ""                    # is not evidence of an illegal state
        return f"{self.message} {self.repair_hint}" if bad else ""


# ── (3') Rule: a query-rewriting instruction. Text, no fields. ─────────────
@dataclass(frozen=True)
class Rule:
    name: str
    # Per corpus language. The rewrite LLM is prompted in the corpus language
    # for the same reason the diagnosis LLM is (JOURNAL 29).
    text: dict[str, str]
    # The lever coordinate this rule creates, as levers.py spells it. This is
    # the field that makes a Rule different in kind from the other two: a
    # rule can ADD a cell to the registry, which no config field can do. It
    # is also the only legitimate route by which tier (2) widens tier (1),
    # and the gate checks that the coordinate exists.
    coordinate: tuple[str, str]
    rationale: str
    # True for the one rule whose text IS the historical prompt. The rewrite
    # cache on disk was produced by that exact string; assembling a different
    # one while reusing the cache would mean the measured numbers came from a
    # prompt nobody can reconstruct. verify.py asserts the identity.
    legacy: bool = False


# ── Pools ──────────────────────────────────────────────────────────────────
# Migrated from RECIPES_LEGAL (5 hand-written cards) on 2026-11-16.
#
#   lexical_first_light_semantic -> BUNDLE + CONSTRAINT (values 0.3/100 dropped)
#   query_rewrite_append_once    -> RULE (the card switched on a rewrite whose
#                                   actual content is the prompt below)
#   entity_swap_shallow          -> dropped: depth scalar, one field
#   undo_structured_view         -> dropped: one field, the loop can undo it
#   undo_chapter_filter          -> dropped: the default was already False;
#                                   reading its lift as a finding was JOURNAL 48
BUNDLES: tuple[Bundle, ...] = (
    Bundle(
        name="semantic_needs_score_fusion",
        fields=frozenset({"fusion_mode", "semantic_top_k"}),
        rationale=(
            "Switching the semantic view on without moving fusion off a "
            "non-scoring mode measured -0.115: under `first_found` the views "
            "are concatenated in a fixed order and the first one fills "
            "max_context. The two fields are not independent, so spending "
            "one step-cap slot on each leaves an intermediate state that "
            "elitist reverts -- which is the whole reason the bundle exists "
            "(METHODOLOGY 3, 'it opens a path, it does not widen the "
            "space'). NOTE: `first_found` is now refused outright, so what "
            "survives here is the weaker dependency between depth and "
            "fusion, and whether that still earns a bundle is the 2x2 this "
            "section specifies."),
        trigger=lambda cfg: int(cfg.get("semantic_top_k") or 0) == 0,
    ),
)

CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint(
        name="no_first_found",
        forbids=lambda cfg: getattr(cfg, "fusion_mode", "") == "first_found",
        message=(
            "fusion_mode='first_found' is not supported in legal retrieval: "
            "multi_retriever._fuse concatenates the views in the fixed order "
            "(semantic, keyword, structured, swap), dedupes and does NOT "
            "sort, and the only cut is fused[:max_context] at the end. So "
            "with semantic_top_k >= max_context the first view takes every "
            "slot and BM25 contributes exactly nothing -- measured on "
            "LeCaRDv2 test, first_found (semantic 100 + BM25 100) scores "
            "0.2797, identical to semantic_only, against keyword_only "
            "0.3948. The fixed order assumes semantic is the strongest view, "
            "true for LoCoMo and backwards here."),
        repair_hint=(
            "Use 'rrf' or 'weighted_sum'. Note that RetrievalConfig's own "
            "default is 'first_found', so a config built without naming the "
            "mode lands here."),
    ),
)

# The seed rule. Its text is the historical query_rewrite.PROMPTS verbatim --
# changing a character would silently decouple cache/rewrite/*.json from the
# prompt that produced it, and the cache is keyed by query id, not by prompt.
#
# Its coordinate is ("strip", "bm25"), which is party-strip's coordinate: this
# rule DELETES (names, places, plate numbers, dates, amounts) and restates
# what is left. That collision is the R2 waiver standing in levers.py, and it
# is why the lever measures +0.0009 CI∋0 -- appended to the original query, a
# deletion-style rewrite introduces no term the original lacks, so it can only
# reweight term frequencies. The cell that is empty is ("expand", "bm25").
SEED_REWRITE = Rule(
    name="strip_normalize",
    coordinate=("strip", "bm25"),
    legacy=True,
    rationale=(
        "The original hand-written rewrite: drop case-specific spans, keep "
        "the conduct and the legal issue, restate in legal terms. Kept as "
        "the seed because 'When Rules Learn' also seeds with one "
        "human-written rule and says a new benchmark should be given an "
        "appropriate one; an empty repository measured 32 proposals and 0 "
        "admissions (JOURNAL 23)."),
    text={
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
    },
)

RULES: tuple[Rule, ...] = (SEED_REWRITE,)


def assemble_rewrite_prompt(lang: str, rules: tuple[Rule, ...] = RULES) -> str:
    """The prompt the rewrite LLM receives, built from the rule pool.

    One legacy rule renders byte-identically to the historical constant, so
    an existing cache stays attributable. Any other pool renders as a
    skeleton plus numbered rules.
    """
    if len(rules) == 1 and rules[0].legacy:
        return rules[0].text[lang]
    head = {
        "zh": "你是法律检索助手。请按下列规则改写事实描述，用于检索相似判例。",
        "ko": "다음 사실관계를 아래 규칙에 따라 유사 판례 검색용 질의로 바꾸세요.",
        "en": "Rewrite the case facts below into a query for retrieving "
              "similar cases, following every rule.",
    }[lang]
    tail = {
        "zh": "事实描述：\n{text}\n\n只输出改写后的查询，不要解释。",
        "ko": "사실관계:\n{text}\n\n변환된 질의만 출력하세요.",
        "en": "Case facts:\n{text}\n\nOutput only the rewritten query.",
    }[lang]
    body = "\n".join(
        f"{i}. {r.text[lang]}" for i, r in enumerate(rules, 1))
    return f"{head}\n\n{body}\n\n{tail}"


BUNDLE_WIRING = ("wired: --recipes bundles installs shims; "
                 "LegalEvolutionEngine widens the cap by the bundle fields "
                 "the LLM proposed")


# ── Reaching the loop ──────────────────────────────────────────────────────
# A Bundle cannot be handed to the vendor engine as a `Recipe`: `proposal` is
# exactly the values a Bundle refuses to carry. What it CAN use is the one
# property of that path we want -- the engine already treats an invoked
# recipe's fields as exempt from the per-round step cap.
#
# So a Bundle enters as a SHIM: a vendor Recipe with `proposal={}`. The
# engine then resolves the name (so we learn which bundle the LLM chose) and
# merges nothing (so no value is injected). The cap widening is done on our
# side, in LegalEvolutionEngine, because the vendor would otherwise charge a
# whole slot for a recipe that set nothing.
#
# `_already_rejected({})` returns False by construction -- it short-circuits
# on an empty scalar set -- which is the right semantics: a bundle is not a
# move, so it cannot have been tried and rejected.
def bundle_by_name(name: str | None) -> Bundle | None:
    if not name:
        return None
    for b in BUNDLES:
        if b.name == name:
            return b
    return None


def as_vendor_shims(bundles: tuple[Bundle, ...] = BUNDLES) -> list:
    """Vendor `Recipe` objects that carry a Bundle's name and nothing else."""
    from evolvemem.evolution_cookbook import Recipe
    out = []
    for b in bundles:
        out.append(Recipe(
            name=b.name,
            description=(
                "BUNDLE. This card sets nothing. Invoking it lets you change "
                "the fields it names in the SAME round without each one "
                "costing a step-cap slot, because they are not independent: "
                + b.rationale + " Fields: " + ", ".join(sorted(b.fields))
                + ". You choose every value yourself."),
            # The vendor trigger signature is (report_fields, config, history);
            # a Bundle's own predicate reads the config alone, because an edge
            # between fields does not depend on this round's failures.
            trigger=(lambda rf, cfg, hist, _b=b: _b.trigger(cfg)),
            proposal={},
            benchmark="legal_retrieval",
        ))
    return out
