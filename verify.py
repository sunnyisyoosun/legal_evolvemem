"""One gate to run before an evolution run. Three checks, one exit code.

    python3 verify.py                 # every dataset, every check
    python3 verify.py --dataset LeCaRD

**prompt** -- assembles the diagnosis prompt through the real call path and
asserts on the finished string: every proposable field described and no
other, no dropped card named, threshold and round budget stated, no
unsubstituted marker, no LoCoMo residue, and CJK present for a CJK corpus.
Checking the pieces is not checking the prompt: on 2026-11-11 two correct
fixes left the field in the prompt anyway, because the action space was
described in two hand-written lists (JOURNAL 38).

**levers** -- checks the lever registry (levers.py) against the live action
space: no two active levers share a coordinate (R2), every proposable field is
accounted for exactly once, and no registered field has fallen out of
RetrievalConfig. Dataset-independent, so it runs once. R2 is the rule that
`enable_query_rewrite` and `enable_entity_swap` broke for weeks while §5 had
already measured the overlap -- a rule nothing checks is a paragraph.

**vectors** -- runs the real retrieval path with the semantic view replaced by
a collector and asserts that every query string it is asked for has a
precomputed vector. A missing one makes `semantic_search` return nothing,
silently, for those queries (JOURNAL 43). Under `SEMANTIC_USES_ROW_TEXT` this
passes with no precomputation at all; the check is what proves that.

Merged from verify_prompt.py and derived_queries.py on 2026-11-12. They were
two files asking the same question -- is this run well formed -- and a gate
you have to remember to run twice is a gate.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import legal_evolve as LE
import diagnosis_prompts as DP
import recipe_learning as RL
import legal_retriever as L
import registry as REG
import query_rewrite as QR
from evaluate import CASE_DIR, weak_config, in_split
from legal_meta import LANG_OF
from evolvemem import diagnosis as DIAG, evolution_cookbook as CB
from evolvemem.multi_retriever import RetrievalConfig

ROUNDS = 7


def assemble(dataset: str) -> tuple[str, list, float, list]:
    """The prompt a round-1 diagnosis would receive, plus what shaped it."""
    importlib.reload(CB)
    CB.RECIPE_POOLS["legal_retrieval"] = list(CB.RECIPES_LEGAL)
    qa = [{"category": c, "meta": {"qtype": f"class{c}"}} for c in (1, 2)]
    LE._install_diagnosis_context(qa, dataset)
    RL.apply_measured_lifts(dataset)
    gone = DP.drop_dead_cards(CB, dataset)
    DP.install_recipe_text(LANG_OF[dataset], CB)
    thr, _ = RL.noise_floor(dataset)
    LE.set_budget_info(thr, ROUNDS)
    # Enough view state for extra_context to render; the numbers are dummies,
    # only the presence of the blocks is under test.
    LE._VIEW_STATS.update({"_n": {"hits": 50}, "_enabled": {"keyword"},
                           "keyword": {"hits": 20, "gold": 5, "sole_gold": 3}})
    LE._LAST_CFG = replace(weak_config(), fusion_mode="rrf",
                           semantic_top_k=100)
    prompt = DIAG._filtered_prompt().format(
        benchmark="legal_retrieval", total_memories=1000, total_questions=100,
        overall_f1=0.3, zero_count=10, current_config="{}",
        failure_summary="(summary)", category_breakdown="(breakdown)",
        sample_failures="(samples)", attempt_history="(none)",
        todo_checklist="", available_recipes=CB.format_recipes_for_prompt(
            CB.RECIPE_POOLS["legal_retrieval"]),
        extra_context=DIAG._extra_context_block())
    pool = [r.name for r in CB.RECIPE_POOLS["legal_retrieval"]]
    return prompt, pool, thr, gone


def check_prompt(dataset: str) -> list[str]:
    bad: list[str] = []
    prompt, pool, thr, gone = assemble(dataset)
    fields = set(DIAG.ACTION_FIELDS or ())

    def fail(msg):
        bad.append(f"{dataset}: {msg}")

    # 1. every proposable field is described; no other field is named
    for f in RL.EFFECTIVE_FIELDS:
        named = f in prompt
        if f in fields and not named:
            fail(f"{f} is proposable but not described in the prompt")
        if f not in fields and named:
            fail(f"{f} is NOT proposable but appears in the prompt")
    # 2. no dropped card is named
    for name in gone:
        if name in prompt:
            fail(f"dropped card {name} still named in the prompt")
    # 3. every surviving card IS named
    for name in pool:
        if name not in prompt:
            fail(f"card {name} is in the pool but not in the prompt")
    # 4. the acceptance threshold and the round budget are stated
    if thr <= 0:
        fail("no measured noise floor; the run would have no threshold")
    # The threshold must come from the config-pair estimator, not from the
    # recipe cards. A card-derived floor is a property of the POOL: it moved
    # whenever a card was added or removed, and three of the eight cards
    # behind the live numbers no longer existed (METHODOLOGY 5,
    # noise_floor.py). The fallback stays readable so an old checkout still
    # runs, but a gate that lets it through is a gate that keeps it.
    _, why = RL.noise_floor(dataset)
    if why.startswith("FALLBACK"):
        fail(f"noise floor is still card-derived ({why}); run "
             f"`python3 noise_floor.py --datasets {dataset}`")
    elif f"{thr:.4f}" not in prompt:
        fail(f"acceptance threshold {thr:.4f} not stated in the prompt")
    if str(ROUNDS) not in prompt:
        fail("round budget not stated in the prompt")
    # 4b. `first_found` is reachable from nowhere
    #
    # Three places had to agree and none of them was checked: the value the
    # diagnosis LLM may propose (diagnosis ENUM_VALUES, overridden in
    # legal_evolve.setup), the value a recipe may carry
    # (recipe_learning.ENUMS), and the prompt text that tells the LLM which
    # values exist. The prompt may NAME it -- it now says why it is refused --
    # so the string check is on the two validators, and the prompt is checked
    # for the refusal instead of for absence.
    if "first_found" in DIAG.MemoryDiagnostics.ENUM_VALUES.get("fusion_mode", ()):
        fail("first_found is still proposable: legal_evolve.setup did not "
             "override diagnosis ENUM_VALUES")
    if "first_found" in RL.ENUMS.get("fusion_mode", ()):
        fail("first_found is still a legal recipe value in recipe_learning.ENUMS")

    # 5. no marker left unsubstituted, no LoCoMo residue
    for marker in (DP.ACTION_SPACE_MARKER, DP.PRIORITY_MARKER):
        if marker in prompt:
            fail(f"{marker} left in the prompt")
    for word in ("locomo", "LoCoMo", "abstention", "answer_style"):
        if word in prompt:
            fail(f"LoCoMo residue in the prompt: {word!r}")
    # 6. the prompt is in the corpus language -- a CJK dataset whose prompt
    #    has no CJK means a variant silently fell back to English
    lang = LANG_OF[dataset]
    if lang in ("zh", "ko"):
        cjk = sum(1 for ch in prompt if "　" <= ch <= "鿿"
                  or "가" <= ch <= "힯")
        if cjk < 200:
            fail(f"prompt for a {lang} corpus has only {cjk} CJK characters")
    print(f"  {dataset:16s} lang={lang} fields={len(fields)} "
          f"cards={len(pool)} thr={thr:.4f} dropped={gone or '-'} "
          f"chars={len(prompt)}")
    return bad


def check_levers() -> list[str]:
    """R2 and registry/action-space agreement. METHODOLOGY 3.

    Dataset-independent: the registry describes the retrieval architecture,
    which is tier (1) and the same for every benchmark by construction.
    """
    bad: list[str] = []

    def fail(msg):
        bad.append(f"levers: {msg}")

    # Vocabulary. A free-form coordinate would make R2 unenforceable -- two
    # spellings of the same cell would not collide.
    for l in REG.LEVERS:
        if l.input not in REG.INPUTS:
            fail(f"{l.name}: input {l.input!r} not in {REG.INPUTS}")
        if l.scorer not in REG.SCORERS:
            fail(f"{l.name}: scorer {l.scorer!r} not in {REG.SCORERS}")
        if l.combine not in REG.COMBINES:
            fail(f"{l.name}: combine {l.combine!r} not in {REG.COMBINES}")
        if l.status not in REG.STATUSES:
            fail(f"{l.name}: status {l.status!r} not in {REG.STATUSES}")
        if l.status == "vacant" and (l.fields or l.shares):
            fail(f"{l.name}: a vacant cell cannot own or read fields")

    # R2 -- coordinate uniqueness among ACTIVE levers. A waiver on either side
    # downgrades the collision to a reported known duplicate: the resolution
    # is a decision about what the lever should be, not a typo, and a gate
    # that blocks every run until a research question is settled gets
    # disabled. A collision with NO waiver is a hard failure.
    seen: dict[tuple, REG.Lever] = {}
    waived: list[str] = []
    for l in REG.active():
        c = REG.coordinate(l)
        if c in seen:
            other = seen[c]
            if l.coord_waiver or other.coord_waiver:
                waived.append(f"{other.name} ~ {l.name} {c}")
            else:
                fail(f"R2: {other.name} and {l.name} share coordinate {c} "
                     f"with no waiver -- one of them has to go")
        else:
            seen[c] = l

    # The registry has to describe the action space the loop actually gets,
    # not a past one. `set_action_fields` is what the engine enforces, and
    # EFFECTIVE_FIELDS is what feeds it before the per-dataset drop.
    proposable = set(RL.EFFECTIVE_FIELDS)
    owned = REG.owned_fields()
    accounted = set(owned) | set(REG.NON_LEVER_FIELDS)
    for f in sorted(proposable - accounted):
        fail(f"{f} is proposable but no lever owns it and it is not listed "
             f"in NON_LEVER_FIELDS -- a field nobody registered")
    for f in sorted(accounted - proposable):
        fail(f"{f} is registered but not proposable (stale registry entry, "
             f"or it was dropped from EFFECTIVE_FIELDS)")

    # A field owned twice means two levers claim the same knob; `shares` is
    # the declared way to read someone else's field (party-strip has no
    # weight of its own).
    counts: dict[str, list[str]] = {}
    for l in REG.LEVERS:
        for f in l.fields:
            counts.setdefault(f, []).append(l.name)
    for f, owners in sorted(counts.items()):
        if len(owners) > 1:
            fail(f"{f} is owned by {owners}; use `shares` for the reader")

    # Drift against the dataclass: a renamed field in the vendor config would
    # otherwise leave the registry quietly describing something that no
    # longer exists.
    real = {fl.name for fl in __import__("dataclasses").fields(RetrievalConfig)}
    for l in REG.LEVERS:
        for f in l.fields + l.shares:
            if f not in real:
                fail(f"{l.name}: {f} is not a RetrievalConfig field")

    # ── the three recipe types (METHODOLOGY 3) ────────────────────────────
    # Bundle: `__post_init__` makes a single-field bundle unconstructible, so
    # the gate checks the other half -- that a bundle names fields that exist
    # and are proposable. A bundle over a field the loop cannot move opens a
    # cap for nothing.
    for b in REG.BUNDLES:
        for f in sorted(b.fields):
            if f not in proposable:
                fail(f"bundle {b.name}: {f} is not in the action space")
    # A bundle reaches the loop as a vendor shim, and the shim has to carry
    # no values (that is the type's whole claim) and have a translation in
    # every corpus language -- `install_recipe_text` raises on a missing one,
    # mid-run, after the index is already built.
    for sh in REG.as_vendor_shims():
        if sh.proposal:
            fail(f"bundle shim {sh.name} carries a proposal {sh.proposal} -- "
                 f"a bundle sets nothing")
        for lang, table in DP.BY_LANG.items():
            if sh.name not in table:
                fail(f"bundle shim {sh.name} has no {lang} text; "
                     f"install_recipe_text would raise mid-run")
    # The cap seam is what makes an empty shim mean anything. Without it the
    # vendor charges a slot for a card that set nothing, which is strictly
    # worse than having no bundle at all.
    if not hasattr(LE.LegalEvolutionEngine, "_install_bundle_cap_seam"):
        fail("LegalEvolutionEngine has no bundle cap seam: an invoked bundle "
             "would spend a step-cap slot and set nothing")
    # Constraint: a predicate that throws is treated as "legal" at runtime by
    # design (a predicate that cannot evaluate is not evidence of an illegal
    # state), so a broken one would disappear silently. Exercise each against
    # theta_0, which must be legal, and against the vendor default, which the
    # standing constraint must refuse.
    t0 = weak_config()
    for c in REG.CONSTRAINTS:
        if c.check(t0):
            fail(f"constraint {c.name} refuses theta_0 -- no run could start")
    if not any(c.check(RetrievalConfig()) for c in REG.CONSTRAINTS):
        fail("no constraint refuses RetrievalConfig()'s own default "
             "(fusion_mode='first_found'); the -0.115 trap is open again")
    # Rule: the coordinate it creates has to be in the registry, and the seed
    # rule has to render byte-identically to the prompt that produced the
    # cache on disk -- the cache is keyed by query id, so a drifted prompt
    # leaves old rewrites attributed to a rule that did not make them.
    coords = REG.coordinates()
    for r in REG.RULES:
        if r.coordinate not in coords:
            fail(f"rule {r.name}: coordinate {r.coordinate} is not in the "
                 f"lever registry -- a rule may create a cell, but the "
                 f"registry has to record it")
        for lang in ("zh", "ko", "en"):
            if lang not in r.text:
                fail(f"rule {r.name}: no {lang} text (the rewrite prompt is "
                     f"in the corpus language)")
    for lang in ("zh", "ko", "en"):
        if REG.SEED_REWRITE.legacy and QR.PROMPTS[lang] != REG.SEED_REWRITE.text[lang]:
            fail(f"assembled rewrite prompt ({lang}) differs from the legacy "
                 f"text: cache/rewrite/*.json is no longer attributable")

    n_act = len(REG.active())
    print(f"  registry         levers={len(REG.LEVERS)} active={n_act} "
          f"fields={len(owned)}+{len(REG.NON_LEVER_FIELDS)} non-lever "
          f"= {len(accounted)} of {len(proposable)} proposable")
    for w in waived:
        print(f"    R2 waived (known duplicate): {w}")
    print(f"    recipes          bundles={len(REG.BUNDLES)} "
          f"constraints={len(REG.CONSTRAINTS)} rules={len(REG.RULES)} "
          f"({REG.BUNDLE_WIRING})")
    for l in REG.LEVERS:
        if l.status == "vacant":
            print(f"    vacant cell: {l.name} "
                  f"({l.input}, {l.scorer}, {l.combine})")
    return bad


def check_vectors(dataset: str) -> list[str]:
    """Every text the semantic view asks for must have a precomputed vector.

    The invariant is now narrow, and that is the point. Each view reads its
    own input and only one of them is the semantic view, which always reads
    the row's original query text (`SEMANTIC_USES_ROW_TEXT`); deletion is a
    lexical act, so party-strip is BM25's twin and has no semantic half
    (`entity_swap_semantic_k` is pinned to 0). So the set of texts the
    semantic view can ask for is exactly the set of row texts.

    This is CHECKED, not assumed: the pass runs the real retrieval path with
    every text-transforming lever on and asserts that nothing outside that
    set was requested. If a later change reopens the semantic view to a
    transformed query, this fails here rather than returning empty results
    silently -- the fault that JOURNAL 43 had to name after the fact. It
    costs one BM25 search per query and no GPU, because the stub returns [].

    What this replaced: a `--write` pass that collected "derived" query
    texts, `build_embeddings --derived` to encode them, and a loader that
    merged them into the query-vector map. All four datasets emitted ZERO
    derived texts and no vectors were ever built, because with the semantic
    view pinned to the row text there is nothing to derive.
    """
    bad: list[str] = []
    cases = json.load(open(f"{CASE_DIR}/{dataset}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{dataset}_queries.json", encoding="utf-8"))
    index = L.build_index(dataset, cases, qs)
    have = set(index._qvec)
    row_texts = {q["text"] for q in qs}

    # Everything that can change the query text, at once.
    cfg = replace(weak_config(), fusion_mode="rrf", semantic_top_k=100,
                  enable_entity_swap=True, entity_swap_semantic_k=15,
                  enable_query_rewrite=bool(L.load_rewrites(dataset)))
    asked: list[str] = []
    original = index.semantic_search
    try:
        # `index.semantic_query(q)` is the resolver `semantic_search` itself
        # uses. Recording the handed string would check a text the search
        # never looks up (JOURNAL 44).
        index.semantic_search = lambda q, top_k=20, _a=asked, _i=index: (
            _a.append(_i.semantic_query(q)) or [])
        for r in qs:
            L.search(index, r, cfg, top_n=30)
    finally:
        index.semantic_search = original

    off_row = set(asked) - row_texts
    if off_row:
        bad.append(f"{dataset}: semantic view asked for {len(off_row)} text(s) "
                   f"that are not a row's own query text -- the view is no "
                   f"longer pinned to the row, so those need vectors")
    missing = (set(asked) | row_texts) - have
    if missing:
        bad.append(f"{dataset}: {len(missing)} query text(s) have no vector")
    print(f"  {dataset:16s} vectors: {len(qs)} queries, "
          f"{len(set(asked))} distinct asked, {len(missing)} missing"
          + ("  <- semantic view would be dark" if missing else "  OK"))
    del index
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LANG_OF))
    ap.add_argument("--only", choices=["prompt", "vectors", "levers"],
                    help="run one check instead of all three")
    a = ap.parse_args()
    targets = [a.dataset] if a.dataset else sorted(LANG_OF)

    bad: list[str] = []
    print("verify: assembling the real prompt and the real retrieval path")
    if a.only in (None, "levers"):
        bad += check_levers()
    for ds in targets:
        if a.only in (None, "prompt"):
            bad += check_prompt(ds)
        if a.only in (None, "vectors"):
            bad += check_vectors(ds)
    print()
    if bad:
        print(f"GATE FAILED ({len(bad)}):")
        for b in bad:
            print(f"  - {b}")
        print("\nDo not start a run: a round under either fault records "
              "evidence the configuration did not actually produce "
              "(METHODOLOGY 13).")
        return 1
    print("GATE OK: registry, prompt and semantic view are sound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
