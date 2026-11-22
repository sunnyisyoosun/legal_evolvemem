"""Let the LLM propose recipes, and let measurement decide whether they stay.

The split of authority
----------------------
Three things were tangled together, and only one of them is a judgement call:

  what this setting actually scores   -> measurement. Not the LLM, not me.
  which move to make this round       -> the LLM. Already its job.
  which recipes should exist at all   -> was ME, hand-writing cards.

The third is the one that contradicts the premise. METHODOLOGY 9 recorded
"evolve the recipes themselves" as the next step and then kept hand-writing
them; METHODOLOGY 18 generated them, but from generation rules I wrote. This
module moves the third decision to the LLM while keeping the first with the
measurement, because a recipe's `expected_lift_pp` is shown to the diagnosis
LLM AS FACT -- an LLM-invented number there is the silent-signal failure of
METHODOLOGY 13, not a shortcut.

So: the LLM proposes, a fold measures, and the record keeps both plus the
provenance. A candidate that cannot be measured is never admitted.

Which fold admits a recipe
--------------------------
Not val. Val selects the final config (METHODOLOGY 5), and admitting recipes
on val would spend the same fold twice -- once to decide what the loop is
allowed to try, once to decide what it ends up with. The second use is no
longer independent of the first.

So train is split, deterministically by query id, into

  train-fit  (80%)  the evolution loop's fitness, as before
  train-dev  (20%)  recipe admission only

val and test are untouched. The cost is that the loop's fitness is measured
on 80% of train rather than all of it, which is the price of having the
admission decision be independent of it.

Triggers without executing model output
---------------------------------------
A hand-written recipe's trigger is a lambda. An LLM-written one cannot be:
running generated code against the config is an arbitrary-execution path for
the sake of a boolean. Candidates therefore declare triggers as data --
a list of {field, op, value} conditions ANDed together -- and this module
compiles them. Anything that does not parse is rejected, not repaired.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L  # noqa: E402  (also puts evolvemem on the path)
from evolvemem.multi_retriever import RetrievalConfig  # noqa: E402
from evolvemem.evolution_cookbook import Recipe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "learned_recipes.json")


def set_store(path: str) -> None:
    """Point the store elsewhere.

    Needed to compare prompt variants: the dedup check compares a candidate
    against everything the store holds, so two arms sharing one file would
    have different collision sets -- the first arm's admissions become the
    second arm's duplicates. The duplicate rates measured across the earlier
    runs are not comparable for exactly this reason (the collision set grew
    from 0 to 29 entries), which is why this exists.
    """
    global STORE
    STORE = path

# Fields a candidate may set. The answer-generation levers are excluded
# because nothing is generated in this task (METHODOLOGY 3): a recipe that
# moves them would be measured as a no-op and admitted or rejected by noise.
_INERT = {
    "answer_style", "answer_model", "answer_model_ensemble",
    "reflection_rounds", "verification_style", "enable_answer_verification",
    "locomo_cat5_mcq", "locomo_cat1_single_fact",
    "locomo_cat2_temporal_format", "locomo_cat3_inferential_nuanced",
    "locomo_cat4_verbatim_copy", "reference_date",
}
_FIELD_TYPE = {f.name: f.type for f in dataclasses.fields(RetrievalConfig)}

# Fields that measurably move the ranking in THIS pipeline.
#
# The first learner runs spent six of eight proposals on kg_expansion,
# coverage_reflection and decomposition bundles, every one of which measured
# exactly +0.0000. Those are LoCoMo levers that our retrieval path never
# reads, so the LLM was being invited to propose into a void -- and a
# proposal that cannot move the output is not rejected for being wrong, it is
# rejected for being unmeasurable, which teaches the loop nothing.
#
# So the list is a MEASUREMENT, not a judgement. Each field was set away from
# two baselines (theta_0, and a fused config with four views) and the returned
# doc ids compared, 25 LeCaRDv2 train queries
# (scratchpad/probe_inert.py, 2026-11-02):
#
#   acts from theta_0 : enable_query_rewrite 25/25, enable_chapter_filter 24/25
#   acts from fused   : semantic_top_k, enable_entity_swap, entity_swap_keyword_k,
#                       fusion_mode, weight_semantic, weight_keyword,
#                       mmr_diversity_weight  25/25;  structured_top_k 19/25,
#                       keyword_top_k 18/25, weight_structured 9/25
#   inert in both     : all kg_*, all coverage_*, all decomposition_*,
#                       intent_max_subqs / intent_merge_top_k /
#                       intent_final_context, max_context,
#                       entity_swap_semantic_k, query_rewrite_repeat,
#                       time_decay_half_life_days, mmr_candidate_pool
#
# Three of those zeros are surprising and are NOT explained yet, so they are
# recorded as observations rather than conclusions:
#   - `max_context` should truncate, but the adapter may be overriding it with
#     the caller's top_n
#   - `entity_swap_semantic_k` is inert while `entity_swap_keyword_k` acts, so
#     the party-strip view's semantic half may contribute nothing after fusion
#   - `query_rewrite_repeat` is inert although `enable_query_rewrite` acts
# `enable_intent_planning` was not probed: probing it ON costs one LLM call
# per query, which is the reason it is a candidate for removal in the first
# place.
EFFECTIVE_FIELDS = [
    "semantic_top_k", "keyword_top_k", "structured_top_k",
    "enable_entity_swap", "entity_swap_keyword_k", "fusion_mode",
    "weight_semantic", "weight_keyword", "weight_structured",
    "enable_query_rewrite", "enable_chapter_filter", "mmr_diversity_weight",
]
ALLOW_INERT = os.environ.get("RECIPE_ALLOW_INERT", "0") != "0"

# Categorical fields need their values enumerated, because the declared type
# is `str` and a string check passes anything. The LLM proposed
# `fusion_mode: "mmr"` four times; it is not an implemented mode
# (multi_retriever.py: rrf / weighted_sum / first_found, plus the
# single-view restrictions), so the config fell through to a degenerate path
# and measured -0.13. That is not a failed idea being rejected, it is an
# invalid value being scored -- the validator has to catch it.
#
# `first_found` is implemented but NOT offered: it concatenates the views in
# a fixed order without sorting, so the first view fills `max_context` and
# the rest are inert (legal_retriever.search raises on it; METHODOLOGY 2
# "fusion_mode is not a dimension"). A card proposing it would measure a
# number that means "semantic_only", which is the same class of mistake as
# the `mmr` value above.
ENUMS = {
    "fusion_mode": ("keyword_only", "semantic_only", "structured_only",
                    "weighted_sum", "rrf"),
}
# Ranges for the numeric levers. Open-ended above, because "how deep" is the
# loop's business; closed below, because a negative top-k or weight is not a
# proposal, it is a bug.
# A working value per lever, used when an inequality has to be satisfied.
# Taking the boundary instead produced a base of `semantic_top_k=1,
# keyword_top_k=1` for the trigger `> 0`, which scored 0.0814 -- and a
# candidate measured against that read +0.3092, a number that would then be
# shown to the diagnosis LLM as a measured fact. The base has to be a config
# someone could plausibly be in, not the weakest one that satisfies the text.
# Per-view candidate depth is 100 everywhere. Measured on train, all
# four datasets (sweep.py --grid depth / --grid swap-depth):
#   k=30  loses most of the semantic view's gain (LeCaRDv2 +0.0140
#         -> +0.0042; LeCaRD falls back to theta_0)
#   k=100 is the knee
#   k=200 adds +0.0024 on LeCaRDv2 and ~0 elsewhere, under its floor
# metadata's 10 was the engine default and party-strip's 30 came off
# the old corpus. Depth is not an echo of top_n=30, which the metric
# fixes (NDCG@30); these are how many candidates a VIEW contributes.
REPRESENTATIVE = {
    "semantic_top_k": 100, "keyword_top_k": 100, "structured_top_k": 100,
    "entity_swap_keyword_k": 100, "weight_semantic": 0.3,
    "weight_keyword": 1.0, "weight_structured": 0.1,
    "mmr_diversity_weight": 0.3,
}

RANGES = {
    "semantic_top_k": (0, 2000), "keyword_top_k": (0, 2000),
    "structured_top_k": (0, 2000), "entity_swap_keyword_k": (0, 2000),
    "weight_semantic": (0.0, 10.0), "weight_keyword": (0.0, 10.0),
    "weight_structured": (0.0, 10.0), "mmr_diversity_weight": (0.0, 1.0),
}

ALLOWED_FIELDS = (sorted(set(_FIELD_TYPE) - _INERT - {"per_category_overrides"})
                  if ALLOW_INERT else sorted(EFFECTIVE_FIELDS))

_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">":  lambda a, b: _num(a) > _num(b),
    "<":  lambda a, b: _num(a) < _num(b),
    ">=": lambda a, b: _num(a) >= _num(b),
    "<=": lambda a, b: _num(a) <= _num(b),
    "in": lambda a, b: a in (b if isinstance(b, (list, tuple)) else [b]),
}

# Levers that call the LLM once per QUERY. They are in the action space so
# the loop can turn them OFF, but a learned recipe that needs them ON is not
# admissible at face value: the cost does not appear in NDCG, and turning one
# on breaks the property that retrieval makes no LLM call per query
# (METHODOLOGY 3). The first learned card was exactly this -- its trigger
# demanded `enable_coverage_reflection=True`, a state theta_0 never reaches,
# so it measured a real gain on a branch the loop would never stand on.
COSTLY_FIELDS = {"enable_intent_planning", "enable_query_decomposition",
                 "enable_coverage_reflection"}
ALLOW_COSTLY = os.environ.get("RECIPE_ALLOW_COSTLY", "0") != "0"

MAX_FIELDS = 5          # a bundle, not a whole configuration
# The loop's STEP CAP is a different number and must not be confused with
# MAX_FIELDS: `legal_evolve.py --max-changes` defaults to 2. The proposal
# prompt was interpolating MAX_FIELDS into the sentence "the loop may change
# only N fields per round", telling the LLM 5 when the loop allows 2 -- and
# the whole argument for a recipe is that it spends ONE of those slots.
STEP_CAP = 2
NAME_MAX = 64
# Containment, not Jaccard. Two cards that agreed on 4 of their 5 fields
# scored identically (0.3924 both) and still passed a Jaccard check, because
# the union counted the one field each had to itself. Dividing by the SMALLER
# proposal catches "the same card with one extra field".
DUP_CONTAINMENT = 0.8
DEV_FRACTION = 0.2
# No module-level admission bar any more. 0.003 was the engine's LoCoMo-era
# default and it sits BELOW the measured noise floor on three of our four
# datasets (LeCaRD by 7.7x) -- the same defect that was fixed for the
# evolution acceptance threshold. `admit` now takes the bar explicitly and
# the caller passes `noise_floor(dataset)`.

# A point estimate over 106 queries is not a decision. Measured on
# LeCaRDv2 train-dev (scratchpad/noise_floor.py, 2026-11-02):
#
#   12 RANDOM configurations, nothing proposed them  ->  4 of 12 cleared
#   the +0.003 threshold, median |delta| 0.0085
#
# So the point threshold admitted roughly a third of arbitrary cards, and it
# did: of the first two admissions, `diversity_boost_mmr_shallow` (+0.0044)
# had a 95% CI of [-0.0063, +0.0152] -- it was admitted on noise.
#
# Admission therefore requires the LOWER BOUND of a paired bootstrap CI to
# clear zero, not the mean to clear a threshold. And because every proposal
# is a test against the same fold, the confidence level tightens with the
# number tested this session: with k proposals the interval is computed at
# 1 - ALPHA/k (Bonferroni). Without that, running more proposals is itself a
# way to manufacture an admission.
# A candidate's own forecast, recorded to SCORE the reasoning -- never to
# decide anything. Section 20 forbade the model from stating an expected gain,
# because a claimed number sitting next to measured ones is indistinguishable
# from them. That rule stands: a prediction lives in its own key, is never
# rendered into a prompt, and never reaches `expected_lift_pp`. The only thing
# it is used for is comparing what was predicted against what was measured,
# which is the first evidence we would have about whether the reasoning is
# worth the tokens.
BANDS = {            # label -> (low, high) in NDCG@30
    "<0.005": (0.0, 0.005),
    "0.005-0.02": (0.005, 0.02),
    ">0.02": (0.02, 1.0),
}
VIEWS = ("keyword", "semantic", "metadata", "party-strip", "fusion")

ALPHA = 0.05
BOOTSTRAP_B = 2000
MAX_POOL = 12            # admitted recipes kept; see `crowding` below


def bootstrap_ci(deltas: list[float], alpha: float = ALPHA,
                 b: int = BOOTSTRAP_B, seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap CI over per-query deltas.

    Paired, because the two configs are scored on the same queries: the
    variance that matters is the variance of the DIFFERENCE, which is much
    smaller than the variance of either score.
    """
    import random
    import statistics as _st
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(_st.fmean(rng.choices(deltas, k=n)) for _ in range(b))
    lo = means[max(0, int((alpha / 2) * b))]
    hi = means[min(b - 1, int((1 - alpha / 2) * b))]
    return (lo, hi)


def measure_pair(index, rows, labels, base_cfg, change: dict,
                 base_scores: list[float] | None = None) -> dict:
    """Score `base_cfg` and `base_cfg+change` on the same queries, paired.

    One function because this IS the measurement, three times over: a
    recipe's lift, a rule's lift, and a noise-floor sample were three copies
    of "score two configs on one fold, difference per query, bootstrap the
    mean". The copies had already drifted -- one of them took its floor from
    a constant and another from the file it was writing.

    `base_scores` lets a caller reuse a base it already scored; the pairing
    is per query, so the two lists must come from the same `rows`.
    """
    import statistics as _st
    from dataclasses import replace as _replace
    from evaluate import per_query as _per_query
    bv = base_scores if base_scores is not None else _per_query(
        index, rows, labels, base_cfg)
    av = _per_query(index, rows, labels, _replace(base_cfg, **change))
    d = [y - x for x, y in zip(bv, av)]
    lo, hi = bootstrap_ci(d, alpha=ALPHA)
    return {"delta": _st.fmean(d) if d else 0.0, "ci95": [lo, hi],
            "halfwidth": (hi - lo) / 2.0,
            "base": _st.fmean(bv) if bv else 0.0,
            "after": _st.fmean(av) if av else 0.0,
            "deltas": d, "base_scores": bv}


def admit(deltas: list[float], after: float, theta0: float,
          n_tested: int = 1, *, threshold: float) -> tuple[bool, str]:
    """Three gates. Any one of them failing is a rejection.

    1. the CI lower bound clears zero, at a level corrected for how many
       candidates this session has tested against the same fold
    2. the mean clears `threshold`
    3. the state it leads to beats theta_0 -- a card can pass 1 and 2 by
       improving on a branch nobody would stand on

    `threshold` is keyword-only and has NO DEFAULT on purpose. It used to be
    the module constant 0.003, inherited from LoCoMo, which is under the
    measured noise floor on three of four datasets. Callers pass
    `noise_floor(dataset)`.

    One caveat the caller should print: that floor is measured on FULL
    train, while these deltas come from train-dev (20% of it). A smaller
    fold is noisier, so the train floor is a LOWER BOUND on the right bar
    here. Gate 1 is what actually binds on dev precision -- its CI is
    bootstrapped on the dev deltas themselves.
    """
    import statistics as _st
    alpha = ALPHA / max(1, n_tested)
    lo, hi = bootstrap_ci(deltas, alpha=alpha)
    mean = _st.fmean(deltas) if deltas else 0.0
    detail = (f"delta={mean:+.4f} CI{100*(1-alpha):.1f}%="
              f"[{lo:+.4f}, {hi:+.4f}] (k={n_tested})")
    if lo <= 0:
        return False, f"CI includes 0; {detail}"
    if mean <= threshold:
        return False, f"below +{threshold:.4f}; {detail}"
    if after <= theta0:
        return False, (f"beats its trigger base but not theta_0 "
                       f"({after:.4f} <= {theta0:.4f}); {detail}")
    return True, f"admitted; {detail}"


def crowding(store: dict) -> str | None:
    """Why a new card cannot be added even if it measures well.

    Nothing capped the pool. METHODOLOGY 18 measured what an uncapped pool
    does to exposure: 24 cards put a mean of 7.8 options per round in front
    of the LLM and the choice scattered. A learner that runs every week has
    no reason to stop, so the cap is here rather than in someone's memory.
    """
    n = len(store.get("admitted", []))
    if n >= MAX_POOL:
        return (f"pool is full ({n}/{MAX_POOL}); retire a card before "
                f"admitting another")
    return None


def _num(x):
    return -1e18 if x is None else (1.0 if x is True else 0.0 if x is False else float(x))


# ── folds ─────────────────────────────────────────────────────────────────
def dev_bucket(query_id) -> bool:
    """True when this train query belongs to train-dev.

    Hash of the id, not a shuffle: the split has to be identical across
    processes and across days without carrying a file around.
    """
    h = hashlib.sha1(str(query_id).encode()).hexdigest()
    return (int(h[:8], 16) % 100) < int(DEV_FRACTION * 100)


def split_train(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    fit = [r for r in rows if not dev_bucket(r["query_id"])]
    dev = [r for r in rows if dev_bucket(r["query_id"])]
    return fit, dev


# ── candidate validation ──────────────────────────────────────────────────
def _coerce(field: str, value):
    """Cast to the field's declared type, or raise ValueError."""
    t = str(_FIELD_TYPE.get(field, ""))
    if "bool" in t:
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "false"):
            return str(value).lower() == "true"
        raise ValueError(f"{field}: expected bool, got {value!r}")
    if "int" in t:
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{field}: expected int, got {value!r}")
        return int(value)
    if "float" in t:
        if value is None and "None" in t:
            return None
        return float(value)
    if "str" in t:
        return str(value)
    raise ValueError(f"{field}: unsupported type {t}")


def compile_trigger(conds: list[dict]):
    """Turn declared conditions into a predicate. No model output is executed."""
    checks = []
    for c in conds:
        f, op, v = c.get("field"), c.get("op"), c.get("value")
        if f not in ALLOWED_FIELDS:
            raise ValueError(f"trigger field not allowed: {f!r}")
        if op not in _OPS:
            raise ValueError(f"trigger op not allowed: {op!r}")
        if f in ENUMS:
            vals = v if isinstance(v, (list, tuple)) else [v]
            bad_v = [x for x in vals if op in ("==", "in", "!=") and x not in ENUMS[f]]
            if bad_v:
                raise ValueError(f"trigger value {bad_v} not implemented for {f}")
        checks.append((f, _OPS[op], v))

    def trigger(report, cfg, hist, checks=checks):
        try:
            return all(fn(cfg.get(f), v) for f, fn, v in checks)
        except Exception:
            return False
    return trigger


def trigger_base(base: dict, conds: list[dict]) -> dict | None:
    """A config that SATISFIES the trigger, built from `base`.

    Measuring a candidate against theta_0 is wrong whenever the trigger does
    not hold at theta_0: the first run recorded +0.0057 for a card whose
    trigger needed `semantic_top_k > 50` while measuring from
    `semantic_top_k = 0`, a state where that card can never fire. The number
    was real and meaningless.

    So the base is moved onto the trigger first, and the delta is measured
    from there. A trigger that cannot be satisfied by construction makes the
    candidate unmeasurable, and unmeasurable means not admitted.
    """
    cfg = dict(base)
    for c in conds:
        f, op, v = c["field"], c["op"], c.get("value")
        t = str(_FIELD_TYPE.get(f, ""))
        try:
            if op == "==":
                cfg[f] = _coerce(f, v)
            elif op == "in":
                opts = v if isinstance(v, (list, tuple)) else [v]
                cfg[f] = _coerce(f, opts[0])
            elif op in (">", ">="):
                if "bool" in t:
                    cfg[f] = True
                else:
                    step = 1 if "int" in t else 0.1
                    floor = float(v) + (step if op == ">" else 0)
                    rep = REPRESENTATIVE.get(f)
                    # the working value when it already satisfies the
                    # condition, the boundary only when it does not
                    cfg[f] = _coerce(f, rep if rep is not None and rep >= floor
                                     else floor)
            elif op in ("<", "<="):
                if "bool" in t:
                    cfg[f] = False
                else:
                    step = 1 if "int" in t else 0.1
                    ceil = float(v) - (step if op == "<" else 0)
                    if ceil < 0:
                        return None            # no non-negative value works
                    rep = REPRESENTATIVE.get(f)
                    cfg[f] = _coerce(f, rep if rep is not None and rep <= ceil
                                     else ceil)
            elif op == "!=":
                if "bool" in t:
                    cfg[f] = not _coerce(f, v)
                else:
                    continue                   # base likely already differs
        except (ValueError, TypeError, IndexError):
            return None
    # The construction has to actually hold, and the proposal has to change
    # something once it does.
    if not compile_trigger(conds)({}, cfg, []):
        return None
    return cfg


def parse_prediction(cand: dict) -> dict:
    """The forecast, normalised. Never a reason to reject a candidate.

    A malformed prediction is dropped rather than rejected: the prediction is
    our instrument for scoring the reasoning, and letting the instrument veto
    the thing it measures would quietly change what gets admitted.
    """
    p = cand.get("predicted")
    if not isinstance(p, dict):
        return {}
    out = {}
    d = str(p.get("direction", "")).lower()
    if d in ("up", "down"):
        out["direction"] = d
    if p.get("band") in BANDS:
        out["band"] = p["band"]
    v = [x for x in (p.get("view") if isinstance(p.get("view"), list)
                     else [p.get("view")]) if x in VIEWS]
    if v:
        out["view"] = v
    return out


def score_prediction(pred: dict, delta: float) -> dict:
    """Did the forecast hold? Direction and band only; `view` needs the
    per-view trace and is left for a separate pass."""
    if not pred:
        return {}
    out = {}
    if "direction" in pred:
        actual = "up" if delta > 0 else "down"
        out["direction_hit"] = pred["direction"] == actual
    if "band" in pred:
        lo, hi = BANDS[pred["band"]]
        out["band_hit"] = lo <= abs(delta) < hi
        out["band_actual"] = next(
            (k for k, (a_, b_) in BANDS.items() if a_ <= abs(delta) < b_), None)
    return out


def validate(cand: dict, existing: list[dict],
             banned: list[str] | None = None) -> dict:
    """Return a normalised candidate or raise ValueError saying what is wrong."""
    name = str(cand.get("name", "")).strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{3,%d}" % NAME_MAX, name):
        raise ValueError(f"name must be snake_case: {name!r}")
    # A reused name is a naming collision, not a duplicate idea. Rejecting on
    # it threw away substance: in the no-CoT arm, 7 candidates sharing one
    # name carried 6 DIFFERENT proposals and 5 sharing another carried 2, so
    # 12 of 14 were discarded before anything was measured -- while the CoT
    # arm, which happened to vary its names, got through to the proposal-level
    # check. The arms were being filtered differently, which invalidated the
    # comparison. Duplication is decided below, on the proposal.
    used = {e["name"] for e in existing}
    if name in used:
        stem, n = name, 2
        while f"{stem}_v{n}" in used and n < 100:
            n += 1
        name = f"{stem}_v{n}"

    prop = cand.get("proposal")
    if not isinstance(prop, dict) or not prop:
        raise ValueError("proposal must be a non-empty object")
    if len(prop) > MAX_FIELDS:
        raise ValueError(f"proposal sets {len(prop)} fields, max {MAX_FIELDS}")
    bad = [k for k in prop if k not in ALLOWED_FIELDS]
    if bad:
        raise ValueError(f"fields not in the action space (or inert): {bad}")
    hit = [k for k in prop if k in (banned or [])]
    if hit:
        raise ValueError(f"withheld this round to force coverage: {hit}")
    prop = {k: _coerce(k, v) for k, v in prop.items()}
    for k, v in prop.items():
        if k in ENUMS and v not in ENUMS[k]:
            raise ValueError(f"{k}={v!r} is not an implemented value; "
                             f"allowed: {', '.join(ENUMS[k])}")
        if k in RANGES and v is not None:
            lo, hi = RANGES[k]
            if not (lo <= float(v) <= hi):
                raise ValueError(f"{k}={v!r} outside [{lo}, {hi}]")

    conds = cand.get("trigger")
    if not isinstance(conds, list) or not conds:
        raise ValueError("trigger must be a non-empty list of conditions")
    compile_trigger(conds)                    # raises if malformed

    # A card that proposes the state its own trigger demands can never
    # change anything: it fires only where it is already satisfied.
    noop = [c for c in conds
            if c.get("op") == "==" and c.get("field") in prop
            and prop[c["field"]] == c.get("value")]
    if noop:
        raise ValueError(f"trigger and proposal agree, so it is a no-op: {noop}")

    if not ALLOW_COSTLY:
        on = [k for k in prop if k in COSTLY_FIELDS and prop[k] is True]
        if on:
            raise ValueError(
                f"proposal turns on per-query LLM levers {on}; their cost is "
                f"not in the metric (set RECIPE_ALLOW_COSTLY=1 to override)")
        need = [c["field"] for c in conds
                if c.get("field") in COSTLY_FIELDS
                and c.get("op") == "==" and c.get("value") is True]
        if need:
            raise ValueError(
                f"trigger requires per-query LLM levers {need} to be ON, a "
                f"state theta_0 never reaches, so the card would be dead")

    # Exact-match dedup is not enough: the first five proposals produced
    # three cards that were the same idea with one number changed
    # ("reduce semantic reliance" at top_k 50, 30, 30). A pool of near-copies
    # gives the LLM more options without giving it more moves.
    for e in existing:
        other = e.get("proposal") or {}
        if not other:
            continue
        shared = set(prop) & set(other)
        same = sum(1 for k in shared if prop[k] == other[k])
        smaller = min(len(prop), len(other)) or 1
        if same / smaller >= DUP_CONTAINMENT:
            raise ValueError(
                f"{same} of {smaller} fields identical to {e['name']}; "
                f"too close to an existing recipe")

    return {
        "name": name,
        "description": str(cand.get("description", "")).strip()[:600],
        "symptom": str(cand.get("symptom", "")).strip()[:400],
        "trigger": conds,
        "proposal": prop,
        "predicted": parse_prediction(cand),
    }


# ── store ─────────────────────────────────────────────────────────────────
def load_store() -> dict:
    if not os.path.exists(STORE):
        return {"admitted": [], "rejected": []}
    with open(STORE, encoding="utf-8") as f:
        return json.load(f)


def save_store(store: dict) -> None:
    with open(STORE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def as_recipes(store: dict | None = None,
               dataset: str | None = None) -> list[Recipe]:
    """The admitted candidates, as engine Recipe objects.

    `expected_lift_pp` is the measured delta, never a proposed one, and the
    description carries where it was measured so the diagnosis LLM can
    discount a thin measurement instead of reading a bare number as fact.

    `dataset` filters to the cards admitted ON that dataset, and callers in
    an evolution run must pass it. Without it every dataset received every
    card: three of the four admitted cards were measured on LEGAR alone and
    were being handed to the LeCaRD loops as though they carried evidence
    there (JOURNAL 34). The dataset was never missing from the record --
    `measured_block` has always written it -- it was simply never read.
    """
    store = store or load_store()
    out = []
    for e in store.get("admitted", []):
        m = e.get("measured", {})
        if dataset is not None and m.get("dataset") != dataset:
            continue
        prov = (f" [measured {m.get('delta', 0) * 100:+.1f} pp on "
                f"{m.get('dataset')} train-dev, n={m.get('n')}, {m.get('date')}]")
        out.append(Recipe(
            name=e["name"],
            description=(e.get("description") or e.get("symptom") or "")[:600] + prov,
            trigger=compile_trigger(e["trigger"]),
            proposal=e["proposal"],
            expected_lift_pp=round(m.get("delta", 0.0) * 100, 2),
            benchmark="legal_retrieval",
        ))
    return out


LIFTS = os.path.join(HERE, "recipe_lifts.json")
FLOORS = os.path.join(HERE, "noise_floor.json")
RULE_LIFTS = os.path.join(HERE, "rule_lifts.json")


def noise_floor(dataset: str) -> tuple[float, str]:
    """Smallest delta on `dataset` train that is distinguishable from noise.

    Why this replaces the engine's `acceptance_threshold = 0.003`
    ------------------------------------------------------------
    0.003 is an absolute delta inherited from the LoCoMo-era default. It was
    never checked against the precision of our folds, and it is BELOW the
    noise floor on three of our four datasets:

        dataset          n      noise floor   0.003 (=0.30 pp) is
        LeCaRD           79        2.40 pp     7.7x TOO SMALL
        LeCaRDv2        576        0.50 pp     1.7x too small
        COLIEE         1148        0.52 pp     1.7x too small
        LEGAR_stricter 4228        0.17 pp     1.7x too large

    A threshold under the noise floor accepts rounds that cannot be
    distinguished from doing nothing, and elitism then carries that state
    forward as the incumbent best.

    What "noise" means here
    -----------------------
    Retrieval is deterministic: the same (query, config) always scores the
    same, so there is no run-to-run variance to average out. The uncertainty
    is that this train fold is ONE SAMPLE OF QUERIES -- a delta measured on
    it may not hold on others. The paired bootstrap over queries is what
    measures that, and `measure_recipes.py` already ran it for every card:
    `recipe_lifts.json` stores a CI95 per card per dataset. The half-width of
    those intervals is the quantity an acceptance threshold has to clear.

    The estimator is the MEDIAN half-width over that dataset's comparisons,
    with zero-width intervals dropped. A zero width means every per-query
    difference was exactly 0 -- an inert lever, which measures the noise of
    nothing.

    WHERE the comparisons come from changed on 2026-11-22. They used to be
    the recipe cards `measure_recipes.py` happened to have measured, which
    made the threshold a property of the POOL: of the eight cards behind the
    live numbers, three no longer existed and three more were removed that
    day, and cards are a biased sample anyway (a card is a move someone
    thought would help). They are now fixed CONFIG PAIRS -- `noise_floor.py`
    -- which depend on no pool and get better with more pairs instead of
    worse as the pool shrinks.

    The card-derived file is still read as a FALLBACK so a checkout that has
    not run the estimator yet keeps its old behaviour, and the provenance
    string says which source answered. Rebuild with:

        python3 noise_floor.py --datasets <ds>

    Returns (threshold, provenance). Raises nothing: a dataset with no row
    gets (0.0, reason) and the caller decides what to do about it.
    """
    try:
        with open(FLOORS, encoding="utf-8") as f:
            floors = json.load(f)
        row = floors.get(dataset) or {}
        if row.get("floor"):
            return (row["floor"],
                    f"median CI95 half-width over {row['usable_pairs']} "
                    f"config pairs on {dataset} {row['split']} "
                    f"(n={row['n']}, noise_floor.py)")
    except (OSError, ValueError):
        pass
    try:
        with open(LIFTS, encoding="utf-8") as f:
            table = json.load(f)
    except (OSError, ValueError):
        return (0.0, "recipe_lifts.json unreadable")
    row = table.get(dataset)
    if not row:
        return (0.0, f"no recipe_lifts.json row for {dataset}")
    widths = []
    for m in (row.get("cards") or {}).values():
        ci = m.get("ci95")
        if not ci:
            continue
        hw = (ci[1] - ci[0]) / 2.0
        if hw > 1e-9:
            widths.append(hw)
    if not widths:
        return (0.0, f"{dataset}: no non-degenerate CI in recipe_lifts.json")
    widths.sort()
    mid = len(widths) // 2
    med = widths[mid] if len(widths) % 2 else (widths[mid - 1] + widths[mid]) / 2
    return (round(med, 5),
            f"FALLBACK card-derived floor: median CI95 half-width over "
            f"{len(widths)} recipe cards on {dataset} {row['split']} "
            f"(n={row['n']}) -- run noise_floor.py to replace it")


def apply_measured_lifts(dataset: str) -> tuple[int, int]:
    """Overwrite the hand-written `expected_lift_pp` with train measurements.

    The eight values in RECIPES_LEGAL were typed in from the sweep in
    METHODOLOGY 8, which ran on TEST. They are not inert: the diagnosis
    prompt renders them as "(expected +N pp)", so the leak has been steering
    every round. measure_recipes.py re-measures on TRAIN with a paired
    bootstrap CI.

    Source changed on 2026-11-22: `rule_lifts.json`, not `recipe_lifts.json`.
    After the type split only a Rule has a lift -- a Bundle carries no values
    and so has no delta to measure -- and measure_recipes.py no longer
    re-measures the hand-written cards. The frozen card table is NOT read
    here on purpose: a number nobody re-measures is exactly the leak this
    function exists to close, and `set_show_unmeasured_lift(False)` means an
    unmeasured card simply shows no figure.

    A card gets 0.0 unless three things hold, and the note says which one
    failed: the paired CI clears zero, the mean clears the same 0.003 floor
    elitism uses, and the END STATE beats theta_0. The third was missing and
    mattered: a card is measured from the state its own trigger accepts
    (section 22), so the lift is CONDITIONAL, not additive -- on LeCaRD train,
    undo_chapter_filter climbed 0.4677 -> 0.6241 and claimed +15.65 pp while
    theta_0 is 0.6637. Climbing out of a hole the trigger state put you in is
    not a gain. That is not a claim the card is useless -- it is the refusal
    to show a number we cannot support (section 21).

    The lift is per DATASET, because one number cannot be right for four
    corpora in three languages: the same lever flipped sign between query
    classes of a single dataset (section 15), let alone across them.

    Returns (cards updated, cards whose claim was zeroed).
    """
    from evolvemem import evolution_cookbook as CB
    try:
        with open(RULE_LIFTS, encoding="utf-8") as f:
            table = json.load(f)
    except (OSError, ValueError):
        return (0, 0)
    row = table.get(dataset)
    if not row:
        return (0, 0)
    n_set = n_zero = 0
    for r in CB.RECIPE_POOLS.get("legal_retrieval", []):
        m = row.get("rules", {}).get(r.name)
        if not m:
            continue
        r.expected_lift_pp = m["claim_pp"]
        # The prose of these descriptions is also full of pp figures from the
        # same test-fold sweep ("k=30 measured +1.1 pp", "appending once
        # measured +1.3 pp"). Replacing only `expected_lift_pp` would leave
        # those in the prompt looking equally measured. They are not
        # re-measurable from this script -- each is its own sweep -- so the
        # note says which figure is the measured one rather than pretending
        # the rest are.
        prov = (f" [measured {m['delta'] * 100:+.1f} pp on {dataset} "
                f"{row['split']}, n={row['n']}, CI95 "
                f"[{m['ci95'][0] * 100:+.1f}, {m['ci95'][1] * 100:+.1f}] pp"
                + (f"; no gain claimed -- {m['zero_reason']}" if
                   m["claim_pp"] == 0.0 and m.get("zero_reason") else
                   "; interval includes 0, so no gain is claimed" if
                   m["claim_pp"] == 0.0 else "")
                + (f". End state {m['vs_theta0']:+.4f} vs theta_0"
                   if "vs_theta0" in m else "")
                + ". Any pp figures in the text above predate this. They "
                  "came from a sweep on the TEST fold AND on the old corpus "
                  "(Fact only, embeddings truncated at 512 tokens), whose "
                  "view ranking has since reversed -- the semantic view is now "
                  "the strongest single addition, and the swap view measures "
                  "nothing. Trust only this bracketed figure.]")
        # Strip a provenance note from an earlier call so repeated installs
        # do not stack them.
        r.description = re.sub(r"\s*\[measured [^\]]*\]\s*$", "",
                               r.description) + prov
        # Marks the number as measured FOR THIS DATASET. Without it the
        # renderer prints no figure at all, which is the point: a hand-typed
        # value rendered as "(expected +N pp)" reads to the LLM exactly like
        # a measured one, and the nine hand-typed values were ordered the
        # inverse of what matters (evolution_cookbook.format_recipes_for_
        # prompt has the audit).
        # Both figures. The conditional delta alone overstated the card:
        # entity_swap_shallow reads +4.57 pp from its trigger base on LeCaRD
        # train while the state it lands in is only +0.61 pp above theta_0,
        # and the header is what a temperature-0 model selects on.
        import diagnosis_prompts as _DP
        if m["claim_pp"] == 0.0:
            r.lift_source = _DP.t("recipe.src_zero").format(
                ds=dataset, split=row["split"], n=row["n"],
                why=_DP.zero_reason(m.get("zero_reason")) or
                    _DP.t("recipe.zero_ci"))
        else:
            r.lift_source = _DP.t("recipe.src_gain").format(
                pp=m["claim_pp"], end=m.get("vs_theta0", 0.0) * 100,
                ds=dataset, split=row["split"], n=row["n"])
        n_set += 1
        n_zero += m["claim_pp"] == 0.0
    return (n_set, n_zero)


def install(extend: bool = True, dataset: str | None = None) -> int:
    """Put the learned recipes into the pool. Returns how many were added.

    Pass `dataset` from an evolution run so the pool holds only the cards
    measured on that dataset. See `as_recipes`.
    """
    from evolvemem import evolution_cookbook as CB
    learned = as_recipes(dataset=dataset)
    base = CB.RECIPE_POOLS.get("legal_retrieval", []) if extend else []
    CB.RECIPE_POOLS["legal_retrieval"] = list(base) + learned
    return len(learned)


# ── proposal prompt ───────────────────────────────────────────────────────
PROMPT = """You extend a library of retrieval "recipes" for legal case retrieval.

A recipe is a bundle of configuration fields that is applied together when a
symptom is observed. It exists because the evolution loop may change only
{step_cap} field(s) per round, and a recipe counts as ONE of those slots
however many fields it carries -- so a recipe is worth proposing when several
fields must move TOGETHER to help, and worthless when one field would do.

Retrieval works by scoring a query against four views and fusing them. Every
view's candidate depth is 100: that is measured, and 30 loses most of the
semantic view's gain. If you switch a view on, set its depth too.
  BM25 lexical (query as written)
  semantic (bge-m3 embedding, the query as written -- never a stripped or
            rewritten one, because deletion is a lexical act)
  metadata (statute/charge overlap)
  party-strip (BM25 again on the query with party, org and place names cut)

Current configuration being evolved:
{config}

Recipes already in the library (do not duplicate these):
{existing}

Fields the library already moves, with how many recipes move each. Proposing
another card on a crowded field is how the last run produced four copies of
one idea; the useful proposals are on fields nothing touches yet:
{coverage}

Candidates already rejected by measurement (do not re-propose these):
{rejected}

What has been measured in this domain so far:
{knowledge}

Fields you may set ({n_fields} of them). These are the only fields that were
MEASURED to change the ranking in this pipeline; the rest of the config is
inert here and a proposal touching them would score exactly 0.0000:
{fields}

Allowed values and ranges (a value outside these is rejected before it is
measured, which wastes the proposal):
{limits}

{reasoning}{answer_lead}

{{"name": "snake_case_name",
  "symptom": "the observable failure this addresses",
  "description": "why these fields must move together",
  "trigger": [{{"field": "<field>", "op": "==|!=|>|<|>=|<=|in", "value": <v>}}],
  "proposal": {{"<field>": <value>, ...}},
  "predicted": {{"direction": "up|down", "band": "<0.005|0.005-0.02|>0.02",
                "view": ["keyword|semantic|metadata|party-strip|fusion"]}}}}

Rules:
{ban_rule}- at most {cap} fields in `proposal`
- `trigger` conditions are ANDed; they must describe a state where the
  proposal would CHANGE something
- do NOT state an expected gain anywhere except inside `predicted`, and there
  only as one of the three bands. A free-form number would be
  indistinguishable from a measured one.
{reasoning_rule}
"""

# Measured facts handed to the proposer. EVERY line here must be a train
# measurement that still holds; a stale line is indistinguishable from a live
# one to the reader, and this block is the proposer's whole picture of the
# domain. Four lines were removed on 2026-11-13 because they had been
# overturned -- they are listed in the prompt itself as corrections, because
# silently dropping a claim leaves the LLM free to rediscover it.
KNOWLEDGE = """- BM25 alone is theta_0 on all four datasets. Turning the semantic view on
  with score fusion is the ONLY move positive on all four (train, +0.0140
  LeCaRDv2 / +0.0134 LeCaRD / +0.0041 COLIEE / +0.0062 LEGAR).
- The semantic view's best weight spans 0.1-2.0 ACROSS datasets -- a 20x
  range. There is no single right value; it is a property of the corpus.
- Every view's candidate depth is 100. At 30 most of the semantic view's gain
  is lost (LeCaRDv2 +0.0140 -> +0.0042). 200 adds nothing outside noise.
- The metadata view never helped in five different formulations: sharing a
  statute means the same offence, not a similar case. (Measured at depth 10;
  being re-measured at 100.)
- Per-query LLM steps (intent planning, decomposition) lost 8.8 pp AND break
  the property that retrieval makes no LLM call per query.
- Almost every gain in this domain came from intervening LESS, not more.
- The same lever can flip sign between query classes: the semantic view at
  weight 0.3 was +0.0119 on one class and -0.3421 on another.
- Gold documents share the query's party names (+16 pp vs random) and its
  place names (+51 pp). The metric therefore rewards a signal the task
  definition says is not relevance. Deletion is kept anyway, as a deliberate
  choice of the task definition over the metric.

CORRECTIONS -- claims that were in this block and are now known false. Do not
re-derive them:
- "The semantic view helps only at weight ~0.3 and degrades above it." FALSE:
  that was an artefact of embeddings truncated at 512 tokens. At 8192 the
  optimum moved and LEGAR's is 2.0.
- "BM25 alone 0.4040 / semantic alone 0.2814 on LeCaRDv2 train." Stale corpus
  (Fact section only). theta_0 is 0.3885 on the current corpus.
- "RRF beats weighted-sum once a 4th view contributes." Fusion mode is
  substantially irrelevant; what matters is whether the semantic view is on.
- "Party-strip at k=30 beat k=100: shallower intervention won." FALSE: every
  such verdict was taken with party-strip at 30 while BM25 ran at 100, which
  compared depths rather than views. At a common 100, party-strip is at or
  above the off state on two datasets.
- "Chapter filtering lost 8.3 pp." The direction is right; read the size
  with care. `undo_chapter_filter` measures +4.5 to +13.6 pp, but only
  because the card is measured from a state with the filter ON, which is
  not where theta_0 stands -- the default is already off, and the card is
  rejected on all four datasets for ending below theta_0. A card's lift is
  CONDITIONAL on its trigger state, never additive to theta_0."""


COT_BLOCK = """Work through these six steps in prose FIRST, then give the JSON. The steps
are not decoration: each one corresponds to a check that rejects candidates
before they are measured, so reasoning through them is how you avoid spending
a proposal on something that cannot be admitted.

1. WHICH AXIS IS FREE. Name a field from "never moved" above, or say why a
   crowded one still needs another card. Half of the rejections so far were
   near-duplicates of a card that already exists.
2. WHAT GOES WRONG NOW. Describe the retrieval failure you are targeting in
   terms of the four views -- which view is contributing what it should not,
   or not contributing what it should.
3. WHY A BUNDLE AND NOT ONE FIELD. A recipe is only worth a step-cap slot if
   the fields are useless apart. If one field alone would do, say so and
   propose that single field instead.
4. WHAT STATE IT FIRES IN. State the trigger in words, then check: in that
   state, does every field in your proposal actually differ from what the
   state already holds? If any field matches, the card is a no-op there.
5. WHY IT IS NOT ON THE REJECTED LIST. Compare against the rejected
   candidates above and say what is different.
6. WHAT YOU EXPECT TO HAPPEN. Commit to a direction, a magnitude band, and
   which view's contribution changes. This is recorded and compared against
   the measurement; it is NOT used to decide anything, so an honest "down,
   <0.005" costs you nothing and a confident wrong answer is recorded as
   such. If you cannot predict, say so and leave `predicted` out.
"""

NO_COT_BLOCK = ""


def crowded_fields(store: dict, k: int = 2, min_uses: int = 3) -> list[str]:
    """The k most-proposed fields, once they pass `min_uses`.

    Arm C needs a pool, and the pool is not growing because ~90% of proposals
    come back as near-duplicates: 16 proposals produced 14 duplicate
    rejections in one arm and 12 in the other, so the LLM is circling one
    region (semantic x mmr x fusion) and almost nothing reaches measurement.
    Telling it "do not re-propose these" has not moved the rate.

    So the crowded fields are withheld for a proposal, which is stratified
    sampling over the action space rather than a judgement about what is
    good: the choice of which fields to withhold is computed from the store's
    own counts, not from my opinion of the levers. At most `k` are withheld
    and only above `min_uses`, because the semantic view needs score fusion and
    banning `fusion_mode` early would make most useful bundles unreachable.
    """
    from collections import Counter
    c = Counter(f for e in (store.get("admitted", []) + store.get("rejected", []))
                for f in (e.get("proposal") or {}))
    return [f for f, n in c.most_common(k) if n >= min_uses]


def build_prompt(config: dict, store: dict, cap: int = MAX_FIELDS,
                 cot: bool = True, banned: list[str] | None = None,
                 step_cap: int = STEP_CAP) -> str:
    def brief(entries, keys=("name", "proposal")):
        """Existing recipes, with inert fields marked.

        The hand-written pool predates the inertness measurement, so some of
        its cards set fields that do nothing here (`query_rewrite_repeat`).
        Listing them unmarked invited a proposal built on one, which was then
        rejected for being inert -- the prompt was contradicting its own
        field list. Marking them keeps the example honest without hiding
        what the pool actually contains.
        """
        if not entries:
            return "  (none)"
        out = []
        for e in entries:
            parts = []
            for k, v in (e["proposal"] or {}).items():
                mark = "" if k in ALLOWED_FIELDS else "  <- INERT, do not use"
                parts.append(f'"{k}": {json.dumps(v, ensure_ascii=False)}{mark}')
            out.append(f"  - {e['name']}: {{{', '.join(parts)}}}")
        return "\n".join(out)
    from evolvemem.evolution_cookbook import RECIPE_POOLS
    pool = [{"name": r.name, "proposal": r.proposal}
            for r in RECIPE_POOLS.get("legal_retrieval", [])]
    from collections import Counter
    counts = Counter(k for e in (pool + store.get("admitted", []))
                     for k in e["proposal"])
    touched = ", ".join(f"{k}({n})" for k, n in counts.most_common()) or "(none)"
    untouched = ", ".join(sorted(set(ALLOWED_FIELDS) - set(counts))) or "(none)"
    return PROMPT.format(
        ban_rule=("- this proposal may NOT touch "
                  + ", ".join(f"`{f}`" for f in banned)
                  + ": those fields are already carried by several recipes, "
                    "and a proposal built on them comes back as a duplicate "
                    "without ever being measured\n" if banned else ""),
        reasoning=COT_BLOCK if cot else NO_COT_BLOCK,
        reasoning_rule=("- the reasoning is recorded with the candidate, so "
                        "write it as you would to\n  a colleague who will "
                        "check it, not as a justification of a decision\n"
                        "  already made" if cot else
                        "- give the JSON only; no explanation"),
        answer_lead=("Then, after the line `RECIPE:`, give the recipe as JSON "
                     "and nothing else:" if cot else
                     "Give the recipe as JSON and nothing else:"),
        cap=cap, step_cap=step_cap,
        coverage=f"  moved already: {touched}\n  never moved:   {untouched}",
        limits="\n".join(
            [f"  {k}: one of {', '.join(v)}" for k, v in ENUMS.items()]
            + [f"  {k}: {lo} .. {hi}" for k, (lo, hi) in sorted(RANGES.items())]),
        config=json.dumps({k: v for k, v in config.items()
                           if k in ALLOWED_FIELDS}, ensure_ascii=False, indent=1),
        existing=brief(pool + store.get("admitted", [])),
        rejected=brief(store.get("rejected", [])),
        knowledge=KNOWLEDGE,
        n_fields=len(ALLOWED_FIELDS) - len(banned or []),
        fields="  " + ", ".join(f for f in ALLOWED_FIELDS
                                if f not in (banned or [])),
    )


RECIPE_MARK = "RECIPE:"


def _json_spans(t: str) -> list[tuple[int, int, dict]]:
    """(start, end, object) for every balanced top-level {...}."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(t[start:i + 1])
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict):
                        out.append((start, i + 1, obj))
                    start = None
    return out


def _json_objects(t: str) -> list[dict]:
    """Every balanced top-level {...} in the text, in order.

    First-brace-to-last-brace stopped working once the prompt asked for
    reasoning: the reasoning mentions field names and example objects, so
    that span swallows the prose and fails to parse. Scanning for balanced
    braces, and taking the LAST complete object, keeps the answer separate
    from the thinking that precedes it.
    """
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(t[start:i + 1])
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict):
                        out.append(obj)
                    start = None
    return out


def split_reasoning(text: str) -> tuple[str, str]:
    """(reasoning, json-part). The marker when present, else the last object."""
    t = (text or "").strip()
    if RECIPE_MARK in t:
        head, tail = t.split(RECIPE_MARK, 1)
        return head.strip(), tail.strip()
    # The split point is where the last TOP-LEVEL object starts, not the last
    # "{" in the text: `trigger` holds nested objects, so rfind("{") landed
    # inside the answer and reported the first half of the JSON as prose.
    # That made the no-CoT arm look like it had written 741 characters of
    # reasoning when it had written none.
    body = re.sub(r"```(?:json)?", "", t)
    spans = _json_spans(body)
    if not spans:
        return t, ""
    start, end, _ = spans[-1]
    return body[:start].strip(), body[start:end]


def parse_response(text: str) -> dict:
    """The candidate object out of a response that may reason first."""
    _, tail = split_reasoning(text)
    body = re.sub(r"```(?:json)?", "", tail or text or "")
    objs = _json_objects(body)
    if not objs:
        raise ValueError("no JSON object in the response")
    # The last complete object: a reasoning block that quotes the schema
    # leaves earlier objects behind, and the answer comes last.
    cand = objs[-1]
    if "proposal" not in cand:
        with_prop = [o for o in objs if "proposal" in o]
        if not with_prop:
            raise ValueError("no object with a `proposal` key")
        cand = with_prop[-1]
    return cand


def record(store: dict, cand: dict, measured: dict, admitted: bool,
           note: str = "", reasoning: str = "", model: str = "",
           prediction_score: dict | None = None) -> None:
    """Store the candidate with its reasoning and what judged it.

    The reasoning is kept for admitted candidates too. Previously only
    malformed ones kept their raw text, so the record of an admitted card
    held its conclusion and not its argument -- and the argument is the part
    a person has to check, because measurement only confirms that the field
    bundle scored higher, never that the stated cause is the real one.
    """
    entry = dict(cand)
    if reasoning:
        entry["reasoning"] = reasoning[:4000]
    if model:
        entry["model"] = model
    if prediction_score:
        entry["prediction_score"] = prediction_score
    entry["measured"] = measured
    entry["note"] = note
    entry["recorded"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    store.setdefault("admitted" if admitted else "rejected", []).append(entry)


def measured_block(delta: float, base: float, after: float, n: int,
                   dataset: str, trigger_state: dict | None = None,
                   ci: tuple | None = None, k_tested: int = 1) -> dict:
    """`base` is the score of the TRIGGER STATE, not of theta_0."""
    out = {"delta": round(delta, 4), "base": round(base, 4),
           "after": round(after, 4), "n": n, "dataset": dataset,
           "fold": "train-dev", "date": date.today().isoformat()}
    if trigger_state is not None:
        out["measured_from"] = trigger_state
    if ci is not None:
        out["ci95"] = [round(ci[0], 4), round(ci[1], 4)]
        out["k_tested"] = k_tested
    return out
