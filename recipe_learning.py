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
#     the party-strip view's dense half may contribute nothing after fusion
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
ENUMS = {
    "fusion_mode": ("keyword_only", "semantic_only", "structured_only",
                    "first_found", "weighted_sum", "rrf"),
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
REPRESENTATIVE = {
    "semantic_top_k": 100, "keyword_top_k": 100, "structured_top_k": 10,
    "entity_swap_keyword_k": 30, "weight_semantic": 0.3,
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
NAME_MAX = 64
# Containment, not Jaccard. Two cards that agreed on 4 of their 5 fields
# scored identically (0.3924 both) and still passed a Jaccard check, because
# the union counted the one field each had to itself. Dividing by the SMALLER
# proposal catches "the same card with one extra field".
DUP_CONTAINMENT = 0.8
DEV_FRACTION = 0.2
ADMIT_THRESHOLD = 0.003  # the same bar elitism uses for a round

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


def admit(deltas: list[float], after: float, theta0: float,
          n_tested: int = 1) -> tuple[bool, str]:
    """Three gates. Any one of them failing is a rejection.

    1. the CI lower bound clears zero, at a level corrected for how many
       candidates this session has tested against the same fold
    2. the mean clears the same bar a round has to clear
    3. the state it leads to beats theta_0 -- a card can pass 1 and 2 by
       improving on a branch nobody would stand on
    """
    import statistics as _st
    alpha = ALPHA / max(1, n_tested)
    lo, hi = bootstrap_ci(deltas, alpha=alpha)
    mean = _st.fmean(deltas) if deltas else 0.0
    detail = (f"delta={mean:+.4f} CI{100*(1-alpha):.1f}%="
              f"[{lo:+.4f}, {hi:+.4f}] (k={n_tested})")
    if lo <= 0:
        return False, f"CI includes 0; {detail}"
    if mean <= ADMIT_THRESHOLD:
        return False, f"below +{ADMIT_THRESHOLD}; {detail}"
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
    # ("reduce dense reliance" at top_k 50, 30, 30). A pool of near-copies
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


def as_recipes(store: dict | None = None) -> list[Recipe]:
    """The admitted candidates, as engine Recipe objects.

    `expected_lift_pp` is the measured delta, never a proposed one, and the
    description carries where it was measured so the diagnosis LLM can
    discount a thin measurement instead of reading a bare number as fact.
    """
    store = store or load_store()
    out = []
    for e in store.get("admitted", []):
        m = e.get("measured", {})
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


def apply_measured_lifts(dataset: str) -> tuple[int, int]:
    """Overwrite the hand-written `expected_lift_pp` with train measurements.

    The eight values in RECIPES_LEGAL were typed in from the sweep in
    METHODOLOGY 8, which ran on TEST. They are not inert: the diagnosis
    prompt renders them as "(expected +N pp)", so the leak has been steering
    every round. measure_recipes.py re-measures each card on TRAIN, from a
    state the card's own trigger accepts, with a paired bootstrap CI.

    A card whose CI includes zero gets 0.0 and says so. That is not a claim
    that the card is useless -- it is the refusal to show a number we cannot
    support, which is the same rule the learned recipes follow (section 21).

    The lift is per DATASET, because one number cannot be right for four
    corpora in three languages: the same lever flipped sign between query
    classes of a single dataset (section 15), let alone across them.

    Returns (cards updated, cards whose claim was zeroed).
    """
    from evolvemem import evolution_cookbook as CB
    try:
        with open(LIFTS, encoding="utf-8") as f:
            table = json.load(f)
    except (OSError, ValueError):
        return (0, 0)
    row = table.get(dataset)
    if not row:
        return (0, 0)
    n_set = n_zero = 0
    for r in CB.RECIPE_POOLS.get("legal_retrieval", []):
        m = row.get("cards", {}).get(r.name)
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
                + ("; interval includes 0, so no gain is claimed" if
                   m["claim_pp"] == 0.0 else "")
                + ". Any pp figures in the text above predate this and came "
                  "from a sweep on the TEST fold; this bracketed figure is "
                  "the only one measured on the fold being optimised.]")
        # Strip a provenance note from an earlier call so repeated installs
        # do not stack them.
        r.description = re.sub(r"\s*\[measured [^\]]*\]\s*$", "",
                               r.description) + prov
        n_set += 1
        n_zero += m["claim_pp"] == 0.0
    return (n_set, n_zero)


def install(extend: bool = True) -> int:
    """Put the learned recipes into the pool. Returns how many were added."""
    from evolvemem import evolution_cookbook as CB
    learned = as_recipes()
    base = CB.RECIPE_POOLS.get("legal_retrieval", []) if extend else []
    CB.RECIPE_POOLS["legal_retrieval"] = list(base) + learned
    return len(learned)


# ── proposal prompt ───────────────────────────────────────────────────────
PROMPT = """You extend a library of retrieval "recipes" for legal case retrieval.

A recipe is a bundle of configuration fields that is applied together when a
symptom is observed. It exists because the evolution loop may change only
{cap} fields per round, and a recipe counts as ONE of those slots however many
fields it carries -- so a recipe is worth proposing when several fields must
move TOGETHER to help, and worthless when one field would do.

Retrieval works by scoring a query against four views and fusing them:
  BM25 lexical | dense (bge-m3) | metadata (statute/charge overlap) | party-strip

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

KNOWLEDGE = """- BM25 alone scores 0.4040 on LeCaRDv2 train; the dense view alone 0.2814.
  The dense view helps only at weight ~0.3 and degrades monotonically above it.
- The metadata view never helped in five different formulations: sharing a
  statute means the same offence, not a similar case.
- RRF beats weighted-sum once a 4th view contributes; with one view it lost.
- Party-strip at k=30 beat k=100: shallower intervention won.
- Widening party-strip to also delete organisations and places lost on train
  in all three languages.
- Chapter filtering lost 8.3 pp: only 61% of relevant cases share the query's
  chapter.
- Per-query LLM steps (intent planning, decomposition) lost 8.8 pp AND break
  the property that retrieval makes no LLM call per query.
- Almost every gain in this domain came from intervening LESS, not more.
- The same lever can flip sign between query classes: dense at weight 0.3 was
  +0.0119 on one class and -0.3421 on another."""


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
    and only above `min_uses`, because the dense view needs score fusion and
    banning `fusion_mode` early would make most useful bundles unreachable.
    """
    from collections import Counter
    c = Counter(f for e in (store.get("admitted", []) + store.get("rejected", []))
                for f in (e.get("proposal") or {}))
    return [f for f, n in c.most_common(k) if n >= min_uses]


def build_prompt(config: dict, store: dict, cap: int = MAX_FIELDS,
                 cot: bool = True, banned: list[str] | None = None) -> str:
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
        cap=cap,
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
