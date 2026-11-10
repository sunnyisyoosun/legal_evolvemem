"""Recipes built as axes x directions, instead of hand-written one at a time.

Why this exists
---------------
The hand-written pool (`RECIPES_LEGAL`, 8 cards) moves in ONE direction. Read
off the proposals:

  semantic_top_k        only ever -> 100       never -> 0
  enable_entity_swap    only ever -> True      never -> False
  enable_query_rewrite  only ever -> True      never -> False
  fusion_mode           keyword_only -> ws -> rrf, never back
  weight_semantic       0.3 / 0.3 / 0.2        never 0, never above 0.3

The four `undo_*` cards look like the other direction, but three of the five
fields they target are already off at theta_0, so they are dead on arrival:
they exist to clean up a move the LLM made with raw fields, and they cannot
walk back a move a recipe itself made. The pool is a one-way ratchet on
exactly the levers theta_0 can reach.

That is not a cosmetic gap. METHODOLOGY 15 measured the same lever
(`semantic_top_k=100` + `weight_semantic=0.3`) at +0.0119 on A2 and -0.3421 on
A3. A3 needed semantic turned OFF, and no card in the pool can say that. The
loop's only recovery was elitist revert, which restores the whole previous
config rather than the one lever -- and once reverted, the same card becomes
eligible again and can propose the identical harmful move. That is the
mechanism behind the repeat proposals that ate 29% of the round budget.

The construction
----------------
Each tunable is an AXIS with ordered LEVELS, from "not intervening" upward.
A card is one step along one axis, and every step gets a card in BOTH
directions. The pool is then generated, not written:

    axis "semantic": off -> light -> heavy
      semantic_off_to_light      (up)
      semantic_light_to_off      (down)
      semantic_light_to_heavy    (up)
      semantic_heavy_to_light    (down)

Three properties fall out of the construction rather than being maintained
by hand:

1. **Symmetry** — every move has its inverse. A class that needs a lever off
   can get there by the same machinery that turned it on.
2. **Bounded exposure** — a card's trigger requires the config to be AT its
   source level, so at most 2 cards per axis can fire, and prerequisites
   prune most axes entirely (nothing that needs score fusion is offered
   while fusion is off).
3. **Declaration order is dependency order** — axes are declared in the order
   they must be built, and `match_recipes` preserves declaration order
   (sorting by expected lift is what made LoCoMo pick moves out of sequence).

Honesty about the numbers
-------------------------
`expected_lift_pp` is a measurement we made, shown to the diagnosis LLM as
fact. Steps we have not measured carry 0.0 and say "unmeasured" in the
description, because a fabricated number here is the same failure as the
silent signals in METHODOLOGY 13: the LLM cannot tell an invented expectation
from a measured one.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "SimpleMem", "EvolveMem"))
from evolvemem.evolution_cookbook import Recipe, _consec_noaccept  # noqa: E402

# How many rounds without an accept counts as stalled, and therefore as the
# moment to offer exploration. Same number the hand-written escape card used.
STALL_ROUNDS = 2


@dataclass
class Level:
    """One rung: a name, the fields that put the config on this rung, and the
    probe that recognises a config as already being here."""
    name: str
    fields: dict
    probe: callable


@dataclass
class Axis:
    key: str
    levels: list          # ordered, index 0 = least intervention
    requires: callable = lambda c: True
    requires_note: str = ""
    # (from_index, to_index) -> (expected_pp, note). Absent = unmeasured.
    measured: dict = None


def _lvl(cfg, axis: Axis) -> int:
    """Which rung the config is on; -1 when it matches none of them."""
    for i, lv in enumerate(axis.levels):
        if lv.probe(cfg):
            return i
    return -1


# ── The axes ──────────────────────────────────────────────────────────────
# Declaration order IS dependency order: fusion first, because every view
# below it needs score fusion to exist before it can contribute anything.

AXES: list[Axis] = [
    # One axis short of the obvious design. Splitting "turn on score fusion"
    # from "turn on the dense view" into two rungs would cost two rounds to
    # reach a state one hand-written recipe reached in one -- re-creating the
    # step-cap wall the recipes were built to get through (METHODOLOGY 9).
    # So the dense view OWNS the fusion switch: its `light` rung turns fusion
    # on, its `off` rung turns it back off. This axis is only the choice
    # BETWEEN fusion modes, which is meaningful once a second view exists.
    Axis(
        key="fusion",
        levels=[
            Level("weighted", {"fusion_mode": "weighted_sum"},
                  lambda c: c.get("fusion_mode") == "weighted_sum"),
            Level("rrf", {"fusion_mode": "rrf"},
                  lambda c: c.get("fusion_mode") == "rrf"),
        ],
        requires=lambda c: c.get("semantic_top_k", 0) > 0,
        requires_note="needs a second view to fuse",
        measured={
            (0, 1): (1.5, "RRF beats weighted-sum once a 4th view exists "
                          "(LeCaRDv2 +0.0154, LeCaRD +0.0141, COLIEE +0.0186); "
                          "with a single view RRF measured -0.010"),
            (1, 0): (0.0, "unmeasured as a reversal; weighted-sum has to "
                          "reconcile BM25 (thousands) with cosine (0-1) under "
                          "one max-normalisation, which RRF avoids"),
        },
    ),
    Axis(
        key="semantic",
        levels=[
            Level("off", {"semantic_top_k": 0, "fusion_mode": "keyword_only"},
                  lambda c: c.get("semantic_top_k", 0) == 0),
            Level("light", {"semantic_top_k": 100, "weight_semantic": 0.3,
                            "keyword_top_k": 100, "weight_keyword": 1.0,
                            "fusion_mode": "weighted_sum"},
                  lambda c: c.get("semantic_top_k", 0) > 0
                  and c.get("weight_semantic", 1.0) <= 0.35),
            Level("heavy", {"semantic_top_k": 100, "weight_semantic": 0.6},
                  lambda c: c.get("semantic_top_k", 0) > 0
                  and c.get("weight_semantic", 1.0) > 0.35),
        ],
        measured={
            (0, 1): (0.1, "dense alone scores far below BM25 on case law "
                          "(0.28 vs 0.40); it helps only at weight ~0.3"),
            (1, 2): (0.0, "measured NEGATIVE: the dense view degrades "
                          "monotonically above weight 0.3"),
            (1, 0): (0.0, "unmeasured globally, but per-class it was large: "
                          "A3 lost -0.3421 from the light level, so turning "
                          "the view off is the right move for some classes"),
        },
    ),
    Axis(
        key="swap",
        levels=[
            Level("off", {"enable_entity_swap": False},
                  lambda c: not c.get("enable_entity_swap")),
            Level("shallow", {"enable_entity_swap": True,
                              "entity_swap_keyword_k": 15,
                              "entity_swap_semantic_k": 15},
                  lambda c: bool(c.get("enable_entity_swap"))
                  and c.get("entity_swap_keyword_k", 0) <= 20),
            Level("deep", {"enable_entity_swap": True,
                           "entity_swap_keyword_k": 30,
                           "entity_swap_semantic_k": 30},
                  lambda c: bool(c.get("enable_entity_swap"))
                  and c.get("entity_swap_keyword_k", 0) > 20),
        ],
        requires=lambda c: c.get("semantic_top_k", 0) > 0,
        requires_note="needs the dense view, which it re-searches with",
        measured={
            (0, 1): (1.4, "party names are incidental to whether two cases "
                          "are alike; k=30 measured +1.1 pp, k=100 only +0.4"),
            (1, 2): (0.0, "unmeasured as a step; k=30 was the best single "
                          "value but was never compared against k=15 directly"),
            (2, 1): (0.3, "the escape direction: every measured gain in this "
                          "domain came from intervening LESS"),
        },
    ),
    Axis(
        key="metadata",
        levels=[
            Level("off", {"structured_top_k": 0},
                  lambda c: c.get("structured_top_k", 0) == 0),
            Level("light", {"structured_top_k": 10, "weight_structured": 0.1},
                  lambda c: c.get("structured_top_k", 0) > 0
                  and c.get("weight_structured", 0.5) <= 0.2),
            Level("heavy", {"structured_top_k": 30, "weight_structured": 0.5},
                  lambda c: c.get("structured_top_k", 0) > 0
                  and c.get("weight_structured", 0.5) > 0.2),
        ],
        requires=lambda c: c.get("fusion_mode") not in ("keyword_only", "first_found"),
        requires_note="needs score fusion",
        measured={
            (1, 0): (9.0, "statute and charge matching was measured five ways "
                          "and never helped; turning the view off is worth "
                          "+9.0 pp from the heavy level"),
            (2, 1): (2.4, "every auxiliary signal here degrades monotonically "
                          "with weight under weighted-sum fusion"),
            (0, 1): (0.0, "measured NEGATIVE; offered only so the loop can "
                          "test it against a class that might differ"),
        },
    ),
    Axis(
        key="rewrite",
        levels=[
            Level("off", {"enable_query_rewrite": False},
                  lambda c: not c.get("enable_query_rewrite")),
            Level("once", {"enable_query_rewrite": True, "query_rewrite_repeat": 1},
                  lambda c: bool(c.get("enable_query_rewrite"))
                  and c.get("query_rewrite_repeat", 1) <= 1),
            Level("twice", {"enable_query_rewrite": True, "query_rewrite_repeat": 2},
                  lambda c: bool(c.get("enable_query_rewrite"))
                  and c.get("query_rewrite_repeat", 1) >= 2),
        ],
        requires=lambda c: c.get("fusion_mode") not in ("keyword_only", "first_found"),
        requires_note="needs score fusion",
        measured={
            (0, 1): (1.3, "appending an LLM rewrite once measured +1.3 pp"),
            (1, 2): (0.0, "measured NEGATIVE: twice fell to +0.0, four times "
                          "to -3.4, and replacing the original -17.0"),
            (2, 1): (1.3, "back to one append, the measured optimum"),
        },
    ),
    Axis(
        key="chapter_filter",
        levels=[
            Level("off", {"enable_chapter_filter": False},
                  lambda c: not c.get("enable_chapter_filter")),
            Level("on", {"enable_chapter_filter": True},
                  lambda c: bool(c.get("enable_chapter_filter"))),
        ],
        measured={
            (1, 0): (8.3, "restricting candidates to the query's chapter "
                          "measured -8.3 pp: only 61% of relevant cases share "
                          "the query's chapter"),
            (0, 1): (0.0, "measured NEGATIVE globally; offered so a class "
                          "whose chapter is more predictive can test it"),
        },
    ),
    Axis(
        key="llm_per_query",
        levels=[
            Level("off", {"enable_intent_planning": False,
                          "enable_query_decomposition": False,
                          "enable_coverage_reflection": False},
                  lambda c: not (c.get("enable_intent_planning")
                                 or c.get("enable_query_decomposition")
                                 or c.get("enable_coverage_reflection"))),
            Level("on", {"enable_intent_planning": True},
                  lambda c: bool(c.get("enable_intent_planning")
                                 or c.get("enable_query_decomposition")
                                 or c.get("enable_coverage_reflection"))),
        ],
        measured={
            (1, 0): (8.8, "a fact pattern has no multi-hop structure to split; "
                          "intent planning measured -8.8 pp"),
            (0, 1): (0.0, "measured NEGATIVE, and it calls the LLM once per "
                          "QUERY, which breaks the 0-LLM-calls-per-query "
                          "property of the retrieval loop"),
        },
    ),
    Axis(
        key="diversity",
        levels=[
            Level("off", {"mmr_diversity_weight": 0.0},
                  lambda c: not c.get("mmr_diversity_weight")),
            Level("on", {"mmr_diversity_weight": 0.3},
                  lambda c: bool(c.get("mmr_diversity_weight"))),
        ],
        requires=lambda c: c.get("fusion_mode") not in ("keyword_only", "first_found"),
        requires_note="needs score fusion",
        measured={},   # both directions unmeasured
    ),
]


def _reverses_last_accept(proposal: dict, history: list) -> bool:
    """True when this card would undo the move that was just accepted.

    Symmetry buys the loop a way back, and it also buys it a way to
    oscillate: the hand-written pool already did this once, with
    `entity_swap_shallow` raising k to 30 and `escape_plateau` lowering it to
    15 on alternate rounds. A step is therefore withheld when the previous
    round's ACCEPTED change set moved any of the same fields -- the loop has
    to spend a round elsewhere before reversing a fresh win.
    """
    for att in reversed(history or []):
        if att.get("llm_status") not in (None, "ok"):
            continue          # not a real measurement (METHODOLOGY 13)
        if not att.get("accepted"):
            return False      # last real round was a rejection: reversing is fine
        prev = att.get("changed_fields") or att.get("proposal") or {}
        return bool(set(proposal) & set(prev))
    return False


def _is_exploitation(pp: float) -> bool:
    """A card we measured as positive is exploitation; everything else is
    exploration.

    Symmetry alone over-exposes. Generated both ways, 26 cards put a mean of
    7.8 options in front of the LLM per round (max 11), against 0-2 for the
    hand-written pool -- and most of the extra were moves we had already
    MEASURED as harmful (turn the chapter filter on, raise the dense weight
    past 0.3). Offering a known-bad move every round is not exploration, it
    is noise in the choice, and METHODOLOGY 9 records what a scattered choice
    did on LoCoMo.

    So the two tiers are split by what we know, not by direction:

      measured positive  -> offered every round
      negative/unmeasured -> offered only once the loop has stalled

    This also removes the bias METHODOLOGY 15 found in the hand-written pool,
    where the one escape card required a deep swap to be already on, so it
    opened only for classes that had already made progress. Here a stall
    opens exploration for every class, whatever its history.
    """
    return pp > 0.0


def _card(axis: Axis, i: int, j: int) -> Recipe:
    src, dst = axis.levels[i], axis.levels[j]
    pp, note = (axis.measured or {}).get((i, j), (0.0, "unmeasured"))
    direction = "deeper" if j > i else "lighter"
    fields = dict(dst.fields)

    exploit = _is_exploitation(pp)

    def trigger(r, c, h, axis=axis, i=i, fields=fields, exploit=exploit):
        if _lvl(c, axis) != i:
            return False
        if not axis.requires(c):
            return False
        if not exploit and _consec_noaccept(h or []) < STALL_ROUNDS:
            return False
        return not _reverses_last_accept(fields, h)

    return Recipe(
        name=f"{axis.key}_{src.name}_to_{dst.name}",
        description=(
            f"Move the {axis.key} axis one step {direction}: "
            f"{src.name} -> {dst.name}. {note}."
            + ("" if exploit else
               " EXPLORATION: this step is not a measured win, and is offered "
               "because the loop has stalled.")
            + (f" Offered only when it {axis.requires_note}."
               if axis.requires_note else "")
        ),
        trigger=trigger,
        proposal=fields,
        expected_lift_pp=pp,
        benchmark="legal_retrieval",
    )


def build_pool() -> list[Recipe]:
    """Every adjacent step on every axis, both directions, in axis order."""
    out = []
    for axis in AXES:
        for i in range(len(axis.levels) - 1):
            out.append(_card(axis, i, i + 1))
            out.append(_card(axis, i + 1, i))
    return out


LADDER_POOL = build_pool()


def install() -> int:
    """Swap the ladder pool in for the hand-written one. Returns its size."""
    from evolvemem import evolution_cookbook as CB
    CB.RECIPE_POOLS["legal_retrieval"] = LADDER_POOL
    return len(LADDER_POOL)
