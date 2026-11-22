"""Diagnosis prompts, one per corpus language.

Why this module exists
----------------------
The diagnosis LLM is shown evidence in the corpus language: Chinese
judgments (LeCaRD, LeCaRDv2), Korean judgments (LEGAR_stricter), English
civil-code articles (COLIEE). Until 2026-11-11 the *instructions* around
that evidence were English for all four, so on three benchmarks out of four
the model reasoned across a language boundary between the task description
and the data. `DATASET_PROMPT` below pairs each dataset with a prompt in its
own language.

What is held constant
---------------------
All three variants carry the SAME content in the same order -- the same
action space, the same step cap, the same rubric, the same output schema --
so the only variable across datasets is the language of the wording. They
also keep every identifier in English: RetrievalConfig field names, enum
values such as "rrf", recipe names, and every JSON key. Those are code, and
`recipe_learning.validate()` drops a proposal whose field name it cannot
match; a translated key would read as a failed idea instead of an invalid
one.

How this differs from the engine's built-in prompt
--------------------------------------------------
The vendored `DIAGNOSIS_PROMPT` is LoCoMo's. `legal_evolve.INERT_DIMENSIONS`
strips its dead lines at runtime, but a line-level filter leaves residue:
"1. **Adapter prompt-surface flags** (+5-30 pp each, pick the ones whose" has
no identifier in it, so it survives with its list gone, and rubric item 9
still tells the LLM that LoCoMo prompt-surface flags are its highest-ROI
lever when no such flag is in the action space. The variants here are written
from scratch instead, which means:

  * English is NOT byte-identical to what ran before 2026-11-11. COLIEE's
    numbers move too, and the comparison against that baseline measures
    "our prompt vs LoCoMo's filtered prompt", not language.
  * Only the 12 fields in `recipe_learning.EFFECTIVE_FIELDS` are documented.
    The engine already filters `current_config` to the action space; showing
    descriptions of 13 further fields the validator will drop was the same
    leak from the other side.

Leak discipline (METHODOLOGY 10)
--------------------------------
No measured optimum appears in any variant. The `weight_semantic` answer key
(train optima 20x apart across the four benchmarks) is held out: a prompt
that named those values would be testing whether the LLM can copy, not
whether it can diagnose.
"""

from legal_meta import LANG_OF

# ── shared skeleton ───────────────────────────────────────────────────────
# Written once per language rather than templated from English, because a
# mechanically substituted prompt reads as a translation and we want each to
# read as instructions. The section order and the information content are
# identical by construction; `diagnosis.check_prompt_variants()` enforces
# that the placeholder set matches.

PROMPT_EN = """You are the diagnosis engine of a self-evolving legal case retrieval system. Your job is to turn this round's retrieval failures into a concrete configuration change for the next round.

The task: given a legal query (a fact pattern, or a statutory question), rank candidate cases or articles. The metric is NDCG@30. There is no answer generation step -- nothing here is about phrasing an answer, only about which documents come back and in what order.

## System Info
- Benchmark: {benchmark}
- Indexed documents: {total_memories}
- Queries this round: {total_questions}
- Score (NDCG@30): {overall_f1:.4f}
- Queries scoring zero: {zero_count}/{total_questions}
- Current config (JSON): {current_config}

## Failure Analysis Summary
{failure_summary}

## Per-Category Breakdown
{category_breakdown}

## Sample Failures (worst cases)
{sample_failures}

{extra_context}
{todo_checklist}
## Available Recipes (triggered by current symptoms)

A recipe is a bundle of config fields that must move TOGETHER for the
mechanism it names to work at all. Each recipe carries one mechanism, and
it carries structure only -- which fields are coupled -- not magnitudes.
The numbers are yours to choose: read the data above and decide them.

Invoking a recipe counts as ONE move no matter how many fields it sets.
Only the recipes whose trigger matches the current state are listed:

{available_recipes}

To invoke one, include `"use_recipe": "<recipe_name>"` in your JSON. You may
propose up to 1 further field alongside it (total budget = 2 moves). If no
listed recipe fits the failures you see, propose your own fields instead.

A recipe is a hypothesis, not a guarantee. A rejected invocation appears as
REJECTED in Prior Attempts like any move; do not re-invoke a recipe whose
identical application is already listed as REJECTED.

## Prior Attempts (most recent last)
{attempt_history}

## STEP-SIZE CONSTRAINT (HARD RULE)

Strict hill-climbing: **at most 2 changes per round**. A round is accepted
only if it beats the incumbent best by more than the acceptance threshold;
otherwise the state reverts and your move is recorded as REJECTED.

Propose AT MOST 2 fields in `parameter_suggestions` (one
`per_category_overrides` entry counts as 1 change however many sub-fields it
holds). Anything beyond the first 2 is truncated.

Never re-propose an exact (field, value) pair listed as REJECTED. That move
has been measured and did not help. Pick a different field, or a different
value for the same field -- a rejected value does not rule out the field.

## ANTI-PLATEAU RULE (apply after 2+ consecutive rejections)

If the last 2+ rounds are REJECTED, the obvious single-field moves are
exhausted and another tweak to an already-explored dimension will reject
too. Break out with a COUPLED move: two fields in one round that
only pay off together, because the intermediate states where one has moved
and the other has not are worse than both ends -- which is why no sequence
of single-field rounds can reach them. An unused recipe whose trigger
matches is one such move ready-made; the pool is small, so when none is
left, construct the pair yourself.

## PRIORITY ORDER

With 2 slots, spend them on the fields that decide whether a whole
retrieval view helps or hurts, and list those FIRST in
`parameter_suggestions` -- later entries are the ones truncation drops.

<<PRIORITY_ORDER>>

A round that moves only one scalar by a small step rarely clears the
acceptance threshold. Prefer a move whose mechanism you can name from the
Sample Failures above.

## Available Action Space

These are the only proposable fields. Each has been MEASURED to change the
ranking in this pipeline; fields not listed here are fixed at their defaults
and a proposal naming one is dropped.

<<ACTION_SPACE>>

## Decision Rubric

1. Many queries scoring zero -> recall problem. The relevant document is not
   in the candidate set at all: raise a view's depth, or turn on a view that
   is off, or widen fusion from a single view.
2. Score is non-zero but low -> ordering problem. The relevant documents are
   retrieved but ranked below noise: adjust weights, or rerank.
3. One category far below the others -> use `per_category_overrides` rather
   than a global change, so the gains elsewhere survive.
4. A lever that is ON but reported as never acting (see the levers section
   above, when present) is not evidence against the mechanism -- it did not
   run. Fix the condition that gates it, or stop spending slots on it.
5. Prefer turning on something that is off over nudging an int that is
   already in a reasonable range.
6. If the breakdown shows a category whose queries are long fact patterns
   while another's are short statutory questions, the same depth is unlikely
   to suit both; that is what `per_category_overrides` is for.

## Output

Return JSON. `parameter_suggestions` is a flat dict of field -> new value,
with field names matching the action space above EXACTLY.

```json
{{
  "root_causes": {{
    "recall_miss": {{"count": N, "description": "..."}},
    "ordering_error": {{"count": N, "description": "..."}}
  }},
  "parameter_suggestions": {{
    "fusion_mode": "rrf",
    "weight_semantic": 0.5
  }},
  "per_category_proposals": {{
    "3": {{"structured_top_k": 0}}
  }},
  "priority_actions": ["action1", "action2"]
}}
```

Return ONLY JSON."""


PROMPT_ZH = """你是一个自我演化的法律案例检索系统的诊断引擎。你的任务是把本轮的检索失败转化为下一轮的具体配置修改。

任务说明：给定一个法律查询（案件事实描述，或法条问题），对候选案例或法条进行排序。评价指标是 NDCG@30。本系统没有答案生成环节——这里的一切都与"措辞"无关，只关乎哪些文书被召回、以什么顺序排列。

## 系统信息
- 基准数据集：{benchmark}
- 已建索引文书数：{total_memories}
- 本轮查询数：{total_questions}
- 得分（NDCG@30）：{overall_f1:.4f}
- 得分为零的查询：{zero_count}/{total_questions}
- 当前配置（JSON）：{current_config}

## 失败分析汇总
{failure_summary}

## 分类别细分
{category_breakdown}

## 失败样例（最差情况）
{sample_failures}

{extra_context}
{todo_checklist}
## 可用配方（由当前症状触发）

一个"配方"是一组必须**同时**改动的配置字段，否则它所代表的机制根本不会生效。每个配方只承载一个机制，而且只承载结构——哪些字段相互耦合——不承载数值。数值由你来定：阅读上面的数据，自己判断。

调用一个配方算作**一次**改动，无论它设置了多少个字段。下面只列出触发条件与当前状态匹配的配方：

{available_recipes}

调用方式：在你的 JSON 中写入 `"use_recipe": "<recipe_name>"`。你可以在配方之外再追加最多 1 个字段（总预算 = 2 次改动）。如果列出的配方都不符合你看到的失败模式，就直接提出你自己的字段。

配方是假设，不是保证。被拒绝的调用会和任何改动一样，在"历史尝试"中标记为 REJECTED；不要重复调用一个完全相同的、已被标记为 REJECTED 的配方。

## 历史尝试（最新的在最后）
{attempt_history}

## 步长约束（硬性规则）

严格爬山法：**每轮最多 2 处改动**。只有当本轮得分超过当前最优解且超出接受阈值时才会被接受；否则状态回滚，你的改动被记为 REJECTED。

在 `parameter_suggestions` 中最多提出 2 个字段（一个 `per_category_overrides` 条目无论包含多少子字段都只算 1 次改动）。超出前 2 个的部分会被截断。

绝不要重复提出在历史尝试中已标记为 REJECTED 的（字段，取值）组合——那个改动已经被测量过，没有帮助。换一个字段，或者为同一字段换一个取值：某个取值被拒绝并不排除这个字段本身。

## 反平台期规则（连续 2 轮以上被拒后适用）

如果最近 2 轮以上都是 REJECTED，说明显而易见的单字段改动已经穷尽，再去微调一个已探索过的维度同样会被拒绝。此时应当用一个**耦合**的改动脱离平台期：一轮里改两个只有同时改才有收益的字段。只改了其中一个的中间状态比两端都更差，这正是任何单字段轮次的序列都到不了那里的原因。触发条件匹配且尚未使用过的配方就是一个现成的耦合改动；卡池很小，所以当没有剩下时，就自己把这一对配出来。

## 优先顺序

只有 2 个名额时，应当花在决定"某个检索视图整体上是帮助还是损害"的字段上，并把这些字段写在 `parameter_suggestions` 的**最前面**——被截断丢掉的是靠后的条目。

<<PRIORITY_ORDER>>

只把某个标量小幅移动一步的轮次很少能通过接受阈值。优先选择你能从上面的失败样例中说出其机制的改动。

## 可用动作空间

以下是唯一可提出的字段。每一个都已被**实测**会改变本流水线的排序结果；未列出的字段固定为默认值，提到它们的提案会被丢弃。

<<ACTION_SPACE>>

## 判断准则

1. 大量查询得分为零 → 召回问题。相关文书根本没有进入候选集：提高某个视图的深度，或打开一个关闭的视图，或把融合从单一视图扩展开。
2. 得分非零但偏低 → 排序问题。相关文书被召回了，但排在噪声之下：调整权重，或做重排。
3. 某一类别远低于其他类别 → 使用 `per_category_overrides` 而不是全局改动，以保住别处的收益。
4. 某个开关处于 ON 但被报告为"从未实际生效"（见上方相关小节，如果存在）——这不是否定该机制的证据，它根本没有运行过。要么修正门控它的条件，要么别再为它花名额。
5. 优先打开一个处于关闭状态的机制，而不是去挪动一个已在合理范围内的整数。
6. 如果细分表显示某一类别的查询是长篇事实描述、而另一类别是简短的法条问题，那么同一个深度不太可能同时适合两者；这正是 `per_category_overrides` 的用途。

## 输出

返回 JSON。`parameter_suggestions` 是一个"字段 → 新取值"的扁平字典，字段名必须与上面动作空间中的写法**完全一致**。

```json
{{
  "root_causes": {{
    "recall_miss": {{"count": N, "description": "..."}},
    "ordering_error": {{"count": N, "description": "..."}}
  }},
  "parameter_suggestions": {{
    "fusion_mode": "rrf",
    "weight_semantic": 0.5
  }},
  "per_category_proposals": {{
    "3": {{"structured_top_k": 0}}
  }},
  "priority_actions": ["action1", "action2"]
}}
```

只返回 JSON。"""


PROMPT_KO = """당신은 스스로 진화하는 법률 판례 검색 시스템의 진단 엔진이다. 이번 라운드의 검색 실패를 다음 라운드의 구체적인 설정 변경으로 바꾸는 것이 당신의 일이다.

과제: 법률 질의(사실관계 서술 또는 법조문 질문)가 주어지면 후보 판례나 조문의 순위를 매긴다. 평가 지표는 NDCG@30이다. 이 시스템에는 답변 생성 단계가 없다. 답을 어떻게 서술할지는 여기서 다루지 않는다. 어떤 문서가 검색되어 어떤 순서로 놓이는지만 다룬다.

## 시스템 정보
- 벤치마크: {benchmark}
- 색인된 문서 수: {total_memories}
- 이번 라운드 질의 수: {total_questions}
- 점수(NDCG@30): {overall_f1:.4f}
- 0점 질의: {zero_count}/{total_questions}
- 현재 설정(JSON): {current_config}

## 실패 분석 요약
{failure_summary}

## 카테고리별 분해
{category_breakdown}

## 실패 사례(최악의 경우)
{sample_failures}

{extra_context}
{todo_checklist}
## 사용 가능한 레시피(현재 증상이 촉발한 것)

레시피는 **함께 움직여야만** 그것이 가리키는 메커니즘이 작동하는 설정 필드들의 묶음이다. 레시피 하나는 메커니즘 하나를 담으며, 담고 있는 것은 구조뿐이다 — 어떤 필드들이 서로 묶여 있는지. 크기는 담지 않는다. 숫자는 당신이 정한다. 위의 데이터를 읽고 직접 판단하라.

레시피를 호출하면 그것이 몇 개 필드를 설정하든 **한 번**의 변경으로 계산된다. 아래에는 촉발 조건이 현재 상태와 맞는 레시피만 나열된다:

{available_recipes}

호출 방법: JSON에 `"use_recipe": "<recipe_name>"`을 넣는다. 레시피와 함께 필드 1개를 더 제안할 수 있다(총 예산 = 2회 변경). 나열된 레시피 중 당신이 본 실패 양상에 맞는 것이 없으면 직접 필드를 제안하라.

레시피는 가설이지 보장이 아니다. 거부된 호출은 다른 모든 변경과 똑같이 이전 시도에 REJECTED로 남는다. 완전히 동일한 적용이 이미 REJECTED로 올라 있는 레시피를 다시 호출하지 마라.

## 이전 시도(최근 것이 마지막)
{attempt_history}

## 보폭 제약(강제 규칙)

엄격한 언덕 오르기: **한 라운드에 최대 2개 변경**. 현재 최고 상태를 수용 임계값 이상으로 이겨야만 받아들여지고, 그렇지 않으면 상태는 되돌려지며 당신의 변경은 REJECTED로 기록된다.

`parameter_suggestions`에 최대 2개 필드를 제안하라(`per_category_overrides` 항목 하나는 하위 필드가 몇 개든 1회 변경으로 센다). 앞의 2개를 넘는 것은 잘린다.

이전 시도에서 REJECTED로 표시된 (필드, 값) 조합을 다시 제안하지 마라. 그 변경은 이미 측정되었고 도움이 되지 않았다. 다른 필드를 고르거나, 같은 필드에 다른 값을 고르라. 값 하나가 거부된 것이 그 필드를 배제하지는 않는다.

## 정체 탈출 규칙(2회 이상 연속 거부 이후 적용)

최근 2라운드 이상이 REJECTED라면 뻔한 단일 필드 변경은 소진된 것이고, 이미 탐색한 차원을 또 손봐도 역시 거부된다. **결합된** 변경으로 정체를 벗어나라. 한 라운드에 두 필드를, 둘이 함께일 때만 값이 나오는 쌍으로 움직이는 것이다. 하나만 움직인 중간 상태가 양쪽 끝보다 더 나쁘고, 그래서 단일 필드 라운드를 어떻게 이어 붙여도 거기에 닿지 못한다. 촉발 조건이 맞고 아직 쓰지 않은 레시피가 그런 변경의 완성품이다. 묶음이 작으니 남은 것이 없을 때는 그 쌍을 직접 구성하라.

## 우선순위

자리가 2개뿐이므로, 검색 뷰 하나가 전체적으로 도움이 되는지 해가 되는지를 결정하는 필드에 써라. 그리고 그 필드를 `parameter_suggestions`의 **맨 앞**에 적어라. 잘려 나가는 것은 뒤쪽 항목이다.

<<PRIORITY_ORDER>>

스칼라 하나를 조금 움직이는 라운드는 수용 임계값을 넘기는 일이 드물다. 위의 실패 사례에서 그 메커니즘을 말로 짚을 수 있는 변경을 우선하라.

## 사용 가능한 행동 공간

제안할 수 있는 필드는 다음이 전부다. 각각은 이 파이프라인에서 순위를 실제로 바꾼다고 **실측된** 것이다. 여기 없는 필드는 기본값에 고정되어 있고, 그것을 지명한 제안은 버려진다.

<<ACTION_SPACE>>

## 판단 기준

1. 0점 질의가 많다 → 재현율 문제. 정답 문서가 애초에 후보 집합에 없다. 어떤 뷰의 깊이를 올리거나, 꺼진 뷰를 켜거나, 단일 뷰에서 융합을 넓혀라.
2. 0점은 아니지만 점수가 낮다 → 순서 문제. 정답 문서는 검색되었으나 잡음 아래에 놓여 있다. 가중치를 조정하거나 재순위를 걸어라.
3. 한 카테고리가 다른 것들보다 크게 낮다 → 전역 변경 대신 `per_category_overrides`를 써서 다른 곳의 이득을 지켜라.
4. ON이지만 "한 번도 실제로 작동하지 않았다"고 보고된 레버(위의 해당 절이 있다면 참조)는 그 메커니즘에 대한 반증이 아니다. 그것은 돌지 않았다. 그것을 막고 있는 조건을 고치거나, 더 이상 자리를 쓰지 마라.
5. 이미 합리적인 범위에 있는 정수를 밀기보다, 꺼져 있는 것을 켜는 쪽을 우선하라.
6. 분해표에서 어떤 카테고리의 질의는 긴 사실관계 서술이고 다른 카테고리는 짧은 법조문 질문이라면, 같은 깊이가 둘 모두에 맞을 가능성은 낮다. `per_category_overrides`가 바로 그것을 위한 것이다.

## 출력

JSON을 반환하라. `parameter_suggestions`는 "필드 → 새 값"의 평평한 딕셔너리이며, 필드명은 위 행동 공간의 표기와 **정확히** 일치해야 한다.

```json
{{
  "root_causes": {{
    "recall_miss": {{"count": N, "description": "..."}},
    "ordering_error": {{"count": N, "description": "..."}}
  }},
  "parameter_suggestions": {{
    "fusion_mode": "rrf",
    "weight_semantic": 0.5
  }},
  "per_category_proposals": {{
    "3": {{"structured_top_k": 0}}
  }},
  "priority_actions": ["action1", "action2"]
}}
```

JSON만 반환하라."""


PROMPTS = {"en": PROMPT_EN, "zh": PROMPT_ZH, "ko": PROMPT_KO}

# Dataset -> prompt language. Reuses legal_meta.LANG_OF rather than a second
# mapping, so a new benchmark cannot be added with its corpus language and
# its prompt language disagreeing.
DATASET_PROMPT = {ds: PROMPTS[lang] for ds, lang in LANG_OF.items()}


def install(dataset: str, diagnosis_module, fields=None, tiers=None) -> str:
    """Register all three variants and select the one for `dataset`.

    `fields` is the action space for THIS dataset. Each variant has its
    `<<ACTION_SPACE>>` marker replaced with those fields' descriptions before
    it is registered, so the prompt describes exactly what the validator will
    accept. Passing None documents every EFFECTIVE_FIELD, which is right only
    when nothing was dropped.

    All three languages are registered, not just the selected one, so
    `check_prompt_variants()` validates every variant on every run -- a
    broken Korean prompt is caught during a Chinese run instead of three
    weeks later.
    """
    lang = LANG_OF[dataset]
    if fields is None:
        import recipe_learning as _RL
        fields = list(_RL.EFFECTIVE_FIELDS)
    check_field_docs()
    for lg, tpl in PROMPTS.items():
        rendered = tpl.replace(ACTION_SPACE_MARKER,
                               render_action_space(fields, lg))
        rendered = rendered.replace(
            PRIORITY_MARKER, render_priority(tiers or {}, fields, lg))
        for marker in (ACTION_SPACE_MARKER, PRIORITY_MARKER):
            if marker in rendered:
                raise ValueError(f"{lg}: {marker} not substituted")
        diagnosis_module.set_prompt_lang(lg, rendered)
    diagnosis_module.set_prompt_lang(lang)
    diagnosis_module.check_prompt_variants()
    set_block_lang(lang)
    check_blocks()
    return lang


# ── runtime blocks ────────────────────────────────────────────────────────
# `legal_evolve.view_report()` and `inert_report()` are injected into the
# prompt through {extra_context}. They are prose, and they are ours, so they
# follow the template's language. Without this the Chinese prompt explained
# its own per-view evidence table in English.
#
# Status tokens (OFF/on) and field names stay as they are: they name config
# state the LLM has to write back verbatim.
BLOCKS = {
    "view.header": {
        "en": "Across {n} queries, of the top-{k} documents each view helped "
              "rank (a document can be credited to several views):",
        "zh": "在 {n} 条查询中，前 {k} 名文书里各视图参与排序的情况"
              "（同一份文书可以同时归功于多个视图）：",
        "ko": "{n}개 질의에서, 상위 {k}개 문서 중 각 뷰가 순위에 기여한 몫"
              "(한 문서가 여러 뷰에 동시에 귀속될 수 있다):",
    },
    "view.cols": {
        "en": ("view", "status", "share", "relevant", "sole"),
        "zh": ("视图", "状态", "占比", "相关", "独有"),
        "ko": ("뷰", "상태", "비중", "관련", "단독"),
    },
    "view.off_note": {
        "en": "OFF means the view did not run under the current config -- it "
              "has no numbers here, which is NOT evidence that enabling it "
              "would not help.",
        "zh": "OFF 表示该视图在当前配置下没有运行——它在这里没有数字，"
              "这并不是\"打开它也不会有帮助\"的证据。",
        "ko": "OFF는 현재 설정에서 그 뷰가 돌지 않았다는 뜻이다. 여기에 숫자가 "
              "없다는 것이 켜도 도움이 안 된다는 증거는 아니다.",
    },
    # The semantic view can be ON and return nothing, for a reason no other
    # block can express: the query TEXT changed, so its precomputed vector
    # was not found. Printing "semantic: on, 0% share" without this line
    # tells the LLM the semantic view is useless -- the opposite of the truth,
    # and the semantic view is the only lever positive on all four datasets.
    "view.sem_miss": {
        "en": "!! The semantic view was asked for {miss} of {total} queries "
              "whose text had been MODIFIED, and returned nothing for them: "
              "vectors are precomputed per original query text, so a changed "
              "query has none. `enable_query_rewrite` changes it, and the "
              "party-strip view always does. Any number for the semantic view "
              "in this round is measured with it partly or wholly dark, and "
              "a low share for it is NOT evidence against it. Example of a "
              "query with no vector: {sample}",
        "zh": "!! 本轮有 {miss}/{total} 次对 semantic 视图的检索使用了**被修改过的**"
              "查询文本，并且全部返回空：向量是按原始查询文本预先计算的，"
              "改动过的查询没有对应向量。`enable_query_rewrite` 会改动它，"
              "party-strip 视图则总是改动它。因此本轮 semantic 视图的任何数字都是在"
              "它部分或完全熄灭的状态下测得的，它占比低**不是**否定它的证据。"
              "没有向量的查询示例：{sample}",
        "ko": "!! 이번 라운드에 semantic view 검색 {total}회 중 {miss}회가 **수정된** "
              "질의 텍스트로 들어와 전부 빈 결과를 돌려주었다. 벡터는 원래 질의 "
              "텍스트별로 미리 계산돼 있어서, 바뀐 질의에는 벡터가 없다. "
              "`enable_query_rewrite`가 텍스트를 바꾸고, party-strip view는 항상 "
              "바꾼다. 따라서 이번 라운드 semantic view의 어떤 숫자도 그것이 일부 또는 "
              "전부 꺼진 상태에서 측정된 것이고, 비중이 낮은 것은 그것에 대한 "
              "**반증이 아니다**. 벡터가 없는 질의 예: {sample}",
    },
    "view.sole_note": {
        "en": "`sole` = relevant documents this view alone surfaced, i.e. "
              "documents no other active view found. An active view with many "
              "hits but sole=0 is only re-ranking what another view already "
              "found; raising its weight or top_k will not reach new "
              "documents.",
        "zh": "\"独有\" = 只有这个视图找出来的相关文书，即其他活动视图都没有找到的文书。"
              "一个命中很多但独有=0 的活动视图，只是在给别的视图已经找到的东西重新排序；"
              "提高它的权重或 top_k 不会触及新的文书。",
        "ko": "'단독' = 이 뷰만이 찾아낸 관련 문서, 즉 활성인 다른 어떤 뷰도 찾지 못한 "
              "문서다. 적중은 많지만 단독=0인 활성 뷰는 다른 뷰가 이미 찾은 것을 다시 "
              "순서만 매기고 있을 뿐이다. 그 가중치나 top_k를 올려도 새 문서에 닿지 않는다.",
    },
    "inert.line": {
        "en": "- {name}: ON for {on} queries, acted on 0. {reason}. This field "
              "cannot change any score, so a delta of +0.0000 is NOT evidence "
              "against it. Do not conclude it is ineffective.",
        "zh": "- {name}：在 {on} 条查询上处于 ON，实际生效 0 条。{reason}。"
              "这个字段无法改变任何分数，所以 +0.0000 的差值**不是**否定它的证据。"
              "不要据此断定它无效。",
        "ko": "- {name}: {on}개 질의에서 ON, 실제로 작동한 것은 0개. {reason}. 이 필드는 "
              "어떤 점수도 바꿀 수 없으므로 +0.0000이라는 차이는 이 필드에 대한 반증이 "
              "아니다. 효과가 없다고 결론 내리지 마라.",
    },
    "fail.zero_by_cat": {
        "en": "Zero-scoring queries by category:",
        "zh": "按类别统计的 0 分查询：",
        "ko": "카테고리별 0점 질의:",
    },
    "fail.zero_row": {
        "en": "  {label}: {zero}/{total} scored zero",
        "zh": "  {label}：{total} 条中 {zero} 条得 0 分",
        "ko": "  {label}: {total}개 중 {zero}개가 0점",
    },
    "fail.low_band": {
        "en": "Queries scoring between 0 and 0.3 (retrieved something "
              "relevant, ranked it badly): {n}/{total}",
        "zh": "得分在 0 与 0.3 之间的查询（检索到了相关文书，但排得很差）：{n}/{total}",
        "ko": "0과 0.3 사이 점수의 질의(관련 문서를 찾았으나 순위가 나쁘다): {n}/{total}",
    },
    "fail.split": {
        "en": "Of the {z} zero-scoring queries: {absent} returned NO relevant "
              "document anywhere in the ranking (recall problem -- the "
              "document never entered the candidate set), {below} returned "
              "one but below rank {k} (ordering problem -- it was found and "
              "then outranked by noise).",
        "zh": "在 {z} 条 0 分查询中：{absent} 条在整个排序里**没有**返回任何相关文书"
              "（召回问题——该文书从未进入候选集），{below} 条返回了但排在第 {k} 名之后"
              "（排序问题——找到了，却被噪声压下去了）。",
        "ko": "0점 질의 {z}개 중: {absent}개는 순위 전체에서 관련 문서를 **하나도** "
              "돌려주지 않았다(재현율 문제 — 그 문서는 후보 집합에 들어온 적이 없다). "
              "{below}개는 돌려주었으나 {k}위 아래였다(순서 문제 — 찾았는데 잡음에 "
              "밀렸다).",
    },
    "fail.ranklen": {
        "en": "Ranking length returned per query: median {med}, min {lo}, max "
              "{hi}. Nothing deeper than that was returned, so a relevant "
              "document ranked past it cannot be seen from here at all.",
        "zh": "每条查询返回的排序长度：中位数 {med}，最小 {lo}，最大 {hi}。"
              "比这更深的结果没有被返回，所以排在它之后的相关文书在这里根本看不到。",
        "ko": "질의당 반환된 순위 길이: 중앙값 {med}, 최소 {lo}, 최대 {hi}. "
              "그보다 깊은 결과는 반환되지 않았으므로, 그 뒤에 놓인 관련 문서는 "
              "여기서 전혀 보이지 않는다.",
    },
    "cat.row": {
        "en": "  [{cat}] {label}: {metric}={mean:.3f}  n={n}",
        "zh": "  [{cat}] {label}：{metric}={mean:.3f}  n={n}",
        "ko": "  [{cat}] {label}: {metric}={mean:.3f}  n={n}",
    },
    "cat.key_note": {
        "en": "The number in brackets is the key to use in "
              "`per_category_overrides`; the name after it is what that class "
              "is.",
        "zh": "方括号里的数字就是 `per_category_overrides` 里要用的键；"
              "后面的名称说明该类别是什么。",
        "ko": "대괄호 안의 숫자가 `per_category_overrides`에 쓸 키이고, "
              "그 뒤의 이름이 그 분류가 무엇인지다.",
    },
    "cat.none": {
        "en": "(no category breakdown: no query carried a class label)",
        "zh": "（无类别细分：没有任何查询带类别标签）",
        "ko": "(카테고리 분해 없음: 어떤 질의도 분류 라벨을 갖고 있지 않다)",
    },
    # No raw ranked-id line. It listed up to 40 opaque ids, which duplicated
    # the ids already in "sample.retrieved" -- where each one arrives with
    # the view that surfaced it and a digest of the document -- and the gold
    # document's position is in the header. It was the largest piece of
    # non-information in the prompt.
    "sample.body": {
        "en": "    Query: {q}\n    Relevant: {ref}",
        "zh": "    查询：{q}\n    相关文书：{ref}",
        "ko": "    질의: {q}\n    관련 문서: {ref}",
    },
    "sample.head_miss": {
        "en": "  [{cat}] {label} -- scored {score:.3f}; no relevant document "
              "anywhere in the returned ranking",
        "zh": "  [{cat}] {label} —— 得分 {score:.3f}；返回的排序里没有任何相关文书",
        "ko": "  [{cat}] {label} — 점수 {score:.3f}; 반환된 순위 어디에도 관련 "
              "문서가 없다",
    },
    "sample.head_rank": {
        "en": "  [{cat}] {label} -- scored {score:.3f}; relevant document(s) "
              "came back at rank {ranks} of {n} returned",
        "zh": "  [{cat}] {label} —— 得分 {score:.3f}；相关文书落在返回的 {n} 条中的"
              "第 {ranks} 名",
        "ko": "  [{cat}] {label} — 점수 {score:.3f}; 관련 문서가 반환된 {n}건 중 "
              "{ranks}위에 있다",
    },
    "sample.strata": {
        "en": "Two kinds of failure are sampled below: queries that scored "
              "zero, and queries that scored above zero but poorly. The "
              "second kind is where a relevant document WAS retrieved and "
              "then outranked -- read the ranks in each header before "
              "deciding between depth and weighting.",
        "zh": "下面采样了两类失败：得 0 分的查询，以及得分高于 0 但很差的查询。"
              "第二类是相关文书**已经**被检索到、随后被压下去的情形——"
              "在决定调深度还是调权重之前，先读每条标题里的名次。",
        "ko": "아래에는 두 종류의 실패를 표집했다: 0점을 받은 질의, 그리고 0점보다는 "
              "높지만 나쁜 점수를 받은 질의. 두 번째가 관련 문서를 **이미** 찾아 놓고 "
              "밀린 경우다. 깊이와 가중치 중 무엇을 건드릴지 정하기 전에 각 머리말의 "
              "순위를 읽으라.",
    },
    "sample.retrieved": {
        "en": "    Top hits, with the view that surfaced each: {shown}",
        "zh": "    靠前的命中，以及把每一条带出来的视图：{shown}",
        "ko": "    상위 적중과 각각을 올린 뷰: {shown}",
    },
    "sample.none": {
        "en": "(no zero-scoring queries this round)",
        "zh": "（本轮没有 0 分查询）",
        "ko": "(이번 라운드에 0점 질의가 없다)",
    },
    "recipe.plain": {
        "en": "\u2022 `{name}`",
        "zh": "\u2022 `{name}`",
        "ko": "\u2022 `{name}`",
    },
    "recipe.measured": {
        "en": "\u2022 `{name}` ({src})",
        "zh": "\u2022 `{name}`（{src}）",
        "ko": "\u2022 `{name}` ({src})",
    },
    # Both figures, because they answer different questions and the big one
    # is not the one that matters. A card is measured from the state its own
    # trigger accepts, so its delta is CONDITIONAL: entity_swap_shallow
    # climbs +4.57 pp from its trigger base on LeCaRD train while the state
    # it lands in is only +0.61 pp above theta_0. Printing the conditional
    # figure alone overstated the card by 7.5x in the header the LLM reads.
    "recipe.src_gain": {
        "en": "{pp:+.2f} pp from the state this card's trigger accepts; that "
              "end state is {end:+.2f} pp vs theta_0 -- {ds} {split}, n={n}",
        "zh": "从本卡触发条件接受的状态出发 {pp:+.2f} pp；该终点相对 theta_0 为 "
              "{end:+.2f} pp —— {ds} {split}，n={n}",
        "ko": "이 카드의 촉발 조건이 받아들이는 상태에서 {pp:+.2f} pp, 그 종점은 "
              "theta_0 대비 {end:+.2f} pp — {ds} {split}, n={n}",
    },
    "recipe.src_zero": {
        "en": "no gain admitted on {ds} {split}, n={n} -- {why}",
        "zh": "在 {ds} {split}，n={n} 上未认定增益 —— {why}",
        "ko": "{ds} {split}, n={n}에서 이득 인정 안 됨 — {why}",
    },
    "recipe.zero_ci": {
        "en": "the confidence interval includes 0",
        "zh": "置信区间包含 0",
        "ko": "신뢰구간이 0을 포함한다",
    },
    "recipe.fields": {
        "en": "Fields set: {fields}",
        "zh": "设置的字段：{fields}",
        "ko": "설정하는 필드: {fields}",
    },
    "recipe.none": {
        "en": "(no recipe's trigger matches the current state this round)",
        "zh": "（本轮没有任何配方的触发条件与当前状态匹配）",
        "ko": "(이번 라운드에 촉발 조건이 현재 상태와 맞는 레시피가 없다)",
    },
    "budget.line": {
        "en": "### What this round has to clear, and what is left\n"
              "- Acceptance: a round is kept only if it beats the incumbent "
              "best by more than **{thr:.4f}** ({thr_pp:.2f} pp). That number "
              "is this fold's measured noise floor, not a preference: a "
              "smaller difference cannot be told apart from changing nothing. "
              "A proposal you expect to be worth less than that will be "
              "rejected however sound it is.\n"
              "- Rounds: this is round {round} of {total}. {left} left after "
              "this one.",
        "zh": "### 本轮必须跨过的门槛，以及还剩多少\n"
              "- 接受条件：只有比当前最优解高出 **{thr:.4f}**（{thr_pp:.2f} pp）"
              "以上的轮次才会被保留。这个数字是本折的**实测噪声下限**，不是偏好："
              "比它更小的差异无法与\"什么都不改\"区分开。你预计收益小于它的提案，"
              "无论多合理都会被拒绝。\n"
              "- 轮次：这是第 {round} 轮，共 {total} 轮。本轮之后还剩 {left} 轮。",
        "ko": "### 이번 라운드가 넘어야 하는 선, 그리고 남은 것\n"
              "- 수용 조건: 현재 최고 상태를 **{thr:.4f}**({thr_pp:.2f} pp) 넘게 "
              "이겨야만 라운드가 유지된다. 이 숫자는 이 폴드의 **실측 잡음 "
              "바닥**이고 취향이 아니다. 그보다 작은 차이는 아무것도 바꾸지 않은 "
              "것과 구분할 수 없다. 그보다 작은 이득을 예상하는 제안은 아무리 "
              "타당해도 거부된다.\n"
              "- 라운드: 전체 {total} 중 {round}번째다. 이번 라운드 뒤로 {left}개 "
              "남는다.",
    },
    "state.heading": {
        "en": "### Values in the current config that nobody chose",
        "zh": "### 当前配置中没有任何人选择过的取值",
        "ko": "### 현재 설정에서 아무도 고르지 않은 값",
    },
    # Deliberately direction-free. The four cards these notes replace each
    # proposed going DOWN, and on LEGAR_stricter the semantic weight's train
    # optimum is ABOVE the default -- so "undo the overweight" pointed the
    # wrong way on one dataset in four. A note that says only "this is a
    # default, and it differs across corpora" is correct everywhere.
    "state.unchosen": {
        "en": "- {field} is {value}, which is the library default. The {view} "
              "view is active, so this value is now deciding the ranking, and "
              "it is not the same value on every corpus. Read the per-category "
              "scores and set it yourself.",
        "zh": "- {field} 当前为 {value}，这是库的默认值。{view} 视图处于活动状态，"
              "所以这个取值正在决定排序，而它在每个语料上并不相同。"
              "请阅读分类别得分后自己设定。",
        "ko": "- {field}가 {value}인데, 이것은 라이브러리 기본값이다. {view} 뷰가 "
              "활성이므로 이 값이 지금 순위를 결정하고 있고, 코퍼스마다 같은 값이 "
              "아니다. 카테고리별 점수를 읽고 직접 정하라.",
    },
    "state.view_semantic": {"en": "semantic", "zh": "semantic（语义）",
                            "ko": "semantic(의미)"},
    "state.view_swap": {"en": "party-strip", "zh": "party-strip",
                        "ko": "party-strip"},
    "inert.heading": {
        "en": "### Levers that were ON but never acted",
        "zh": "### 处于 ON 但从未实际生效的开关",
        "ko": "### ON이었지만 한 번도 작동하지 않은 레버",
    },
    "inert.fallback": {
        "en": "it could not take effect",
        "zh": "它无法生效",
        "ko": "효력을 가질 수 없었다",
    },
    "inert.query_rewrite": {
        "en": "no rewrite exists for these queries (cache/rewrite/<dataset>"
              ".json missing or empty)",
        "zh": "这些查询没有改写本（cache/rewrite/<dataset>.json 缺失或为空）",
        "ko": "이 질의들에 대한 재작성본이 없다(cache/rewrite/<dataset>.json이 "
              "없거나 비어 있다)",
    },
    "inert.chapter_filter": {
        "en": "no query carries a legal_class, so the filter fell back to the "
              "unrestricted corpus every time",
        "zh": "没有任何查询带 legal_class，所以过滤每次都退回到未受限的全库",
        "ko": "어떤 질의도 legal_class를 갖고 있지 않아, 필터가 매번 제한 없는 "
              "전체 코퍼스로 되돌아갔다",
    },
    "inert.view:semantic": {
        "en": "semantic_top_k=0, so the semantic view returns no candidates. "
              "weight_semantic cannot change any score while top_k is 0 -- "
              "raise semantic_top_k first",
        "zh": "semantic_top_k=0，所以语义视图不返回任何候选。只要 top_k 是 0，"
              "weight_semantic 就改变不了任何分数——先把 semantic_top_k 提上去",
        "ko": "semantic_top_k=0이므로 의미 뷰는 후보를 전혀 돌려주지 않는다. top_k가 "
              "0인 동안 weight_semantic은 어떤 점수도 바꿀 수 없다. 먼저 "
              "semantic_top_k를 올려라",
    },
    "inert.view:metadata": {
        "en": "structured_top_k=0, so the metadata view returns no candidates. "
              "weight_structured cannot change any score while top_k is 0",
        "zh": "structured_top_k=0，所以元数据视图不返回任何候选。只要 top_k 是 0，"
              "weight_structured 就改变不了任何分数",
        "ko": "structured_top_k=0이므로 메타데이터 뷰는 후보를 전혀 돌려주지 않는다. "
              "top_k가 0인 동안 weight_structured는 어떤 점수도 바꿀 수 없다",
    },
    "inert.view:party-strip": {
        "en": "enable_entity_swap is on but its top_k is 0, so the party-strip "
              "view returns no candidates",
        "zh": "enable_entity_swap 已打开但它的 top_k 是 0，所以 party-strip 视图"
              "不返回任何候选",
        "ko": "enable_entity_swap는 켜져 있지만 그 top_k가 0이므로 party-strip 뷰는 "
              "후보를 전혀 돌려주지 않는다",
    },
    "inert.fusion_mode": {
        "en": "only ONE view is active, and every fusion mode reduces to that "
              "view's own ranking when there is nothing to fuse. rrf with a "
              "single view is arithmetically identical to keyword_only. "
              "Activate a second view before changing fusion_mode",
        "zh": "只有**一个**视图处于活动状态，而在无可融合时每种融合方式都退化为"
              "该视图自身的排序。单视图下的 rrf 在算术上与 keyword_only 完全相同。"
              "先激活第二个视图，再去改 fusion_mode",
        "ko": "활성인 뷰가 **하나**뿐이고, 융합할 것이 없을 때 모든 융합 방식은 그 뷰 "
              "자신의 순위로 환원된다. 뷰가 하나일 때의 rrf는 산술적으로 keyword_only와 "
              "완전히 같다. fusion_mode를 바꾸기 전에 두 번째 뷰를 활성화하라",
    },
}

_BLOCK_LANG = "en"


def set_block_lang(lang: str) -> None:
    global _BLOCK_LANG
    _BLOCK_LANG = lang if lang in ("en", "zh", "ko") else "en"


def t(key: str):
    """Runtime block text in the selected language, English as the fallback."""
    entry = BLOCKS[key]
    return entry.get(_BLOCK_LANG, entry["en"])


def check_blocks() -> None:
    """Raise unless every block is present in all three languages with the
    same {placeholders}. A block missing a placeholder would render the
    per-view evidence table without its counts."""
    import re as _re
    ph = _re.compile(r"\{([a-z_]+)\}")
    for key, entry in BLOCKS.items():
        if set(entry) != {"en", "zh", "ko"}:
            raise ValueError(f"BLOCKS[{key!r}] languages: {sorted(entry)}")
        if isinstance(entry["en"], tuple):
            n = len(entry["en"])
            for lg, v in entry.items():
                if len(v) != n:
                    raise ValueError(
                        f"BLOCKS[{key!r}][{lg!r}] has {len(v)} columns, "
                        f"English has {n}")
            continue
        base = set(ph.findall(entry["en"]))
        for lg, v in entry.items():
            if set(ph.findall(v)) != base:
                raise ValueError(
                    f"BLOCKS[{key!r}][{lg!r}] placeholders "
                    f"{sorted(set(ph.findall(v)))} != {sorted(base)}")


def pad(text: str, width: int, right: bool = False) -> str:
    """Pad to `width` DISPLAY columns, not characters.

    A CJK glyph occupies two terminal columns, so `f"{'视图':10s}"` emits ten
    characters that render twenty columns wide and the Chinese table header
    sat two columns off its data rows for every label. An LLM reading a
    misaligned table can attach a number to the wrong column, which is the
    one thing this block exists to prevent.
    """
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            for ch in text)
    gap = " " * max(0, width - w)
    return (gap + text) if right else (text + gap)


# The three admission gates, as `recipe_lifts.json` spells them. They are
# data written by the measuring script, in English, so they are translated by
# lookup rather than regenerated. An atom with no entry passes through
# verbatim: a reason we cannot translate is still a reason, and dropping it
# would turn "no gain admitted because the end state is worse than theta_0"
# into an unexplained zero.
_ZERO_ATOMS = {
    "CI includes 0": {
        "zh": "置信区间包含 0",
        "ko": "신뢰구간이 0을 포함한다",
    },
    "below the 0.003 effect floor": {
        "zh": "低于 0.003 的效应下限",
        "ko": "0.003 효과 하한에 미달",
    },
    "end state is worse than theta_0": {
        "zh": "终点状态比 theta_0 更差",
        "ko": "종점 상태가 theta_0보다 나쁘다",
    },
}


def zero_reason(text: str) -> str:
    """Translate a `zero_reason` string atom by atom."""
    if _BLOCK_LANG == "en" or not text:
        return text
    out = []
    for atom in str(text).split(";"):
        a = atom.strip()
        if not a:
            continue
        out.append(_ZERO_ATOMS.get(a, {}).get(_BLOCK_LANG, a))
    return "; ".join(out)

# ── action space, rendered per run ────────────────────────────────────────
# The three templates carry `<<ACTION_SPACE>>` where the field list used to
# be written out by hand. That hand-written list was the same defect this
# project removed from the vendored prompt in section 29 -- describing fields
# the validator will drop -- reintroduced in a per-dataset form: when
# `enable_query_rewrite` is dropped from the action space because the dataset
# has no rewrite cache, a static list still advertised it, the LLM could
# still propose it, and `diagnosis.can_change` then silently refused the
# proposal. The round changed nothing and scored a rejection.
#
# A marker rather than a `{placeholder}`: the `.format()` call lives in the
# vendored diagnosis.py with a fixed keyword list, so a new placeholder would
# raise KeyError, and `check_prompt_variants()` compares against the base
# prompt's placeholder set. The marker is substituted at install time, before
# the variant is registered, so neither is disturbed.
ACTION_SPACE_MARKER = "<<ACTION_SPACE>>"

FIELD_DOCS = {
    "en": {
        "fusion_mode": "- fusion_mode (str): one of \"keyword_only\" | \"semantic_only\" | \"structured_only\" | \"weighted_sum\" | \"rrf\". How the views are combined. \"keyword_only\" uses BM25 alone. \"weighted_sum\" adds normalised per-view scores using the weights below; \"rrf\" combines ranks instead of scores, so it is insensitive to score scale. No other value is accepted: \"first_found\" exists in the engine but is rejected here, because it concatenates the views in a fixed order without sorting, so the first view fills the context and the others contribute nothing (measured identical to semantic_only).",
        "semantic_top_k": "- semantic_top_k (int, >= 0): depth of the semantic (bge-m3) view. 0 disables it.",
        "keyword_top_k": "- keyword_top_k (int, >= 0): depth of the BM25 view. 0 disables it.",
        "structured_top_k": "- structured_top_k (int, >= 0): depth of the metadata view, which scores overlap of cited statute references between query and document. 0 disables it.",
        "weight_semantic / weight_keyword / weight_structured": "- weight_semantic / weight_keyword / weight_structured (float, >= 0): per-view weight under \"weighted_sum\". A view's depth and its weight are separate decisions: depth sets what it can return, weight sets how much that counts.",
        "enable_entity_swap": "- enable_entity_swap (bool): runs a second search on the query with party names stripped, then merges. Intended for queries whose names are incidental to the legal issue.",
        "entity_swap_keyword_k": "- entity_swap_keyword_k (int, >= 0): depth of that second search.",
        "enable_query_rewrite": "- enable_query_rewrite (bool): uses a pre-computed rewritten form of each query. No LLM call at query time; the rewrites are built offline.",
        "enable_chapter_filter": "- enable_chapter_filter (bool): restricts candidates to the query's penal-code chapter (or the dataset's equivalent taxonomy). Raises precision when the predicted chapter is right and makes the correct document unreachable when it is wrong.",
        "mmr_diversity_weight": "- mmr_diversity_weight (float, 0.0-0.9): post-fusion MMR rerank. 0 = off. Use when the top ranks are near-duplicate documents of the same case.",
        "per_category_overrides": "- per_category_overrides (dict): {{category: {{field: value}}}} applies a different config to one category only. Use it when ONE category lags and a global change would cost the others.",
    },
    "zh": {
        "fusion_mode": "- fusion_mode (str)：取值为 \"keyword_only\" | \"semantic_only\" | \"structured_only\" | \"weighted_sum\" | \"rrf\" 之一。决定各视图如何合并。\"keyword_only\" 只用 BM25。\"weighted_sum\" 用下面的权重对归一化后的各视图分数加权求和；\"rrf\" 合并的是排名而非分数，因此不受分数量纲影响。其他取值一律不接受：\"first_found\" 引擎里有，但本任务拒绝它——它按固定顺序把各视图首尾相接且不排序，排在前面的视图会占满上下文，其余视图毫无贡献（实测与 semantic_only 完全相同）。",
        "semantic_top_k": "- semantic_top_k (int, >= 0)：semantic（bge-m3）视图的深度。0 表示关闭该视图。",
        "keyword_top_k": "- keyword_top_k (int, >= 0)：BM25 视图的深度。0 表示关闭。",
        "structured_top_k": "- structured_top_k (int, >= 0)：元数据视图的深度，该视图计算查询与文书之间引用法条的重合度。0 表示关闭。",
        "weight_semantic / weight_keyword / weight_structured": "- weight_semantic / weight_keyword / weight_structured (float, >= 0)：\"weighted_sum\" 下各视图的权重。视图的深度与其权重是两个独立的决定：深度决定它能返回什么，权重决定这些结果算多少分。",
        "enable_entity_swap": "- enable_entity_swap (bool)：对去掉当事人姓名后的查询再检索一次，然后合并结果。适用于姓名与法律争点无关的查询。",
        "entity_swap_keyword_k": "- entity_swap_keyword_k (int, >= 0)：上述第二次检索的深度。",
        "enable_query_rewrite": "- enable_query_rewrite (bool)：使用每条查询预先改写好的形式。查询时不调用 LLM；改写是离线生成的。",
        "enable_chapter_filter": "- enable_chapter_filter (bool)：把候选限制在查询所属的刑法章节（或该数据集对应的分类体系）内。当预测的章节正确时提升精确率，预测错误时会让正确文书变得不可达。",
        "mmr_diversity_weight": "- mmr_diversity_weight (float, 0.0-0.9)：融合后的 MMR 重排。0 表示关闭。当排名前列是同一案件的近重复文书时使用。",
        "per_category_overrides": "- per_category_overrides (dict)：{{category: {{field: value}}}}，只对某一个类别应用不同的配置。当**某一个**类别明显落后、而全局改动会损害其他类别时使用。",
    },
    "ko": {
        "fusion_mode": "- fusion_mode (str): \"keyword_only\" | \"semantic_only\" | \"structured_only\" | \"weighted_sum\" | \"rrf\" 중 하나. 뷰들을 어떻게 합칠지 정한다. \"keyword_only\"는 BM25만 쓴다. \"weighted_sum\"은 아래 가중치로 정규화된 뷰별 점수를 더한다. \"rrf\"는 점수 대신 순위를 합치므로 점수 척도에 둔감하다. 그 밖의 값은 받지 않는다: \"first_found\"는 엔진에 있지만 이 과제에서는 거부한다 — 정해진 순서로 뷰를 이어붙일 뿐 정렬하지 않아서, 앞 뷰가 컨텍스트를 다 채우면 나머지 뷰의 기여가 0이 된다(실측값이 semantic_only와 동일).",
        "semantic_top_k": "- semantic_top_k (int, >= 0): semantic(bge-m3) 뷰의 깊이. 0이면 끈다.",
        "keyword_top_k": "- keyword_top_k (int, >= 0): BM25 뷰의 깊이. 0이면 끈다.",
        "structured_top_k": "- structured_top_k (int, >= 0): 메타데이터 뷰의 깊이. 이 뷰는 질의와 문서가 인용한 법조문의 겹침을 점수화한다. 0이면 끈다.",
        "weight_semantic / weight_keyword / weight_structured": "- weight_semantic / weight_keyword / weight_structured (float, >= 0): \"weighted_sum\"에서 뷰별 가중치. 뷰의 깊이와 가중치는 별개의 결정이다. 깊이는 그 뷰가 무엇을 돌려줄 수 있는지를, 가중치는 그것을 얼마로 셀지를 정한다.",
        "enable_entity_swap": "- enable_entity_swap (bool): 당사자 이름을 제거한 질의로 한 번 더 검색해 결과를 합친다. 이름이 법적 쟁점과 무관한 질의를 위한 것이다.",
        "entity_swap_keyword_k": "- entity_swap_keyword_k (int, >= 0): 그 두 번째 검색의 깊이.",
        "enable_query_rewrite": "- enable_query_rewrite (bool): 질의마다 미리 만들어 둔 재작성본을 쓴다. 질의 시점에 LLM을 호출하지 않는다. 재작성은 오프라인으로 생성되어 있다.",
        "enable_chapter_filter": "- enable_chapter_filter (bool): 후보를 질의가 속한 형법 장(또는 해당 데이터셋의 대응 분류체계)으로 제한한다. 예측한 장이 맞으면 정밀도가 올라가고, 틀리면 정답 문서가 닿을 수 없게 된다.",
        "mmr_diversity_weight": "- mmr_diversity_weight (float, 0.0-0.9): 융합 후 MMR 재순위. 0이면 끈다. 상위 순위가 같은 사건의 거의 중복인 문서들로 차 있을 때 쓴다.",
        "per_category_overrides": "- per_category_overrides (dict): {{category: {{field: value}}}} 형태로 한 카테고리에만 다른 설정을 적용한다. **한** 카테고리가 뒤처지고 전역 변경이 다른 카테고리를 희생시킬 때 쓴다.",
    },
}


def render_action_space(fields, lang: str) -> str:
    """The `fields` the LLM may propose, documented, in `lang`.

    A key naming several fields ("weight_semantic / weight_keyword /
    weight_structured") is kept when ANY of them is proposable: the three
    weights are one explanation and splitting it would say the same thing
    three times. `per_category_overrides` is always available -- it is a
    wrapper, not a retrieval dimension, and the engine accepts it regardless
    of which fields are in the space.
    """
    table = FIELD_DOCS.get(lang) or FIELD_DOCS["en"]
    allowed = set(fields)
    out = []
    for key, line in table.items():
        names = [n.strip() for n in key.split("/")]
        if key == "per_category_overrides" or any(n in allowed for n in names):
            out.append(line)
    return "\n".join(out)


def check_field_docs() -> None:
    """Raise unless the three languages document the same fields in the same
    order, and unless every EFFECTIVE_FIELD has a line. A field with no entry
    would be proposable but undocumented -- the mirror of the bug above."""
    import recipe_learning as _RL
    keys = [list(FIELD_DOCS[lg]) for lg in ("en", "zh", "ko")]
    if not (keys[0] == keys[1] == keys[2]):
        raise ValueError(f"FIELD_DOCS keys differ across languages: {keys}")
    documented = set()
    for k in keys[0]:
        documented.update(n.strip() for n in k.split("/"))
    missing = set(_RL.EFFECTIVE_FIELDS) - documented
    if missing:
        raise ValueError(
            f"FIELD_DOCS has no line for {sorted(missing)} -- those fields are "
            f"proposable but the prompt would not describe them.")

# ── priority order, rendered per run ─────────────────────────────────────
# The second static field list. `<<ACTION_SPACE>>` fixed the field
# descriptions but this section named the same fields again by tier, so
# `enable_query_rewrite` stayed in front of the LLM on the three datasets
# where it was dropped -- the verification harness caught exactly that.
#
# The tiers come from ONE place: the map legal_evolve installs with
# DIAG.set_impact_tiers(), which is also what the engine sorts by when the
# step cap truncates. Rendering from it means the prompt's stated priority
# and the engine's actual truncation order cannot drift apart.
PRIORITY_MARKER = "<<PRIORITY_ORDER>>"

PRIORITY_PROSE = {
    "en": [
        "1. Whether the semantic view contributes at all: {f}. These are\n   coupled. Turning a view on while its weight is wrong is worse than\n   leaving it off, so a round that switches a view on without setting its\n   weight has tested nothing.",
        "2. Whether the auxiliary views contribute: {f}.",
        "3. Lexical depth and the remaining weights: {f}.",
        "4. {f}."
    ],
    "zh": [
        "1. semantic（语义）视图是否起作用：{f}。这些字段相互耦合。在权重不对的情况下打开一个视图，比不打开更糟，所以一个只打开视图却不设定其权重的轮次什么也没有检验。",
        "2. 辅助视图是否起作用：{f}。",
        "3. 词法深度与其余权重：{f}。",
        "4. {f}。"
    ],
    "ko": [
        "1. semantic(의미) 뷰가 기여하는지 자체: {f}. 이들은 서로 묶여 있다. 가중치가 틀린 상태에서 뷰를 켜는 것은 끄고 두는 것보다 나쁘므로, 뷰를 켜면서 그 가중치를 정하지 않은 라운드는 아무것도 검증하지 않은 것이다.",
        "2. 보조 뷰가 기여하는지: {f}.",
        "3. 어휘 뷰의 깊이와 남은 가중치: {f}.",
        "4. {f}."
    ]
}


def render_priority(tiers: dict, fields, lang: str) -> str:
    """Tier lines for the fields in `fields`, ordered by `tiers`.

    A tier with no proposable field is omitted entirely rather than printed
    empty, and the numbering is left as authored so a missing tier is
    visible as a gap instead of silently renumbering the LLM's mental model
    between datasets.
    """
    prose = PRIORITY_PROSE.get(lang) or PRIORITY_PROSE["en"]
    allowed = [f for f in fields]
    out = []
    for t in (1, 2, 3, 4):
        names = [f for f in allowed if tiers.get(f) == t]
        if not names:
            continue
        joined = ", ".join(f"`{n}`" for n in names)
        out.append(prose[t - 1].format(f=joined))
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════
# Recipe card descriptions, per corpus language
# ════════════════════════════════════════════════════════════════════════
# Merged from recipe_text.py on 2026-11-11. It was a separate module and had
# to import this one for the renderer labels while this one imported
# recipe_learning, which imports this one back -- a three-way cycle that only
# worked because every edge was a function-local import. Both files hold the
# same thing: text the diagnosis LLM reads, in the corpus language. One file.
#
# Recipe descriptions in the corpus language.
#
# The recipe block is the largest piece of static text in the diagnosis prompt
# -- 5,003 characters against a 3,814-character Chinese template -- so leaving
# it English would have made a "Chinese prompt" 57% English by character count,
# and the English half is exactly the part that tells the LLM what each move
# does. These are the same descriptions as `evolution_cookbook.RECIPES_LEGAL`,
# translated.
#
# What is NOT translated, deliberately:
#   * `name` -- it is the key the LLM writes into `"use_recipe"`, and
#     `match_recipes` looks it up by that string.
#   * field names, enum values, and the numbers.
#   * `trigger` and `proposal` -- code.
#
# Keeping the measured figures identical in all three languages matters more
# than it looks: the figures are the only evidence the card carries, and a
# translation that rounded or dropped one would make the Chinese and English
# runs different experiments rather than the same experiment in two languages.
#
# `check()` asserts the three sets of keys agree with the live pool, so a card
# added or renamed in the cookbook fails here instead of silently reverting
# that card to English mid-run.


ZH = {
    # BUNDLE (recipes.BUNDLES). 一张不设任何值的卡：它只是放宽本轮的 step cap。
    "semantic_needs_score_fusion": (
        "【捆绑卡】本卡不设定任何字段的值。调用它只意味着一件事：本轮你可以同时修改 "
        "fusion_mode 和 semantic_top_k，而不必为每个字段各花一个改动名额。\n"
        "    为什么这两个字段不独立：在不做分数融合的模式下打开 semantic 视图实测 -0.115——"
        "视图按固定顺序首尾相接，排在前面的视图会占满 max_context。分开两轮来改，"
        "中间那一步比 theta_0 更差，elitist 会把它回退，于是永远到不了目的地。\n"
        "    值全部由你决定，本卡不给任何数字。同一轮里你还应当自行设定 weight_semantic："
        "theta_0 把它留在 1.0，那是没有人挑选过的默认值，而在那个权重下 semantic 分数会淹没 BM25 的排序。"
        "合适的权重取决于语料，四个数据集并不相同。"),
    "lexical_first_light_semantic": (
        "基础卡。把分数融合和 semantic 视图一起打开：在 keyword_only 融合下 semantic 视图毫无贡献，"
        "而没有第二个视图的分数融合也无可融合——每个字段单独改动实测恰好 +0.0000，所以它们是一步。"
        "entity-swap 之前必须先做这一步，因为没有分数融合它同样什么也不做。这里的深度 100 只是让该视图处于活动状态；它不是调优过的值，修改它由你决定。\n"
        "    你必须在同一轮里设定 weight_semantic。本卡故意不设。theta_0 把它留在 1.0——"
        "一个没有任何人选择过的默认值——而在那个权重下打开 semantic 视图比 theta_0 更差："
        "semantic 分数会淹没 BM25 的排序。合适的权重是语料的性质，不是本系统的性质："
        "它取决于判决书有多长、候选文书之间本来有多相似，而且在四个数据集上并不相同。"
        "它在你正在优化的这个数据集上还没有被测量过，所以把它当作你要去找的东西："
        "提出一个值，读分类别得分，再调整。本卡池中早先那个约 0.3 的数字来自单一数据集、"
        "且当时文书被截断到 512 个 token，它不是关于这个数据集的证据。"),
    "entity_swap_shallow": (
        "从查询中去掉当事人姓名后重新检索，再与原结果取并集。被告人的姓名与两个案件是否相似无关。"
        "需要分数融合；在 first_found 下实测恰好 +0.0000。本卡只负责打开这个视图——"
        "融合方式和 swap 深度现在是独立的卡，所以如果这个视图需要与默认不同的深度，你自己提出来。"),
    "query_rewrite_append_once": (
        "把事实描述的 LLM 改写追加到原查询之后。追加一次实测 +1.3 pp；"
        "追加两次掉到 +0.0，四次掉到 -3.4，而用改写\"替换\"原查询则损失 -17.0——"
        "法律术语本身不携带特异性，原始叙述才携带。与 entity-swap 重叠（逐查询相关系数 +0.52）："
        "两者同时打开实测比单独打开任何一个都差。"),
    "undo_structured_view": (
        "关掉 structured 视图。法条与罪名匹配被用五种方式测量过"
        "（分数加成、硬性章节过滤、查询扩展、抽取修复 36% -> 99.8%、属性学习），"
        "在加权求和下混合时代价最高达 -9.0 pp：逐视图的 max 归一化把数千份只有一项元数据匹配的文书"
        "映射到同一个值，压平了 BM25 排序。在 RRF 下符号反转——只用到排名，"
        "因此一个浅的 structured 视图（k<=10）实测 +0.002，并且是四个数据集上最佳配置的一部分。"
        "你会看到这张卡，是因为某个仍然成立的轮次把该视图打开了——本卡池里没有任何卡会提高 "
        "structured_top_k，而 theta_0 把它留在 0，所以那是你自己的提案被接受了（很可能是作为一对改动的一部分，收益来自另一个字段）。"),
    "undo_chapter_filter": (
        "关掉刑法章节过滤。把候选限制在查询所属章节实测 -8.3 pp："
        "只有 61% 的相关案件与查询同章，所以这个过滤永久丢弃了其中 39%，"
        "而它移除的那些候选本来也不在争夺前列排名。你会看到这张卡，是因为过滤当前处于 ON，而 theta_0 把它关闭且没有任何卡会打开它——所以是你先前的提案设置了它并被接受，也可能是为了同一轮里另一个字段的收益。"),
}

KO = {
    # BUNDLE (recipes.BUNDLES). 값을 하나도 설정하지 않는 카드 — step cap만 넓힌다.
    "semantic_needs_score_fusion": (
        "【묶음 카드】이 카드는 어떤 필드의 값도 설정하지 않는다. 호출하면 단 하나가 "
        "달라진다: 이번 라운드에 fusion_mode와 semantic_top_k를 **함께** 바꿔도 "
        "각각 한 칸씩 쓰지 않는다.\n"
        "    두 필드가 독립이 아닌 이유: 점수 융합이 아닌 모드에서 semantic 뷰를 켜면 "
        "실측 -0.115다. 뷰들이 정해진 순서로 이어붙여지고 앞 뷰가 max_context를 다 채우기 "
        "때문이다. 두 라운드에 나눠 바꾸면 중간 상태가 theta_0보다 나빠 elitist가 되돌리고, "
        "그래서 목적지에 영영 닿지 못한다.\n"
        "    값은 전부 당신이 정한다. 이 카드는 숫자를 하나도 주지 않는다. 같은 라운드에 "
        "weight_semantic도 스스로 설정해야 한다: theta_0은 그것을 1.0에 두는데 아무도 고른 적 "
        "없는 기본값이고, 그 가중치에서는 semantic 점수가 BM25의 순서를 덮어버린다. "
        "적절한 가중치는 코퍼스의 성질이고 네 데이터셋에서 서로 다르다."),
    "lexical_first_light_semantic": (
        "기반 카드. 점수 융합과 semantic 뷰를 함께 켠다: keyword_only 융합에서 semantic 뷰는 "
        "아무것도 기여하지 않고, 두 번째 뷰가 없는 점수 융합은 융합할 것이 없다 — "
        "각 필드를 따로 바꾸면 정확히 +0.0000이 측정되므로 둘은 한 걸음이다. "
        "entity-swap보다 먼저 와야 한다. 점수 융합 없이는 그것도 아무 일도 하지 않는다. 여기의 깊이 100은 뷰를 활성으로 만들기만 한다. 조정된 값이 아니며, 고치는 것은 당신의 몫이다.\n"
        "    당신은 같은 라운드에 weight_semantic을 반드시 설정해야 한다. 이 카드는 "
        "의도적으로 설정하지 않는다. theta_0은 그것을 1.0에 두는데, 아무도 고른 적 없는 "
        "기본값이며, 그 가중치로 semantic 뷰를 켜는 것은 theta_0보다 더 나쁘다 — semantic 점수가 "
        "BM25 순위를 덮어버린다. 맞는 가중치는 이 시스템의 성질이 아니라 코퍼스의 성질이다: "
        "판결문이 얼마나 긴지, 후보 문서들이 이미 서로 얼마나 닮았는지에 달려 있고, "
        "네 데이터셋에서 같지 않다. 당신이 최적화하고 있는 데이터셋에 대해서는 측정된 적이 "
        "없으므로, 당신이 찾아야 할 것으로 다루라: 값을 제안하고, 분류별 점수를 읽고, 옮기라. "
        "이 카드 묶음에 있던 약 0.3이라는 이전 수치는 한 데이터셋에서, 문서가 512 토큰으로 "
        "잘려 있던 코퍼스에서 나온 것이고, 이 데이터셋에 대한 증거가 아니다."),
    "entity_swap_shallow": (
        "질의에서 당사자 이름을 떼고 다시 검색해 결과를 합집합으로 합친다. 피고인의 이름은 "
        "두 사건이 닮았는지와 무관하다. 점수 융합이 필요하다. first_found 아래에서는 정확히 "
        "+0.0000이 측정되었다. 이 카드는 뷰를 켜는 일만 한다 — 융합 방식과 swap 깊이는 "
        "이제 별도의 카드이므로, 이 뷰가 기본값과 다른 깊이를 필요로 한다면 당신이 직접 말하라."),
    "query_rewrite_append_once": (
        "사실관계의 LLM 재작성본을 원래 질의 뒤에 덧붙인다. 한 번 덧붙이면 +1.3 pp가 "
        "측정되었다. 두 번 덧붙이면 +0.0으로, 네 번이면 -3.4로 떨어지고, 원문을 재작성본으로 "
        "'대체'하면 -17.0을 잃었다 — 법률 용어는 그 자체로 특정성을 갖지 않고, 원래의 서술이 "
        "갖는다. entity-swap과 겹친다(질의별 상관 +0.52): 둘을 같이 켜면 어느 하나만 켠 것보다 "
        "나쁘게 측정되었다."),
    "undo_structured_view": (
        "structured 뷰를 끈다. 법조문과 죄명 매칭은 다섯 가지 방식으로 "
        "측정되었고(점수 가산, 강성 장 필터, 질의 확장, 추출 보정 36% -> 99.8%, 속성 학습), "
        "가중합으로 섞을 때 최대 -9.0 pp를 치렀다: 뷰별 max 정규화가 메타데이터 일치 하나를 가진 "
        "수천 건의 문서를 같은 값으로 보내 BM25 순위를 평평하게 만든다. RRF에서는 부호가 뒤집힌다 — "
        "순위만 쓰이므로 얕은 structured 뷰(k<=10)는 +0.002가 측정되었고 네 데이터셋 모두에서 "
        "이 카드가 보이는 것은, 아직 유지되고 있는 어떤 라운드가 그 뷰를 켰기 때문이다. "
        "이 묶음의 어떤 카드도 structured_top_k를 올리지 않고 theta_0은 0으로 두므로, "
        "당신 자신의 제안이 수용된 것이다(이득은 같은 라운드의 다른 필드가 가져왔을 가능성이 높다)."),
    "undo_chapter_filter": (
        "형법 장 필터를 끈다. 후보를 질의의 장으로 제한하면 -8.3 pp가 측정되었다: 관련 사건 중 "
        "61%만이 질의와 같은 장을 공유하므로 이 필터는 그중 39%를 영구히 버리는데, "
        "그것이 제거하는 후보들은 애초에 상위 순위를 다투고 있지도 않았다. 이 카드가 보이는 "
        "것은 필터가 ON이기 때문이고, theta_0은 끈 상태로 두며 켜는 카드도 없으므로 "
        "당신의 이전 제안이 그것을 설정하고 수용된 것이다. 같은 라운드의 다른 필드 "
        "때문이었을 수도 있다."),
}

BY_LANG = {"zh": ZH, "ko": KO}


def check_recipe_text(recipes) -> None:
    """Raise unless every card in the live pool has a translation.

    A table entry with no card is NOT an error: `drop_dead_cards` removes
    cards per dataset (query_rewrite_append_once has nothing to do where no
    rewrite cache exists), so the pool is a subset of the tables on three of
    four datasets. The failure that matters is the other direction -- a card
    the cookbook gained or renamed, which would otherwise silently revert to
    English mid-run.
    """
    live = {r.name for r in recipes}
    for lang, table in BY_LANG.items():
        missing = live - set(table)
        if missing:
            raise ValueError(
                f"recipe_text[{lang!r}] has no entry for {sorted(missing)} -- "
                f"those cards would render in English. Add them, or remove "
                f"them from RECIPES_LEGAL.")


def localize(recipes, lang: str):
    """Return `recipes` with descriptions in `lang`, and the count translated.

    Cards with no entry keep their English description rather than raising:
    the `ladder` and `learned` pools are generated, and a generated card has
    no hand-written translation. Callers print the count so a run says how
    much of its recipe block was English.

    Returns new Recipe objects; the pool in evolution_cookbook is left alone
    so a second install for another language starts from English again.
    """
    table = BY_LANG.get(lang)
    if not table:
        return list(recipes), 0
    from dataclasses import replace
    out, n = [], 0
    for r in recipes:
        t = table.get(r.name)
        if t is None:
            out.append(r)
        else:
            out.append(replace(r, description=t))
            n += 1
    return out, n


def drop_dead_cards(cookbook, dataset: str, key: str = "legal_retrieval") -> list:
    """Remove cards whose proposal cannot act on `dataset`. Returns names.

    `query_rewrite_append_once` sets `enable_query_rewrite`, which reads
    cache/rewrite/<dataset>.json. Only LeCaRDv2 has one, so on the other
    three the card is a slot the LLM can spend for a guaranteed +0.0000 --
    and a card is worth a slot only when it can move something. Section 28
    removed `undo_query_decomposition` for the same reason, by hand; this
    does it from the condition instead of from a list.
    """
    # Same predicate the retriever uses, not a file-existence proxy: an
    # existing but empty cache/rewrite/<ds>.json leaves expand_query a no-op.
    import legal_retriever as _L
    need = {"enable_query_rewrite": bool(_L.load_rewrites(dataset))}
    pool, keep, gone = cookbook.RECIPE_POOLS.get(key) or [], [], []
    for r in pool:
        dead = [f for f in r.proposal if f in need and not need[f]]
        (gone if dead else keep).append(r.name if dead else r)
    cookbook.RECIPE_POOLS[key] = keep
    return gone


def install_recipe_text(diagnosis_lang: str, cookbook,
                        key: str = "legal_retrieval") -> str:
    """Localise the pool AND the renderer. Returns a one-line report.

    The renderer labels live in the cookbook, not in the card, so a pool
    translated without them produced a Chinese description under an English
    "(expected +0.1 pp)" header. It also stops printing a lift that nothing
    measured on this dataset -- see format_recipes_for_prompt.
    """
    check_recipe_text(cookbook.RECIPE_POOLS.get(key) or [])
    cookbook.set_show_unmeasured_lift(False)
    cookbook.set_recipe_labels({
        "plain": t("recipe.plain"),
        "measured": t("recipe.measured"),
        "fields": t("recipe.fields"),
        "none": t("recipe.none"),
    })
    pool = cookbook.RECIPE_POOLS.get(key) or []
    if not pool:
        return "recipe text: pool empty, nothing to localise"
    localised, n = localize(pool, diagnosis_lang)
    cookbook.RECIPE_POOLS[key] = localised
    if diagnosis_lang not in BY_LANG:
        return f"recipe text: {diagnosis_lang} (authored language, {len(pool)} cards)"
    return (f"recipe text: {diagnosis_lang} for {n}/{len(pool)} cards"
            + (f", {len(pool) - n} left in English" if n < len(pool) else ""))
