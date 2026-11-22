# 방법론 — EvolveMem을 법률 판례 검색으로

작성 2026-10-22, 갱신 2026-11-16. 실험 기록은 [PROGRESS.md](PROGRESS.md),
날짜별 발견과 실수는 [JOURNAL.md](JOURNAL.md). 여기서는 **무엇을 어떻게 재는가**와
**거기서 무엇이 밝혀졌는가**만 적는다.

---

## 이 문서는 무엇인가

**시스템이 지금 어떻게 동작하는지**만 적는다. 날짜가 붙은 발견과 실수는
[JOURNAL.md](JOURNAL.md)로 옮겼고, 지금 무엇이 돌고 있는지는
[PROGRESS.md](PROGRESS.md)에 있다.

| 찾는 것 | 파일 |
|---|---|
| 용어, 검색 구조, action space, 평가 프로토콜 | **이 문서** |
| 왜 그렇게 정했나, 무엇이 틀렸었나 (날짜순) | [JOURNAL.md](JOURNAL.md) |
| 지금 돌고 있는 것, 다음 할 일, 실행 방법 | [PROGRESS.md](PROGRESS.md) |


---

## 처음 읽는 사람을 위해

### 무엇을 하는 시스템인가

**법률 질의를 받아 판례나 조문의 순위를 매긴다.** 질의는 사실관계 서술이거나
법조문 질문이고, 정답은 그 질의와 관련 있다고 판정된 문서들이다. 지표는
**NDCG@30** 하나다.

출발점은 EvolveMem 논문의 **self-evolution loop**다. 원래 용도는 대화 메모리
QA인데, 우리는 그 loop만 떼어 와서 검색 설정을 스스로 고치게 한다. 핵심 아이디어는
한 문장이다 — **실패를 LLM에게 보여주고, LLM이 설정을 고치고, 고친 설정을 다시
채점한다. 좋아지면 유지하고 아니면 되돌린다.**

중요한 구조 하나: loop가 **둘**이고 LLM이 사는 곳은 하나뿐이다.

| | 무엇을 하나 | LLM 호출 |
|---|---|---|
| **outer loop** (evolution) | 라운드마다 설정 θ를 고친다 | 라운드당 **1회** (diagnosis) |
| **inner loop** (retrieval) | 질의마다 문서를 뽑아 순위를 매긴다 | 질의당 **0회** |

질의당 0회라는 것이 이 프로젝트를 논문과 가르는 지점이다. 논문은 질의마다 답을
생성하므로 라운드가 15~20분 걸리지만, 우리는 **순위 자체가 출력**이라 답변 생성
단계가 아예 없다. 그래서 라운드가 수십 초~수 분이고, 그 대신 LLM이 할 일은
"답을 잘 쓰는 것"이 아니라 **"설정을 잘 고치는 것"** 하나로 좁혀진다.

### 한 라운드는 이렇게 돈다

1. 현재 설정 θ로 train fold의 모든 질의를 검색하고 NDCG@30을 낸다.
2. 실패를 정리한다 — 0점 질의가 어느 카테고리에 몰렸는지, 정답 문서가 **아예
   안 잡혔는지**(recall 문제) **잡혔는데 밀렸는지**(순서 문제), 어느 view가
   무엇을 찾았는지.
3. 그 정리를 **diagnosis prompt**에 담아 LLM에게 한 번 보낸다. 프롬프트는
   코퍼스 언어로 간다(중국어 판결문이면 중국어 프롬프트).
4. LLM이 JSON으로 답한다 — 바꿀 필드와 값, 또는 쓰려는 **recipe** 이름.
5. 제안을 검증해 적용한다. 한 라운드에 **최대 2개 필드**만 움직인다(step cap).
   recipe는 필드를 몇 개 묶어도 1칸으로 센다.
6. 다시 채점한다. 지금까지의 최고 상태를 **acceptance threshold** 이상으로
   이기면 유지하고, 아니면 **되돌린다**(elitist). 거부된 수는 기록에 남아
   다음 라운드 프롬프트에 들어간다 — 같은 수를 다시 제안하지 않도록.

### 네 개의 view

"view"는 후보를 뽑는 방식 하나다. 넷을 쓰고, 각각 켜고 끌 수 있다.

| view | 입력 | 무엇으로 찾나 | 설정 필드 |
|---|---|---|---|
| **BM25** | 질의 원문 | 어휘 일치 (jieba / mecab / spaCy) | `keyword_top_k`, `weight_keyword` |
| **party-strip** | **인명+사명+지명 삭제** | 어휘 일치 | `enable_entity_swap`, `entity_swap_keyword_k` |
| **semantic** | 질의 원문, **불변** | bge-m3 임베딩 유사도 | `semantic_top_k`, `weight_semantic` |
| **metadata** | `legal_refs` | 인용한 **법조문의 겹침** | `structured_top_k`, `weight_structured` |

**입력이 왜 저렇게 갈리는가 — 삭제는 어휘적 행위다.** BM25는 희귀 토큰 하나에 자체
가중치를 주므로, 공유된 지명 하나가 무관한 판례를 끌어올린다. 임베딩은 문장 전체를
표현하므로 이름이 벡터를 지배하지 않고 조금 움직인다. 그래서 **party-strip은 BM25의
어휘적 쌍둥이**이고, semantic은 무엇을 켜든 질의 원문을 읽는다
(`SEMANTIC_USES_ROW_TEXT`). 따라오는 결론: party-strip에는 semantic 절반이 애초에
없다(`entity_swap_semantic_k = 0`) — 설계가 말하는 값이고, 그래서 삭제-질의 벡터
선계산은 **필요하지 않다**(미뤄둔 과제에서 제거).

**"view"와 "질의 변환"은 다른 축이다.** view는 *채점 함수*다 — BM25는 어휘 일치로,
semantic은 cosine similarity로, metadata는 법조문 겹침 개수로 점수를 낸다. 반면
party-strip은 **질의 텍스트를 바꾸고(삭제) 다시 검색한 결과**이므로, 엄밀히는
*질의 변환*이다. 엔진이 이것을 `entity_swap`이라 부른 이유도 그것이다. 그래서
"네 개의 view"는 정확히는 **세 개의 채점 함수 + 질의 변환 하나**다 — 섞여 보이는 것은
느낌이 아니라 구조다.

> **2026-11-16: 이 서술을 좌표계가 대체한다.** "채점 함수 셋 + 질의 변환 하나"는 지금
> 등록된 네 레버를 묘사한 것일 뿐이고, 왜 그 넷인지는 말하지 않는다. 레버는
> **(입력 × 채점 × 결합)** 세 좌표로 등록하고, view는 그 격자의 한 칸이다 —
> §3 "레버 등록부와 등록 규칙". 그 표로 다시 적으면 **비어 있는 칸**(질의에 법률 용어를
> *더하는* 레버가 하나도 없다)과 **겹치는 칸**(rewrite가 party-strip과 같은 삭제
> 연산이다)이 같이 드러난다.

축을 셋으로 갈라 두면 누가 무엇을 정하는지가 바로 보인다.

| 축 | 우리 것 | 누가 정하나 | 측정된 효과 |
|---|---|---|---|
| **문서 변환** | `DOC_FIELD_OF`(판결문의 어느 절을 색인하나), `embed_text`, 임베딩 길이 | **사람 (tier ①)** | COLIEE θ₀ **+31%**, LEGAR **+42%** |
| **질의 변환** | `enable_query_rewrite`, party-strip(`STRIP_SCOPE`) | LLM (tier ②) | rewrite +0.0009(CI∋0), strip 전부 음수 |
| **view 조합** | 채점 함수 3개 × on/off × fusion × 가중치 | LLM (tier ②) | loop 최대 **+1.9%** |

문서 변환이 tier ①에 있는 이유는 **바꾸면 재임베딩(≈57분/데이터셋)이 필요해** 라운드당
한 번 움직이는 축으로 쓸 수 없기 때문이다. 그런데 **효과가 가장 큰 축이 거기 있다.**
이것이 이 연구의 가장 약한 지점이고, 숨기지 않고 그대로 보고한다 — 주장의 범위는
"loop가 ① 고정 아래에서 ②를 진화시킨다"이고, 분모로 +31%/+42%를 함께 적는다.

네 view의 결과를 하나의 순위로 합치는 방식이 **fusion**이다 —
`weighted_sum`(정규화 점수 가중합) / `rrf`(점수 대신 순위를 합산, k=60).

**2026-11-16에 이 축을 분해했다.** 엔진의 `fusion_mode` enum은 네 값
(`*_only` / `first_found` / `weighted_sum` / `rrf`)을 갖지만, 그 넷은 같은 축의 네
수준이 **아니다** — `*_only`는 view 마스크이고 `first_found`는 캐스케이드다. 둘을
둘을 빼면 fusion은 **2값**이 되는데, **지금 코드는 거기까지 가지 않았다** — `first_found`
하나만 막았으므로 **받는 값은 6개에서 5개**가 됐고, `*_only` 셋을 마스크 축으로 옮기는
일은 미착수다(§2의 B). "2값"은 **설계 목표**이고 현재 상태가 아니다.
근거와 남은 검증은 §2 "fusion_mode는 차원이 아니다", 액션 공간 표기는 §3 B.

출발 설정 **θ₀는 BM25 단독**이다. 보조 view를 전부 끈 상태에서 시작해, loop가
무엇을 켤 가치가 있는지 스스로 찾게 한다.

BM25는 항상 켜져 있으므로 **조합은 나머지 셋의 on/off = 8가지**다(1+3+3+1). 사람이
재는 격자(`sweep.py --grid fusion`)는 그 8가지를 전부 덮지만, **가중치를 각각 한 값에
고정한 얇은 단면**이다 — `weight_semantic`의 최적값만 데이터셋마다 0.1~2.0으로 20배
벌어진다. 격자는 채점용 정답지이고, 8가지 × fusion × 가중치의 전체 공간에 **어느 축이
중요한지 듣지 않고** 도달하는 것이 loop의 과제다.

### 무엇을 결과로 치는가

fold는 셋이고 역할이 다르다.

| fold | 쓰는 곳 |
|---|---|
| **train** | loop가 점수를 보는 곳. θ를 고치는 근거 |
| **val** | 라운드들 중 어느 설정을 쓸지 **고르는** 곳 |
| **test** | **아직 한 번도 채점하지 않았다.** 마지막에 한 번 |

벤치마크는 넷이고 언어가 셋이다 — **LeCaRD**·**LeCaRDv2**(중국어),
**COLIEE**(영어), **LEGAR_stricter**(한국어). 같은 파이프라인으로 넷을 똑같이
돌리는 것이 비교의 전제다.

**크기가 많이 다르다는 점을 먼저 알고 읽어야 한다.** 질의 수가 LeCaRD 107개부터
LEGAR_stricter 5,960개까지다. LeCaRD는 train 79 / val 8 / test 20이라, 우리가 아는
가장 좋은 한 수(+2.09 pp)조차 그 fold의 noise floor(2.40 pp) 안에 들어간다.
**LeCaRD에서 나온 숫자는 신뢰구간과 함께만 의미가 있고**, 원데이터가 그것뿐이므로
fold를 바꾸지 않고 보고 방식을 바꿨다 — JOURNAL §35·§36.

그리고 이 프로젝트가 보이려는 것은 "LLM이 튜닝보다 낫다"가 **아니다.**
LLM self-evolution engine은 **검증 대상이 아니라 전제**다. 보여야 할 것은
**데이터셋마다 다른 값에 스스로 도달하는가**, 그리고 **손으로 grid를 훑는 것보다
싸게 도달하는가**다.

---

### 지금 서 있는 결론 (근거는 JOURNAL)

표 하나에 51행이 쌓여 있었고, 그중 절반은 결론이 아니라 **날짜 붙은 정정**이었다 —
이 문서가 스스로 "JOURNAL에 있어야 한다"고 정한 것들이다. 셋으로 나누고, 정정은
JOURNAL §50으로 옮겼다(2026-11-22).

#### 지형 — 무엇이 실제로 작동하는가

검색 쪽 측정의 현재 상태. 전부 train 기준이고, Δ는 그 데이터셋의 noise floor와 함께 읽어야 한다.

| | 결론 | 근거 |
|---|---|---|
| θ₀ | BM25 단독. 이 도메인의 이득은 거의 전부 **"덜 개입하기"**에서 나왔다 | §8 |
| fusion | **"무관하다"는 단면의 산물이다 (2026-11-16 정정).** rrf 0.4025 / weighted_sum 0.4002는 **활성 view 조합을 한 값에 고정하고** 잰 것이고, 같은 문서가 §5에서 경고한 바로 그 함정이다 — view 1개면 rrf −0.010, 4개면 +0.0154로 **부호가 뒤집힌다.** fusion은 주효과로 보고할 수 없고 **(view 조합 × fusion) 셀**로만 읽는다. 그리고 축을 분해하기로 했다 — 목표는 **2값**(`weighted_sum`/`rrf`). **현재 코드는 6값 → 5값**이고(`first_found`만 차단), `*_only` 셋은 마스크 축으로 옮기기 전까지 남아 있다 | §2·§3, JOURNAL §27 |
| semantic view | **유일하게 일관된 양수 레버**(네 데이터셋 전부). 최적 가중치는 **0.1~2.0으로 20배** 벌어지고, θ₀ 기본값 1.0에서 켜면 LeCaRD −0.1067. 옛 "0.3에서만"은 **임베딩 512 잘림의 산물**이었다 | §8, JOURNAL §27·§28 |
| 메타데이터 view | **단독으로는 해롭고 보완재로는 기여한다.** 단독(③)은 네 곳 중 셋에서 음수지만 semantic과 함께면(④) COLIEE +0.0061·LEGAR +0.0064로 ②를 앞서고, 네 view 전부인 ⑧이 세 곳에서 1위다. 옛 결론 "네 데이터셋 train 전부 음수"는 **깊이 10 + 가중치 0.5**에서 잰 것이고 둘 다 엔진 기본값이었다 — 깊이를 100으로 통일하니 바뀌었다. `weight_structured`는 아직 sweep 중 | §8, JOURNAL §47 |
| party-strip view | **중립이다 — "죽었다"는 철회됐다.** 그 판정(최대 +0.0004, LeCaRD −0.0349)은 **전부 깊이 30에서 잰 것**이었고, 30은 옛 코퍼스의 잔존값이다(R3). off/10/30/100에서 LeCaRD +0.0134/+0.0139/+0.0172/**+0.0191**, LeCaRDv2 +0.0140/+0.0125/+0.0118/**+0.0161** — k=100에서 두 데이터셋 모두 끄는 것보다 높다. 다만 Δ가 floor 안이므로 정직한 결론은 **"깊이 30 판정은 무효, 100은 중립"**이고 "이득이다"는 아니다. 삭제 대상은 인명+사명+지명 (`STRIP_SCOPE`, ① 결정). **독립 가중치가 없다** — `weight_semantic`을 공유한다(§3) | §8, JOURNAL §16·§47 |
| 쿼리의 이름·지명 | **지우면 손해이거나 무효다. 양수인 곳이 없다.** 당사자명 삭제: LeCaRD −0.0457 / LeCaRDv2 −0.0032 / COLIEE −0.0003 / LEGAR +0.0002. **지명이 단독 최대 원인** (LeCaRDv2 −0.0136) — 정규식 결함을 고쳐도 그대로이고, 정답 문서가 쿼리의 지명을 **+51 pp** 더 공유한다. 개념적으로 지워야 할 신호를 벤치마크가 보상한다. 그리고 party-strip view는 이 오염을 치우지 않는다 — 주 BM25는 원문을 계속 읽는다 | §4, JOURNAL §41 |
| 깊이의 천장 | `keyword_top_k`는 **100 위에서 정확히 무효** — RRF의 1/(60+rank) 할인 때문. `semantic_top_k`는 200에서도 작동(+0.0018). 프롬프트의 "깊이를 올려 재현율을 얻으라"가 전자에 대해 **거짓** | JOURNAL §42 |
| **view 조합 8가지** | BM25는 항상 켜지므로 나머지 셋의 on/off = 1+3+3+1 = **8**. §8은 7개만 쟀고 네 view 전부 켜는 하나가 빠져 있었다. **깊이 전부 100으로 완주하니 그 ⑧이 세 데이터셋에서 1위**(LeCaRDv2 +0.0159 / COLIEE +0.0061 / LEGAR +0.0073), LeCaRD만 ②(+0.0134). **깊이 10/30에서는 ②가 네 곳 전부 1위로 보였다** — 기본값을 선택으로 읽은 결과다. **COLIEE·LeCaRD는 어느 설정도 floor를 못 넘어 "어떤 view를 켤지"가 측정 불가**다 | §8, JOURNAL §46·§47 |
| **view별 깊이** | **전부 100.** 100/100/10/30으로 흩어져 있었고 metadata의 10은 엔진 기본값, party-strip의 30은 옛 코퍼스 산물이었다. k∈{30,100,200} 측정(train, 네 데이터셋): **30은 semantic 이득 대부분을 잃는다**(LeCaRDv2 +0.0140 → +0.0042, LeCaRD는 θ₀로 복귀), 100이 knee, 200은 LeCaRDv2 +0.0024 뿐이고 floor 아래. **반환 수 30(=NDCG@30, 지표가 고정)과 무관한 별개의 숫자**다 | §8, JOURNAL §47 |
| 열려 있는 크기 | **셋이다** — `weight_semantic`(최적 0.1~2.0, 20배), 그리고 2026-11-13에 전제가 무너진 둘: `weight_structured`(metadata가 보완재로 기여하는 것이 깊이 100에서 드러났다 — sweep 중)와 `entity_swap_keyword_k`(party-strip "죽었다" 판정이 깊이 30의 산물이었다). 옛 근거 "각자의 view가 네 데이터셋 전부 음수/무효라 찾을 값이 없다"는 **둘 다 기본값(10/30, 0.5)에서 잰 것**이었다 | §8, JOURNAL §47 |
| BM25 레버 | `keyword_top_k`·`weight_keyword`는 **작동하는데 한 번도 sweep하지 않았고 어떤 레시피에도 없다** — 근거 없이 100/1.0 고정 | JOURNAL §25 |
| **semantic view의 입력** | **행의 원문 질의로 고정**(`SEMANTIC_USES_ROW_TEXT`). 변환은 BM25에만 — 두 레버의 존재 이유가 어휘적이고, semantic에는 **끄는 효과만** 있었다. 선계산 0으로 semantic이 온전해진다. swap의 semantic 절반은 중복 기여를 막으려 0 고정 = **은퇴**. **단 §44의 "rewrite가 +0.0161"은 기각됐다** — 그 측정은 카드 트리거 기반이 아닌 임의 120건이었고, 트리거 기반 재측정은 **+0.0009 CI[-0.0021,+0.0037]**(noise floor 0.0050 미달). D가 semantic을 원문에 고정하므로 rewrite는 BM25에만 닿고, 거기서는 0과 구분되지 않는다 — **D와 rewrite는 같은 통로를 다툰다** | JOURNAL §44·§46 |
| **semantic view의 조용한 소실** | 질의 벡터가 **질의 텍스트를 키로** 미리 계산돼 있어, 텍스트가 바뀌면 `semantic_search`가 빈 결과를 돌려준다. `enable_query_rewrite`는 주 경로를, party-strip은 자기 semantic 절반을 **전부** 끈다(40/40, 182/182). 그래서 rewrite 카드의 +0.0088은 묶음이고, `entity_swap_semantic_k`의 "무효"는 구조적 원인이다. 지금은 명명·계수하고 프롬프트에 찍는다 — **파생 질의 벡터 선계산이 남았다** | JOURNAL §43 |

#### 측정 — 무엇을 어떻게 재는가

숫자를 믿기 전에 알아야 하는 것들. 이 절의 항목 중 하나라도 어긋나면 위 표는 읽을 수 없다.

| | 결론 | 근거 |
|---|---|---|
| **지표의 의미** | **NDCG@30이 넷에서 다른 과제를 잰다.** 쿼리당 정답 수 중앙값 LeCaRD 29 / LeCaRDv2 **30** / COLIEE 3 / LEGAR **1** — 30배 차이. LeCaRDv2는 60%의 쿼리에서 만점 = precision@30 1.0(ordering 과제), LEGAR는 사실상 MRR. 모든 표가 네 Δ를 나란히 놓고 읽어 왔다 — **결정 필요** | JOURNAL §42 |
| 수용 임계값 | `0.003`은 엔진 기본값이고 **네 중 셋에서 noise floor보다 낮았다** (LeCaRD 7.7배). 즉 잡음을 수락하고 있었다. 데이터셋별 **측정된 noise floor**로 바꿨다. **값은 이 문서에 적지 않는다** — `noise_floor.json`이 유일한 출처이고, 실행이 그 값과 출처를 함께 찍는다. 문서에 박아 둔 숫자는 두 번 낡았다(§35의 넷 → 카드 기반 폴백 → 설정 쌍) | §5, JOURNAL §35 |
| **floor의 출처를 바꿨다 (2026-11-22)** | 그 floor는 `recipe_lifts.json`, 즉 **`measure_recipes.py`가 마침 측정해 둔 카드들**의 CI 중앙값이었다. 두 가지가 틀렸다 — (1) **pool의 성질이지 데이터셋의 성질이 아니다**: 살아 있던 숫자를 만든 카드 8장 중 **3장은 이미 존재하지 않았고**(`escape_plateau_shallower_swap`·`undo_auxiliary_overweight`·`undo_query_decomposition`) 2026-11-22에 3장이 더 지워졌다. LeCaRD의 0.0228은 카드 둘의 중앙값이고 그중 하나가 §48이 오독이라 판정한 `undo_chapter_filter`다. (2) **카드는 편향 표본이다** — 카드는 "누군가 도움이 되리라 본 수"라 몇 필드에 몰린다. 이제 **고정된 설정 쌍 13개**로 잰다(`noise_floor.py` → `noise_floor.json`): pool에 독립이고, 쌍을 늘릴수록 **좋아진다**(옛 방식은 pool이 줄수록 나빠졌다). 실측 LeCaRD **0.0228 → 0.0103**(옛 값은 이상치 둘이 중앙값을 끌어올린 것). `verify.py`가 **폴백이 쓰이면 실행을 막는다** | §5 |
| LeCaRD 해상도 | 107질의 = train 79 / **val 8** / **test 20**. train noise floor 2.40 pp > 최선의 알려진 수 2.09 pp, val Δ 구간 **±14 pp**, test 약 ±10 pp. **원데이터이므로 fold는 그대로 두고 보고를 바꿨다** — 모든 숫자가 CI를 달고, val 선택은 '순위'로 명시 | JOURNAL §35~36 |
| **test fold** | **깨끗하지 않다 — 이미 채점됐다.** §8의 네 데이터셋 표(7설정 × 4)가 **test fold**에서 나왔고(LeCaRDv2 160 / LeCaRD 20 / COLIEE 398 / LEGAR 1161), 거기서 얻은 "⑤가 1위"가 레시피와 진단 프롬프트로 들어갔다. 원인은 `evaluate.py`의 `--split` 기본값이 **`test`** 였다는 것 — 문장 하나로 막으려 했으나 `--split test`는 그냥 또 하나의 문자열이었다. 지금은 **벽**이다: `in_split`이 test·`all`·`None`을 `ALLOW_TEST=1` 없이 거부하고, 기본값은 train이다. §8은 train으로 다시 재서 교체하고, test는 봉인한다 | §5, §8, §10.1 |
| **test 가드** | `in_split()`이 `test`·`all`·`None`을 `ALLOW_TEST=1` 없이 거부한다. 문서의 문장은 막지 못했다 — `evaluate.py --split` **기본값이 `test`** 였고 그래서 §8이 test에서 돌았다. 기본값도 `train`으로 | §5, JOURNAL §46 |
| 비용 | 모든 비교는 사실 **비용 비교**인데 계측이 없었다. 지금은 `(쿼리,설정)` 채점 횟수를 센다. 축 하나 전수 grid ≈55,500 vs 진화 7라운드×4데이터셋 ≈84,400 — **축이 하나면 grid가 이긴다** | JOURNAL §28 |
| 실험의 전제 | LLM self-evolution 엔진은 **검증 대상이 아니라 전제**다. 보여야 할 것은 **데이터셋마다 다른 값에 스스로 도달하는가**이고, 정답지(`weight_semantic` 전수 sweep)는 채점용으로만 쓴다 — 프롬프트·카드에 넣으면 순환이다 | JOURNAL §28 |
| **이득의 귀속** | 사람이 내린 ① 결정 하나(색인 절 + 임베딩 길이)가 COLIEE θ₀를 **+31%**, LEGAR를 **+42%** 올렸다. 루프가 찾은 최대는 LeCaRDv2 val **+1.9%**. **한 자리 차이다.** 올바른 주장은 "루프가 ① 고정 아래에서 ②를 진화시킨다"이고, 분모를 함께 적어야 한다 | §10.0, JOURNAL §26 |
| `max_context` | **자유도가 아니다.** `search()`가 `max_context >= top_n`을 검사하고 아니면 `ValueError` — 채점 창이 조용히 짧아지는 것을 막는다 | JOURNAL §42 |

#### 구조 — 무엇이 어디에 사는가

누가 무엇을 바꿀 수 있는가, 그리고 그 경계가 코드의 어디에 강제되는가.

| | 결론 | 근거 |
|---|---|---|
| **결정의 세 층** | ① **view의 정의**(사람+측정, `RetrievalConfig` 밖) / ② **설정** 12필드(LLM, 라운드당 2칸) / ③ **recipe**(구조는 사람, 크기·시점은 LLM). ①을 열면 데이터셋마다 **다른 시스템**이 되어 비교 전제가 깨진다. `per_category_overrides` 우회로도 같은 필터로 막혀 있다(실측) | §0.3 |
| action space | 45개 중 **12개만** 실제로 순위를 바꾼다. 나머지 33개는 **제안 대상에서 제거했다**(2026-11-10) | JOURNAL §20·§26 |
| **레버 등록 규칙** | **새로 세웠다 (2026-11-16).** 이번에 찾은 다섯(축이 아닌 `fusion_mode` · 벤더 기본값 `first_found` · 레시피와 action space의 타입 동일 · rewrite ≡ party-strip · 단면에서 나온 fusion Δ)은 서로 다른 버그가 아니라 **레버 등록 규칙의 부재**라는 한 결함이다. 레버는 **(입력 × 채점 × 결합)** 좌표로 등록하고 일곱 규칙을 건다 — R1 축의 동질성 · R2 좌표 유일성 · R3 **기본값 위에서 판정 금지** · R4 Δ에 조건 명시 · R5 층이 다르면 타입이 다름 · R6 중복 표현 금지 · R7 접은 값은 네 곳에서 도달 불가. R3 하나가 과거 오판 넷(metadata 깊이 10 · party-strip 깊이 30 · `weight_structured` 0.5 · `first_found`)을 전부 설명한다 | §3 |
| **등록부의 빈칸** | **질의에 법률 용어를 *더하는* 레버가 없다.** 질의를 바꾸는 레버 둘(party-strip · rewrite)이 **모두 삭제**다. BM25는 문서에 있는 용어만 매칭하므로 삭제만으로는 재현율이 구조적으로 오르지 않는다 — rewrite의 +0.0009 CI∋0은 레버가 약한 것이 아니라 **삭제형 재작성을 원문에 덧붙이는 것이 tf 재가중 이상을 못 하기** 때문이다. 비교 대상 "When Rules Learn"의 규칙은 전부 확장이다. 법률 요소는 `legal_refs`로 **metadata view의 채점에만** 쓰이고 질의 용어로는 쓰이지 않는다 — 단 그 필드는 쿼리 텍스트가 아니라 **어노테이션(`Crime`)에서 왔으므로 누출 논점이 있다** | §3 |
| **파악된 구현 사실 둘** | 코드에서 확인했고 문서에 없었다. (1) **party-strip에 독립 가중치가 없다** — `retrieve_multiview`가 `"swap": cfg.weight_semantic`로 묶는다. §3 B의 "가중치 4개"는 실질 3개이고 JOURNAL §47의 깊이별 측정이 전부 이 묶임 아래 값이다. (2) **swap view 내부가 `sw_sem + sw_kw` 단순 연결**이라 RRF rank가 BM25 절반에 대해 `entity_swap_semantic_k`만큼 밀려 시작한다 — 지금은 0이라 무해하지만 켜면 조용히 페널티 | §3 |
| **레시피라는 장치** | **세 타입으로 분리 (2026-11-16).** `Recipe` 하나가 서로 다른 셋을 담으려 하고 있었다. ② **노드** `dict[field, value]` / ③ **Bundle** = 간선 `frozenset[field]`(값 없음) + **Constraint** = 금지 상태 / ③′ **Rule** = **질의 재작성 규칙 텍스트**. 셋째는 `RetrievalConfig`에 자리가 아예 없어 `query_rewrite_repeat` 정수 하나로 눌려 있었고, **비교 대상 "When Rules Learn"이 진화시키는 대상이 바로 그것**이다. 승인 경로도 갈린다 — Bundle은 **2×2 상호작용 항 ≠ 0**(필드가 둘 이상이라 정의 가능), Rule은 **효과 자체**(필드가 없어 상호작용을 정의할 수 없다). `admit()` 하나로 둘을 재려 한 것이 설계 오류. 손으로 쓴 5장 → **Bundle 1 + Constraint 1 + Rule 1**. **코드 반영 완료** — Rule은 `query_rewrite`, Constraint는 `search()`, Bundle은 shim+cap seam(§3) | §3, JOURNAL §46 |
| **Rule(③′)이 푸는 것** | **①을 넓히는 합법적 통로가 이것 하나다.** 규칙은 필드가 아니라 텍스트라 공간이 유한하지 않고, LLM이 규칙을 쓰면 **등록부에 새 입력 좌표가 생긴다** — §10.0의 *"루프가 할 수 있는 일의 크기가 ②의 공간에 묶여 있고 그 공간은 ①이 정한다"* 에 대한 유일한 답이다. 등록부의 빈칸(확장 레버)도 필드로는 못 만들고 규칙으로만 만들어진다. 대신 **비용 구조가 바깥 축**이다 — 규칙을 바꾸면 재작성 캐시 재생성(쿼리당 LLM 1회)이라 라운드당 레버가 될 수 없다. 비교 가능성은 **pool을 네 데이터셋에 공유**하고 쓸지 말지만 ②가 정하게 해서 지킨다 | §3 |
| 레시피 | **자격 요건이 바뀌었다 (2026-11-16): 상호작용 항이 0과 구분될 때만 카드다** → 손으로 쓴 5장 중 **1장만 남는다**(§3 "레시피와 action space의 분리"). 아래는 그 전의 상태다. **5장(데이터셋에 따라 4장), 카드당 메커니즘 하나.** 카드는 **구조만** 말하고 **크기(가중치·깊이)는 LLM이 고른다** — 상수를 박으면 데이터셋마다 다른 최적값을 원리적으로 못 맞춘다. 32칸 중 주장 가능 10칸이고 어느 카드도 4개 중 2개를 못 넘긴다. seed로 낸 성적을 self-evolution의 성과로 읽으면 순환이다(§10 경고) | §9, JOURNAL §18·§27·§28 |
| 레시피 승인 | paired bootstrap CI 하한 > 0 **AND** 평균 > **그 데이터셋의 측정된 noise floor** **AND** 종점 > θ₀. 0.003은 LoCoMo 상수이고 넷 중 셋에서 floor 아래였다 — `admit(threshold=)`는 **키워드 전용·기본값 없음**이라 호출자가 `noise_floor(dataset)`를 넘겨야 한다. 단 floor는 full train에서 쟀고 승인은 train-dev(20%)에서 하므로 **하한**이다; 실제로 묶는 것은 dev 델타로 bootstrap한 CI다. 마지막 조건이 빠져 있어 LeCaRD에서 **+15.65 pp**를 주장한 카드의 종점이 θ₀보다 낮았다 | JOURNAL §21·§28 |
| 카테고리별 설정 | **요구사항이다** — 실제 판례 검색은 죄종별로 들어가고, 카테고리는 데이터셋마다 다르다. 측정이 뒷받침: 같은 레버 부호가 클래스마다 갈리고(−0.34 ~ +0.012) 조립이 val에서 +0.0079 대 전역 +0.0012 | §6, JOURNAL §15 |
| 쿼리당 LLM | `intent_planning` 계열은 −8.8 pp이고 **쿼리당 LLM 0회라는 구조를 깬다** | §3 |
| 프롬프트 언어 | 진단 프롬프트를 **코퍼스 언어**로 맞춘다(zh/zh/ko/en). 템플릿·레시피 설명·런타임 블록 셋 다. 식별자·enum·JSON 키·측정 수치는 세 언어에서 동일하게 유지 — 번역하면 검증기가 '실패한 아이디어'로 기록한다 | JOURNAL §29 |
| **논문과의 이탈** | Eq. 1의 **recency 가산항이 구현에 아예 없었고**(그래서 rrf에서 `time_decay`가 무효였다 — 구현이 논문과 달랐던 것), 갱신 규칙이 논문 Eq. 4보다 **엄격**했다(최고 대비 vs 이전 대비, 평평한 라운드를 되돌림 vs 유지). 둘 다 선택지로 추가 — `recency_weight`(기본 0.0) · `--update-rule {elitist,paper}` | JOURNAL §45 |

---

## 0. 용어

### 0.1 영어를 그대로 쓰는 이유

**이 프로젝트의 세 문서 전부가 학술 용어와 코드 식별자를 영어로 둔다** —
METHODOLOGY, [JOURNAL](JOURNAL.md), [PROGRESS](PROGRESS.md). 억지로 옮기면 평소
쓰지 않는 단어가 되고, 논문이나 코드와 대조할 때 다시 영어로 되돌려야 한다.
문서마다 표기가 다르면 그 되돌리기를 문서를 옮길 때마다 해야 한다.

영어로 두는 말:

| | |
|---|---|
| 검색 구조 | `view`, `fusion`, `pool`, `grid`, `sweep` |
| 진화 루프 | `self-evolution`, `round`, `step cap`, `action space`, `elitist`, `trigger`, `recipe`, `seam` |
| 평가 | `fold`, `noise floor`, `paired bootstrap`, `pp` |

표준 한국어가 이미 있는 것은 한국어로 쓴다 — 재현율, 정밀도, 순위, 가중치,
신뢰구간, 깊이, 증상, 정체.

예외 둘: **레시피**와 **카드**는 한국어 표기를 쓴다(이미 일상 외래어이고 세 문서에서
일관된다). 그리고 **코드 식별자는 번역 대상이 아니다** — `weight_semantic`,
`fusion_mode`, `"rrf"`, JSON 키는 프롬프트에서도 영어로 고정한다(§0.4, JOURNAL §29).

### 0.2 핵심 용어

| 용어 | 정의 |
|---|---|
| **θ₀** | 비교 기준이 되는 출발 설정. **BM25 단독**(`fusion_mode="keyword_only"`, 보조 view 전부 off). 코드상 `evaluate.weak_config()`이고 evolution loop의 R0이다 |
| **lever** | 성능을 움직이는 설정 하나 또는 한 bundle. 측정 단위. "11개 lever를 측정했다" = 11가지 조작을 각각 켜 보고 Δ를 쟀다 |
| **action space** | evolution engine이 바꿀 수 있는 필드 전체. `RetrievalConfig`는 45필드지만 **순위를 실제로 바꾸는 것은 12개**이고 나머지는 제안 대상에서 제외했다. 데이터셋에 따라 11개로 더 줄어든다 — §3, JOURNAL §26·§37 |
| **view** | 후보를 뽑는 방식 하나. BM25 / semantic / metadata / party-strip 넷 |
| **fusion** | view의 결과를 하나의 순위로 합치는 방식. `first_found` / `weighted_sum` / `rrf`(k=60) / `*_only` |
| **recipe (= card)** | `Recipe` 인스턴스 1개. **(증상 → trigger → 함께 움직여야 하는 필드들)**. action space의 필드 여러 개를 **step cap 1칸**으로 묶는 단축키이지 별도의 action space이 아니다. **크기(가중치·깊이)는 담지 않는다** — LLM이 고른다 → §9 |
| **recipe pool** | 어댑터별 recipe 목록. `RECIPE_POOLS["legal_retrieval"]`. 현재 **5장**이고, rewrite 캐시가 없는 데이터셋에서는 **4장**으로 줄어든다 — JOURNAL §33·§37 |
| **trigger** | 그 recipe가 지금 제시될 조건. `trigger(report, config, history) -> bool`. card는 **자기 trigger가 받아들이는 상태에서 측정**되므로 그 Δ는 **조건부**다 — 가산이 아니다 |
| **step cap** | 한 라운드에 바꿀 수 있는 필드 수 상한. `max_changes_per_round=2`. recipe는 필드 몇 개를 묶어도 **1칸** |
| **elitist** | 제안을 적용해 보고 **acceptance threshold 이상 오르지 않으면 되돌린다**. 단조 개선을 보장하는 대신 탐색을 포기한다 — §7 |
| **acceptance threshold** | 라운드가 현재 최고 상태를 이겨야 하는 폭. 엔진 기본값 0.003은 LoCoMo의 척도에서 온 상수였고 **우리 fold 셋에서 noise floor보다 낮았다**. 이제 **데이터셋별 측정값**: 0.0240 / 0.0050 / 0.0052 / 0.0018 — JOURNAL §35 |
| **noise floor** | 그 fold에서 **잡음과 구분 가능한 최소 Δ**. 검색은 결정론적이라 실행 간 분산이 없다. 불확실성은 "이 fold가 질의의 한 표본"이라는 것이고, **paired bootstrap** CI95 반폭의 중앙값으로 잰다 |
| **paired bootstrap** | 두 설정을 **같은 질의들**에 돌려 질의별 차이를 재표집한다. 차이의 분산이 점수 자체의 분산보다 훨씬 작으므로, 비교에는 반드시 이쪽을 쓴다 (`recipe_learning.bootstrap_ci`) |
| **fold** | train(loop가 점수를 보는 곳) / val(설정을 고르는 곳) / test(**이미 한 번 채점됐다 — 위 "test fold" 행**) — §5 |
| **Δ / pp** | Δ는 θ₀ 대비 NDCG@30 차이. `pp` = ×100. `+1.3 pp` = `+0.013` |
| **seam** | 레포를 포크하지 않고 **런타임에 함수·설정을 갈아 끼우는 지점**. 예: `EV.MultiViewIndex`를 우리 팩토리로 치환, `DIAG.set_failure_summary_fn()`. 엔진 코드는 그대로 두고 동작만 바꾼다 — §2·§7 |
| **per-category override** | 쿼리 클래스별로 다른 설정을 쓰는 기능. "클래스"는 각 벤치마크의 공식 분류 체계 — §6 |
| **Axis A / Axis B** | 쿼리 분류 축. A는 **법률 분류**(형법 장 · 공식 카테고리 · 법 분야), B는 BM25 점수 첨예도로 만든 난이도 — **B는 폐기** — §6 |

### 0.3 누가 무엇을 정하는가 — 세 층

이 시스템의 결정은 셋으로 갈리고 **LLM이 만지는 것은 가운데 하나뿐**이다. 어느 층에
속하는지가 그것을 바꾸는 방법과 바꿀 수 있는 사람을 정한다.

| | 층 | 무엇을 정하나 | 누가 | 어디에 사나 |
|---|---|---|---|---|
| **①** | **view의 정의** | view가 **무엇인지** | **사람** (근거는 측정) | 모듈 전역 · 환경변수 · `resolve_config` — **`RetrievalConfig` 필드가 아니다** |
| ② | **설정** | view를 **얼마나** 쓸지 | **LLM** (라운드당 2칸) | `action space` 12필드 |
| ③ | **recipe** | 어느 필드를 **함께** 움직일지 | 사람이 구조, **LLM이 크기와 호출 시점** | `RECIPES_LEGAL` |

②와 ③의 구분은 §3에 있다. 여기서는 ①을 적는다 — 그것이 **의도적으로 LLM에게서
감춰져 있다**는 것이 설계이고, 그 이유가 한 곳에 없었다.

#### 먼저: 전제를 정확히

*"사람이 ①에 관여하면 LLM이 모든 것을 한다는 전제가 틀린 것 아닌가"* — 맞는 지적이고,
**그 전제는 세운 적이 없다.** 이 프로젝트의 전제는 *"LLM self-evolution engine은
검증 대상이 아니라 전제이고, 보여야 할 것은 **데이터셋마다 다른 설정값에 스스로
도달하는가**"* 다(JOURNAL §28). 논문도 같다 — self-evolution loop가 바꾸는 것은
retrieval **정책**이고 아키텍처가 아니다.

따라서 올바른 주장은 **"루프가 ① 고정 아래에서 ②를 진화시킨다"** 이고, 거기에
**"이 프로젝트에서 가장 큰 개선은 ①에서 나왔다"** 를 함께 적어야 한다(§10의 귀속 표).
그것을 빼면 §10이 경고하는 과대 보고가 된다.

#### ①은 한 덩어리가 아니다 — 세 종류의 정당성이 다르다

| | 종류 | 사람이 정해도 되는가 |
|---|---|---|
| **A** | **실험의 틀** | **그렇다.** LLM이 지표를 고르면 과제가 아니라 **지표를 최적화**한다 |
| **B** | **고정된 백본** | 그렇다 — §25가 이미 "의도된 설계"로 적었다. 토크나이저가 데이터셋마다 다르면 비교가 성립하지 않는다 |
| **C** | **아키텍처 결정** | **여기가 문제다.** ②와 같은 공간의 결정인데 손으로 했다 |

**A — 실험의 틀**

| 결정 | 현재 값 | 어디 |
|---|---|---|
| 주 지표·채점 깊이 | NDCG@30 · top-30 | `evaluate.PRIMARY` · `EVAL_K` |
| fold | 각 벤치마크 공식 분할 | `evaluate.in_split` |
| 반환 순위의 상한 | `max_context >= top_n` **강제** | `search()`가 아니면 `ValueError` |

**B — 고정된 백본**

| 결정 | 현재 값 | 어디 |
|---|---|---|
| 언어별 토크나이저 | jieba / mecab / spaCy | `legal_retriever` (대체 시 예외) |
| RRF의 k | 60 | 벤더 `multi_retriever._fuse` |
| 임베딩 모델 | bge-m3 | `build_embeddings` |

**C — 아키텍처 결정** (각 항목에 "왜 닫혀 있나"와 "열면 얼마가 드나")

| 결정 | 현재 값 | 왜 닫혀 있나 | 열면 |
|---|---|---|---|
| **색인하는 판결문 절** | `Fact`+`Reason`+`Result` | 비교 가능성뿐이다. **측정값은 열라고 말한다** — 상대 +26%/+62% | 라운드당 레버는 불가(코퍼스·임베딩 재생성). **데이터셋당 한 번 고르는 외곽 축**은 가능. 비용 = 절 조합 × 재임베딩(≈57분) |
| **semantic view의 입력** | 행의 원문 질의 | 열면 **묶음**이 된다 — 끄면 `enable_query_rewrite`가 semantic view를 함께 끈다(§43). 그리고 데이터셋마다 다르면 다른 시스템이 된다 | 선계산 800건(LeCaRDv2 rewrite)으로 **한 질문만** 측정 가능 — JOURNAL §44 |
| **삭제 대상의 추출 출처** | 행의 원문 질의 | 재작성본에서 뽑으면 절 조각이 섞인다(300건 중 37건) | 수리이므로 되돌릴 이유가 없다 |
| **party-strip의 삭제 범위** | **인명 + 사명 + 지명** (`STRIP_SCOPE=parties+org+place`, 기본값) | **과제 정의가 지표보다 우선이라는 결정.** 같은 지역·같은 회사가 걸려 나오는 것은 유사 판례가 아니다. 지표는 반대로 보상한다(정답이 쿼리의 지명을 **+51pp** 더 공유) — 그 괴리를 숨기지 않고 두 숫자를 함께 보고한다. 클래스마다 다른 범위를 쓰면 클래스 간 비교가 깨지므로 **범위는 전역 하나**다(§16). 법원명은 넓은 범위에서만 (`+court`) — metadata view의 '같은 법원' 신호로 쓰인다 | `STRIP_SCOPE` 환경변수, 데이터셋당 한 패스 |
| **party-strip의 semantic 절반** | 끔 | 켜면 주 semantic view와 **같은 ranking을 두 번** 센다(§44) | semantic 입력을 열어야 비로소 의미가 생긴다 |
| 임베딩 길이 | 8192 | 512는 **결정이 아니라 잔존물**이었다(§27). 8192가 모델의 한도 | 이미 올렸다 |
| **최신성(날짜)** | **쓰지 않는다** (`recency_weight=0.0`) | **관련성 신호가 아니다.** 라벨은 두 사건이 **닮았는가**를 말하고, 판결문은 최근이라고 더 닮지 않는다. 논문이 최신성을 쓰는 이유는 LoCoMo가 **시간 순서가 있는 대화**에 시간 질문을 하기 때문이고, 우리 질의는 사실관계를 닮은 선례를 묻는다 | 구현은 돼 있다(§45). 실무용 목적함수(개정법으로 폐기된 선례 등)는 **다른 과제**이고, 그때는 레버와 척도가 준비돼 있다 |

C의 첫 줄이 당신 지적의 핵심이다. **`DOC_FIELD_OF`는 수리도 틀도 아니고, ②와 같은
공간에서 손으로 내린 결정인데 루프가 찾은 어떤 것보다 한 자리 크다.**

#### 왜 ①을 LLM에게 주지 않는가

§16이 `STRIP_SCOPE`에 대해 적어 둔 이유가 전부에 적용된다:

> 액션 공간에 넣으면 §15의 클래스별 진화에서 A2는 인명만 지우는 view를, A5는 전부
> 지우는 view를 갖게 되고 **두 클래스의 숫자가 서로 비교 불가능해진다.**

같은 논리가 데이터셋 사이에도 걸린다. ①을 열면 데이터셋 A와 B가 **다른 시스템**이
되고, "같은 파이프라인으로 넷을 똑같이 돌린다"는 비교의 전제가 깨진다. 그러면
"데이터셋마다 다른 값에 도달했다"가 **설정의 차이인지 시스템의 차이인지** 구분되지
않는다.

그리고 ①의 일부는 **묶음**이다 — `SEMANTIC_USES_ROW_TEXT`를 끄면 `enable_query_rewrite`를
켜는 것이 semantic view를 **함께 끄는** 일이 된다(JOURNAL §43). 설정으로 열면 LLM이
자기가 호출하지 않은 변경을 겪는다.

#### 우회로도 막혀 있다

`per_category_overrides`는 임의의 필드를 담을 수 있는 유일한 자리인데, 그 안쪽에도
같은 필터가 걸린다(`diagnosis.py`, *"inert here; same rule as top level"*). 실측:

```
제안: {entity_swap_semantic_k: 30, max_context: 10,
       per_category_overrides: {1: {entity_swap_semantic_k: 30,
                                    max_context: 10, weight_semantic: 0.2}}}
통과: {per_category_overrides: {1: {weight_semantic: 0.2}}}
```

#### ①을 바꾸는 방법

측정뿐이다. 그리고 바꾸면 **그 축이 ON인 과거 측정이 전부 무효**가 되므로, 무엇이
무효가 되는지 함께 적는다 — JOURNAL §44의 "다시 재야 하는 과거 측정" 표가 그 예다.

### 0.4 EvolveMem 논문의 키워드 → 우리 것

논문은 **memory 구축**과 **self-evolution** 두 덩어리로 돼 있다. 우리는 뒤쪽만
쓴다. 패키지 모듈 25개 중 실제로 import하는 것은 **5개**다 — `evolution`,
`diagnosis`, `evolution_cookbook`, `multi_retriever`, `benchmarks`.

**(가) 그대로 쓰는 것 — 이름도 개념도 유지**

| 키워드 | 우리에게서의 의미 |
|---|---|
| **self-evolution loop** | 라운드마다 설정을 고치는 outer loop. 그대로 |
| **diagnosis** | 실패를 LLM에게 보여주고 설정 변경을 받는 단계. 그대로. 단 프롬프트는 전부 다시 썼다(§0.4 (다)) |
| **recipe / cookbook** | 증상에 걸린 설정 bundle. 개념 유지, 내용은 전부 우리가 측정한 것으로 교체 |
| **elitist / round / step cap** | 그대로 |
| **multi-view retrieval** | 그대로. view의 정체만 바뀌었다(아래 (나)) |

**(나) 이름을 바꾼 것 — 내용이 달라서**

| 논문·엔진 | 우리 | 왜 |
|---|---|---|
| **memory** | **document** (판례 1건 / 조문 1개) | 논문의 memory는 LLM이 대화에서 **추출·가공한 조각**이다. 우리 것은 가공하지 않은 **판결문 원문**이다. 디스크도 `memories/` → `corpus/`, `{ds}_memories.json` → `{ds}_cases.json` |
| **QA pair** | **query** | 답을 생성하지 않는다. `answer` 필드는 엔진이 요구해서 남아 있을 뿐이고 채점에 쓰이지 않는다 |
| **F1** (`overall_f1`, `r.f1`) | **NDCG@30** | 필드 이름은 엔진 것이라 그대로지만 **담긴 값은 NDCG@30**이다. 로그에서 `f1`을 보면 NDCG로 읽어야 한다 |
| **category** (LoCoMo 질문 유형 5종) | **legal class** (형법 장 / 공식 카테고리 / 법 분야) | 숫자 키는 유지하되 의미가 완전히 다르다. 그래서 프롬프트가 `[1] 제17장 성범죄`처럼 **키와 이름을 함께** 찍는다 — JOURNAL §30 |
| **entity** | **legal_refs** (법조문 인용 + 죄명) | 대화의 "엔티티"는 화자가 언급한 주제다. 우리 것은 법률 참조다 |
| **person** | **parties** (피고인·피해자·원고) | — |
| **structured view** | **metadata view** | 하는 일은 `legal_refs` 일치 **개수**로 점수를 주는 것이다 |
| **entity-swap view** | **party-strip view** | "엔티티를 교체"로 읽히지만 실제로는 **이름을 지우고 재검색**한다 |
| **max_context** | (의미 변경) 답변 LLM에게 넣을 메모리 수 → **반환 순위의 길이 상한** | 답변 생성이 없으므로 "context"가 아니다. 100으로 두고 채점은 top-30에서 한다 |
| **expected_lift_pp** | (규칙 변경) 손으로 적은 기대 이득 → **해당 데이터셋 train 측정이 있을 때만 출력** | 손으로 적은 값이 중요도와 반대로 정렬돼 LLM의 선택을 오도했다 — JOURNAL §31 |

**(다) 버린 것 — 쓰지 않는다**

| 논문 키워드 | 왜 버렸나 |
|---|---|
| **memory extraction** (`MemoryExtractor`, `window_size`, `overlap`) | 판결문을 LLM으로 조각내지 않는다. 메타데이터 추출은 **정규식으로 오프라인**(`legal_meta.py`) |
| **consolidation / promotion / replay / policy store / mini-KG** | 메모리 생애주기 관리. 코퍼스가 고정이므로 할 일이 없다 |
| **answer generation / verification** (`answer_style`, `enable_answer_verification`, `verification_style`, `answer_model*`) | 순위가 출력이다. 이 필드들은 action space에서 제외했고 프롬프트에서도 지웠다 |
| **intent planning / query decomposition / coverage reflection** | **질의당 LLM 호출**이 필요하다. 구조를 깨고, 측정값도 −8.8 pp였다 — §3 |
| **LoCoMo prompt-surface flags** (`locomo_cat*`) | LoCoMo 전용. 전부 제외 |
| **meta-analyzer** (Phase 7b) | 매 라운드 **두 번째 LLM 호출**을 하는데 elitist가 결과를 전부 버린다. 껐다 — JOURNAL §34 |

**(라) 우리가 새로 넣은 것 — 논문에 없다**

| 우리 키워드 | 무엇 |
|---|---|
| **noise floor** | fold별 측정 하한. acceptance threshold를 여기서 받는다 — JOURNAL §35 |
| **state_notes** | "이 값은 라이브러리 기본값이고 아무도 고르지 않았다"를 **값도 방향도 없이** 알리는 프롬프트 블록. 상수를 박은 card 넷을 대체 — JOURNAL §33 |
| **inert_report** | ON인데 한 번도 작동하지 않은 lever를 명시. +0.0000이 **반증이 아님**을 말한다 — JOURNAL §13 |
| **view_report** | 어느 view가 무엇을 찾았는지, `sole`(그 view만 찾은 정답 수)까지 |
| **budget block** | acceptance threshold와 **남은 라운드 수**를 프롬프트에 명시 — JOURNAL §37 |
| **prompt language** | diagnosis prompt를 코퍼스 언어로(zh/zh/ko/en) — JOURNAL §29 |
| **verify.py** | 조립된 프롬프트에 대한 단정. 조각 검사는 프롬프트 검사가 아니다 — JOURNAL §38 |

### 0.5 우리 이름 ↔ 엔진 이름

EvolveMem은 **대화 메모리** 어휘로 쓰여 있다(`persons`, `entities`, `structured`,
`entity-swap`). 판례에서는 이 이름들이 실제 내용과 어긋나 읽을 때마다 오해를 만들어,
**우리 층은 우리 이름을 쓰고 엔진 경계에서만 변환한다**(`legal_retriever.py`의 어댑터 3개).

| 우리 이름 | 엔진 이름 | 내용 | 왜 바꿨나 |
|---|---|---|---|
| `parties` | `persons` | 당사자 — 피고인·피해자·원고 | — |
| `legal_refs` | `entities` | 법조문 인용 + 죄명 + `art:NNN` | 대화 메모리의 "엔티티"는 화자가 언급한 주제다. 우리 것은 법률 참조다 |
| `cases` / `corpus/` | `memories` | 판례 1건 = 검색 단위 | §3.1(LLM 추출)을 버렸으므로 우리 것은 가공된 "메모리"가 아니라 **판결문 원문**이다 |
| **metadata view** | structured view | `legal_refs` 일치 **개수**로 점수 | — |
| **semantic view** | semantic view | bge-m3 벡터의 cosine similarity | **이름은 엔진 것을 그대로 쓴다.** 검색 문헌의 표준 용어는 `dense`(sparse BM25의 반대말)이고 이 문서도 한동안 두 이름을 섞어 썼지만, 바꿀 수 없는 쪽이 `semantic_top_k`·`weight_semantic`이다 — 그것이 action space이고 **진단 LLM이 그 이름으로 제안한다.** 문서가 다른 단어를 쓰면 문서와 LLM이 어긋난다. 우리가 지은 식별자도 맞췄다: `SEMANTIC_USES_ROW_TEXT`·`semantic_query` |
| **party-strip view** | entity-swap view | 쿼리에서 `parties`를 **지우고** 재검색 | "엔티티를 교체"로 읽히지만 실제로는 삭제 후 재검색이다 |

**설정 필드 이름은 엔진 것을 그대로 쓴다.** 그것이 진화 엔진의 action space 자체이고,
진단 LLM과 레시피가 이 이름으로 제안하기 때문이다.

| 필드 | 어느 view |
|---|---|
| `semantic_top_k`, `weight_semantic` | **semantic view** |
| `structured_top_k`, `weight_structured` | **metadata view** |
| `enable_entity_swap`, `entity_swap_semantic_k`, `entity_swap_keyword_k` | **party-strip view** |

구버전 파일(`persons`/`entities` 키)도 읽을 수 있다 — `parties_of()` · `legal_refs_of()`가
양쪽을 받는다.

**엔진이 소유한 이름은 그대로 둔다**: `RetrievedMemory`(클래스), `index.memories`(속성),
`initial_memories=`(kwarg). 레포 전역이라 바꾸면 회귀 위험이 크고, 여기 매핑 한 줄로 족하다.

디스크도 같이 바꿨다: `memories/` → **`corpus/`**, `{ds}_memories.json` → **`{ds}_cases.json`**,
`{ds}_memories_m3.npy` → **`{ds}_cases_m3.npy`**, `build_memories.py` → **`build_cases.py`**,
`MEM_DIR` → **`CASE_DIR`**. `doc_id`와 중복이던 `memory_id` 필드는 **제거**했다(읽기 폴백은
남겨 구버전 파일도 열린다).

---

## 1. 과제 정의

EvolveMem은 **대화 메모리 QA**(검색 → 답변 생성 → token-F1)를 전제한다.
판례 검색은 **랭킹**(검색 → 순위 → NDCG@30)이다. 세 층 중 쓰는 것은 하나다.

| 논문 §  | 층 | 우리 쪽 |
|---|---|---|
| §3.1 | 메모리 저장소 (LLM 추출·통합·타입 메모리) | **버림.** 판례는 이미 검색 단위다 |
| §3.2 | 검색 층 | **action space로만** 사용 |
| §3.3 | **self-evolution 엔진** | **이것이 목적** |

§3.1을 버린 대가로 §3.1의 품질 관리(coverage verifier)도 같이 잃었고, 나중에 추출 실패를
직접 진단해야 했다(§5의 "추출" 항목).

---

## 2. 검색 프레임워크

### 루프가 두 개다

이것이 전체 그림이다. **바깥 루프가 안쪽 루프의 설정을 바꾼다.**

```
┌─ 바깥: 진화 루프 (라운드 단위, LLM 1회/라운드) ────────────────┐
│                                                              │
│   θ(현재 설정) ──┐                                            │
│                 │                                            │
│   ┌─────────────▼──── 안쪽: 검색 루프 (쿼리 단위, LLM 0회) ──┐ │
│   │  for 쿼리 in train(576개):                              │ │
│   │      cat   = 쿼리["category"]        ← 그 쿼리의 죄종    │ │
│   │      θ_eff = resolve_config(θ, cat)  ← 카테고리 설정 적용 │ │
│   │      순위  = retrieve_multiview(쿼리, θ_eff)             │ │
│   │      점수  = NDCG@30(순위, 정답)                         │ │
│   └──────────────────────┬──────────────────────────────────┘ │
│                          │ 평균 NDCG@30 + 실패 사례            │
│                   LLM 진단 1회                                │
│                          │ θ' = θ + (최대 2개 필드 변경)       │
│                   가드: θ'가 +0.003 이상 못 오르면 되돌림       │
│                          └→ 다음 라운드                       │
└──────────────────────────────────────────────────────────────┘
```

### 카테고리가 들어가는 지점 — 쿼리 단위다

위 그림의 `resolve_config` 한 줄이 §6의 요구사항이 실행되는 자리다. **설정 θ 하나가
모든 쿼리에 똑같이 적용되는 것이 아니다.**

θ는 `per_category_overrides`라는 필드를 들고 다닌다 — `{카테고리: {바꿀 필드: 값}}`
형태의 맵이다. 쿼리마다:

```python
cat   = query_row["category"]          # 이 쿼리의 죄종 (§6의 분류)
cfg   = resolve_config(config, cat)    # per_category_overrides[cat]를 config에 덮어씀
index.set_chapter(query_row["legal_class"] if cfg.enable_chapter_filter else None)
hits  = retrieve_multiview(index, text, config=cfg, category=cat)
```

세 가지를 짚어 둔다.

**1. 덮어쓰기는 차이분만 담는다.** 전체 설정을 카테고리마다 복제하지 않고 base 대비
**다른 필드만** 넣는다. 개선이 없던 클래스는 override가 없고 자동으로 base로 떨어진다.
`assemble_percat.py`가 클래스별 진화 결과를 이 형태로 바꿔 넣는다.

**2. 적용 후 맵을 비운다.** `resolve_config`가 돌려주는 설정의
`per_category_overrides`는 `{}`다. 엔진이 같은 필드에서 override를 **다시** 해석하므로,
비우지 않으면 이미 평탄화된 설정에 두 번 적용된다 — 운 좋으면 무효, 나쁘면 나중 레시피가
덮어쓴 값을 되돌린다.

**3. `resolve_config`가 `retrieve_multiview` 안이 아니라 밖에 있다.** chapter filter와
쿼리 재작성이 **해석된** 설정을 봐야 하기 때문이다. 엔진 안에서 pool면 그 두 단계가 base
설정으로 돌아간다.

**LLM은 검색을 하지 않는다.** 라운드가 끝난 뒤 성적표를 보고 "다음엔 이 설정을 이렇게 바꿔봐"라고
말하는 역할만 한다. 검색 자체는 BM25와 임베딩 내적으로만 돌아간다 — 쿼리 640개 × 7라운드 =
**4,480회 검색에 LLM 호출은 7회**다.

원본 EvolveMem은 검색 후 LLM이 답변을 생성하고 token-F1로 채점하므로 **쿼리마다** LLM을 부른다
(논문이 "라운드당 15~20분"이라 한 비용). 우리는 랭킹 자체가 출력이라 `_generate_answer`가
doc_id를 반환하게 덮어 그 비용을 없앴다(§7).

### 안쪽 루프 — 검색 한 번이 도는 방식

`retrieve_multiview()` (`multi_retriever.py:673`), 쿼리 하나당 6단계.

**① per-category override 해석** — 쿼리의 법률 클래스에 전용 설정이 있으면 여기서 갈아탄다.

**② 쿼리 분석** — `analyze_query(question)`이 `keywords` / `parties` / `legal_refs`를 뽑는다.
**영어 전용 가정이 세 겹**이라 그대로는 쓸 수 없다(`multi_retriever.py:501`):

```python
words = query.split()                                      # ①
capitalized = [w for w in words if w[0].isupper() ...]     # ②
keywords = [w.lower() for w in words if len(w) > 2 ...]    # ③
return {"keywords": keywords[:15], "persons": capitalized, "entities": capitalized}
```

| | 전제 | 중국어·한국어에서 |
|---|---|---|
| ① `.split()` | 공백이 단어를 나눈다 | 중국어는 **문서 전체가 1토큰** |
| ② `isupper()` | 대문자 = 고유명사 | **대소문자가 없다** → `persons`·`entities`가 **항상 빈 리스트** |
| ③ `len > 2` | 3자 미만은 불용어 | 한자 2자가 핵심 단어 |

게다가 `persons`와 `entities`에 **같은 리스트**를 넣는다 — 원본은 둘을 구분하지 않는다.

#### "사전 추출 메타데이터"란

`build_cases.py`가 `legal_meta.py`의 규칙 기반 추출(정규식)을 **쿼리 본문에도** 돌려
`corpus/{ds}_queries.json`에 저장해 둔 것이다. 실제 한 건:

```
LeCaRDv2
  text       : 公诉机关指控，2019年5月31日22时许，被告人姚帝锋预谋抢劫后来到重庆市北碚区…
  parties    : ['姚帝锋', '王某']
  legal_refs : ['art:23', 'art:263', 'art:67', 'art:72', 'art:73']
  legal_class: A5_재산 | category: 1 | fold: train

LEGAR
  text       : 범죄사실 [범죄전력] 피고인 C는 2010. 7. 6. 광주지방법원에서…
  parties    : ['들의', 'C', 'A', 'B', 'AJ']
  legal_refs : ['각 형법 제313조', '각 형법 제70조', '각 형사소송법 제334조', …]
```

**검색 시점이 아니라 빌드 시점에 한 번** 뽑는다. 진화는 같은 쿼리를 라운드마다 다시 검색하므로
(640쿼리 × 7라운드) 매번 정규식을 다시 돌릴 이유가 없다. 문서 쪽도 같은 추출기로 처리하므로
쿼리와 문서가 **같은 어휘**를 쓴다(§3 설계 결정).

#### 주입 방식

`MR.analyze_query`를 통째로 교체한다(`legal_retriever.py:76`):

```python
def _legal_analyze_query(query: str) -> dict:
    row = _CURRENT.get("row") or {}      # 지금 검색 중인 쿼리의 행
    return {
        "keywords": [query],             # 토큰화하지 않고 통째로 넘긴다
        "persons":  parties_of(row),     # 엔진 쪽 키 이름 (§0 어휘표)
        "entities": legal_refs_of(row),
    }
```

함수 시그니처가 **문자열 하나뿐**이라 쿼리 행을 인자로 받을 수 없다. `query_context(row)`
contextmanager로 "지금 처리 중인 행"을 꽂아 두고 꺼내 쓴다.

`keywords`를 토큰화하지 않고 원문 그대로 넘기는 것도 의도다 — 토큰화는 인덱스의
`keyword_search`가 언어별 토크나이저로 한다. 여기서 `.split()`하면 중국어가 다시 1토큰이 된다.

#### 이 교체가 없으면 view 둘이 죽는다

| | 원본 | 교체 후 |
|---|---|---|
| `parties` 추출률 (LeCaRDv2 쿼리) | **0%** | 95.0% |
| `legal_refs` 추출률 | **0%** | 100% |
| metadata view | 항상 빈 결과 | 작동 |
| party-strip view | **절대 발동 안 함**(`persons`가 비어서) | 작동 |

LEGAR의 `parties`에 `'들의'` 같은 조사 노이즈가 섞이는데, 그것이 §2 "persons를 매칭에 쓰지
않는 이유"와 "삭제는 노이즈에 관대하다"의 근거다.

**③ 네 view를 독립적으로 검색** — 각 view가 `[(문서인덱스, 점수), ...]`를 돌려준다.
**점수 스케일이 서로 전혀 다르다** — 이것이 ④에서 문제가 된다.

| view | 무엇으로 검색 | 점수의 정체 | 깊이 |
|---|---|---|---|
| **keyword** | `analysis["keywords"]` → BM25 | BM25 점수, **수천 단위** | `keyword_top_k` |
| **semantic** | 쿼리 임베딩 → 문서 임베딩 내적 | 코사인 유사도, **0~1** | `semantic_top_k` |
| **structured** | `persons`∩`entities` 메타데이터 일치 | **일치 개수** (1, 2, 3…) | `structured_top_k` |
| **swap** | **인명을 지운 쿼리**로 semantic+BM25 재검색 | 위 두 개가 섞여 있음 | `entity_swap_*_k` |

`fusion_mode`가 어느 view를 돌릴지도 정한다 — `keyword_only`면 BM25만 돌고 나머지는 빈 리스트다.
**θ₀가 이 상태다.**

**④ fusion — 네 리스트를 하나의 순위로.** 우리 실험의 핵심 지점이므로 산술을 그대로 적는다
(`_fuse()`, `:606`).

네 모드가 **같은 네 리스트를 받아 완전히 다른 규칙으로** 하나의 순위를 만든다.
공통 입력은 view마다의 `[(문서, 점수), ...]`이고, 공통 출력은 상위 `max_context`개다.

#### `first_found` (레포 기본값) — 점수를 아예 안 본다

```python
seen = set(); ordered = []
for view in ("semantic", "keyword", "structured", "swap"):   # ← 고정된 순서가 전부
    for i, s in view_results.get(view, []):
        if i not in seen:
            seen.add(i); ordered.append((i, s, view))
```

**순서 기준은 코드에 박힌 view 순서다**(`:665`). 점수 비교가 한 번도 일어나지 않는다. view를 이
순서로 훑으면서 처음 만난 문서를 그 자리에 놓고, 이미 놓인 문서는 건너뛴다. 결과는:

| 순위 구간 | 들어가는 것 |
|---|---|
| 맨 앞 | **semantic이 뽑은 전부** (semantic 순위대로) |
| 그다음 | BM25가 뽑은 것 중 **semantic이 놓친 것만** |
| 그다음 | metadata view에서 새로 나온 것만 |
| 맨 뒤 | **party-strip에서 새로 나온 것만** |

예를 들어 `semantic_top_k=100`, `keyword_top_k=100`, `max_context=100`이면 **상위 100개가
전부 semantic**이고 BM25는 한 건도 못 들어간다. 점수가 아무리 높아도 소용없다.

이 한 가지가 우리 측정 두 개를 설명한다.

- **`bm25_swap30` = 정확히 +0.0000** — party-strip이 맨 마지막이라, 앞 view이 `max_context`를
  채우면 swap 결과는 잘려 나가 랭킹에 아예 영향을 주지 못한다. **party-strip은 점수 fusion
  (`weighted_sum`/`rrf`)과 짝지어야만 작동한다.**
- **θ₀를 `keyword_only`로 잡은 이유** — `first_found`에서 semantic을 켜면 BM25 랭킹이 아니라
  semantic 랭킹을 재게 된다. BM25 baseline을 재려면 다른 view를 꺼야 한다.

실측(LeCaRDv2 test):

| 설정 | NDCG@30 |
|---|---|
| `keyword_only` (θ₀) | **0.3948** |
| `semantic_only` | 0.2797 |
| **`first_found`, semantic 100 + BM25 100** | **0.2797** ← `semantic_only`와 소수점까지 동일 |
| `first_found`, semantic 20 + BM25 8 / ctx 25 (LoCoMo 기본값) | 0.2899 |

**BM25를 100건 뽑아 넣었는데 기여가 정확히 0이다.** 즉 `first_found`는 fusion이 아니라
**"semantic, 모자라면 BM25로 채움"**이고, 뒤 view는 앞 view가 자리를 남겼을 때만 보인다.
결정하는 것은 **`top_k`와 `max_context`의 비율** 하나다.

- LoCoMo 기본값(semantic 20 / keyword 8 / ctx 25): semantic이 20칸, BM25가 5칸 —
  **캐스케이드**로 작동한다. 의도된 설계로 보인다
- 우리 설정(각 100 / ctx 100): semantic이 100칸 전부 — **BM25는 없는 것과 같다**

→ **레포의 고정 view 순서는 "semantic이 가장 강한 view"를 전제한다.** LoCoMo에서는 맞지만
판례에서는 **정확히 거꾸로다**(BM25 0.3948 vs semantic 0.2797). 더 나은 랭킹이 있어도
구조적으로 묻힌다.

이것이 `rrf`를 쓸 또 하나의 이유다 — 순위만 쓰므로 **어느 view의 1위든 똑같이 `1/60`**을 받고,
view 간 우열을 코드가 미리 정하지 않는다.

현재 파이프라인은 `first_found`를 타지 않는다(θ₀가 `keyword_only`이고 진화는 R1에서 점수
fusion으로 간다). 다만 **`RetrievalConfig`의 기본값이 `first_found`**이므로, fusion 모드를
명시하지 않은 채 semantic을 켜면 −0.115가 난다. 레시피 `lexical_first_light_semantic`이
`fusion_mode in ("keyword_only", "first_found")`에서 발동해 semantic을 켜면서 **동시에**
점수 fusion으로 전환시키는 것이 결과적으로 이 가드 역할을 한다(§9).

#### `weighted_sum` — view 안에서 정규화한 뒤 가중합

```python
max_s = max(s for _, s in items) or 1.0
norm = (s / max_s) * w
fused[i] += norm
```

**정규화가 view별 최댓값 1회뿐**이다. BM25 수천 점과 structured 일치 개수 3을 같은 [0,1]에 밀어
넣는데 분포 모양이 완전히 달라 공정하지 않다. **구조화 view가 −9.0 pp였던 것이 여기서 나온다** —
일치 개수 1인 문서 수천 건이 전부 0.33을 받아 BM25 순위를 평탄하게 만든다.

#### `rrf` — 점수를 버리고 순위만 쓴다

```python
add = 1.0 / (60.0 + rank) * w
fused[i] += add
```

1위는 1/60, 2위는 1/61 — view가 무엇이든 1위는 똑같이 1/60이다. **스케일 정규화 문제가 원천적으로
없다.** 가중치는 여전히 곱해지지만 곱해지는 대상이 점수가 아니라 순위 기여분이다.

**`60`은 컷오프가 아니다.** 60위에서 자르는 것이 아니라 모든 순위가 기여하는 평활 상수다
(Cormack et al. 2009의 표준값). 다만 순위 차이를 압축한다:

| 순위 | 기여 | 1위 대비 |
|---|---|---|
| 1위 | 1/60 = 0.01667 | 1.00 |
| 100위 | 1/160 = 0.00625 | 0.38 |
| 1000위 | 1/1060 = 0.00094 | 0.057 |

#### 깊은 지표를 보고하려면 먼저 풀어야 할 것 (미착수)

주 지표가 NDCG@30이라 아직 부딪히지 않았지만, R@1000 같은 깊은 지표를 보는 순간 두 가지가
동시에 문제가 된다.

**(a) 진짜 컷오프는 `k=60`이 아니라 각 view의 `top_k`다.** RRF는 view이 내놓은 후보의
**합집합만** 순위 매긴다. 실측(LeCaRDv2 val 60쿼리, 코퍼스 55,192건):

| 설정 | RRF가 순위를 매기는 문서 수 |
|---|---|
| 현재 (top_k 100 / max_context 100) | **100** |
| `max_context`만 1000 | **185** ← top_k가 병목 |
| top_k도 1000 | 1000 |

`max_context`만 키우면 185개에서 멈춘다 — 네 view의 top-k 합집합이 그만큼이기 때문이다.
**깊은 지표를 보려면 view별 `top_k`를 함께 키워야 한다.**

**(b) 그때 `k=60`이 실질 하이퍼파라미터가 된다.** 500위(1/560)와 1000위(1/1060)의 절대 차이는
0.0009다. 두 view가 900위에 올린 문서(0.0021)와 한 view가 400위에 올린 문서(0.0022)가 사실상
동점이 된다. 즉 **깊은 구간에서는 "몇 개 view가 찾았나"가 "어느 순위였나"를 압도한다.**
BM25만 `sole`이 0이 아닌 우리 상황(§8)에서는 **BM25의 깊은 순위 정보가 view 개수에 묻힌다.**

`k`는 `_fuse`에 **하드코딩**되어 있어 action space에 없다 — `first_found`의 view 순서가 차원이
아닌 것과 같은 종류다. 깊은 지표를 보고하기로 하면 `rrf_k`를 차원으로 추가해야 한다.
지금 넣지 않는 이유는 @30에서는 효과가 없고 탐색 공간만 넓히기 때문이다.

#### `*_only` — 한 view만 통과

`keyword_only` / `semantic_only` / `structured_only`. 해당 view의 순위를 그대로 내보낸다.
**θ₀가 `keyword_only`다.**

#### 세 줄 요약

| 모드 | 순서를 정하는 것 | 점수 비교 | view가 늘면 |
|---|---|---|---|
| `first_found` | **고정된 view 순서** (semantic→keyword→metadata→party-strip) | 없음 | 앞 view가 자리를 다 차지 |
| `weighted_sum` | view별 max로 나눈 점수의 가중합 | 있음 (스케일 교란) | 정규화 부담 증가 |
| `rrf` | view별 순위의 역수 합 | 없음 (순위만) | **유리해짐** |

→ 이것이 §8 "fusion 모드는 활성 view 수에 의존한다"의 기계적 이유다. view가 하나면 RRF는 순위를
보존할 뿐이라 이득이 없고(실측 **−0.010**), view가 넷이면 가중합의 정규화 부담이 커져 RRF가
이긴다(**+0.0154**). view를 고정한 채 fusion만 비교해 "RRF는 유망하지 않다"고 접은 오류가 여기서 나왔다.

#### `fusion_mode`는 차원이 아니다 — 분해 (2026-11-16)

위 표를 세로로 읽으면 바로 보인다. **네 값이 같은 종류의 연산이 아니다.**

| enum 값 | 실제 정체 | 이미 어디에 있나 |
|---|---|---|
| `keyword_only` · `semantic_only` · `structured_only` | fusion이 **아니라 view 마스크** | `semantic_top_k=0` · `structured_top_k=0` · `enable_entity_swap=False`로 **이미 표현 가능** |
| `first_found` | fusion이 **아니라 캐스케이드** — 실측상 BM25 기여가 정확히 0이고, 결정하는 것은 `top_k`/`max_context` 비율 하나 | 판례에서는 view 순서가 거꾸로라 **쓸 수 없다** |
| `weighted_sum` / `rrf` | **진짜 fusion** (점수 합 vs 순위 합) | — |

범주형 축이라면 네 수준이 교환 가능해야 한다. 여기서는 하나를 고르는 순간 **무슨 질문을
하고 있는지가 바뀐다.** 그래서 "`fusion_mode`의 효과 +X"라는 수를 낼 수 없다. 이유 넷:

1. **기준선이 축 안에 들어 있다.** `keyword_only`가 곧 θ₀다. 그 수준의 Δ는 정의상 0이고,
   나머지 세 수준의 Δ는 전부 "θ₀ 대비"라 **축 내부 비교가 아니다.**
2. **모드가 다른 필드의 의미를 바꾼다.** `rrf`는 점수를 버리므로 `time_decay_half_life_days`가
   무효가 되고(JOURNAL §42), 가중치가 곱해지는 대상이 점수에서 순위 기여분으로 바뀐다.
   `*_only`는 가중치 전부 무효, `first_found`는 깊이 비율만 산다. 즉 이 필드는 축 위의 한
   값이 아니라 **어느 서브 액션 공간에 들어갈지를 고르는 선택자**다.
3. **그래서 step-cap 1칸 회계가 틀렸다.** 필드 하나이므로 1칸인데, 바꾸는 순간 나머지 필드의
   의미를 재정의한다. 위에 적은 *"fusion 모드를 명시하지 않은 채 semantic을 켜면 −0.115"*가
   그 결과이고, `lexical_first_light_semantic` 카드가 결과적으로 가드 노릇을 하는 이유도
   이것이다 — **레시피가 필요한 것이 아니라 회계가 틀렸다**(§3).
4. **부호가 조건부다.** 활성 view 수와의 상호작용이 주효과보다 크다(−0.010 ↔ +0.0154).

**분해 후의 표기.**

```
fusion_mode ∈ {weighted_sum, rrf}                  # 목표: 2값 — 진짜 fusion만
                                                   # 현재: 5값 (*_only 셋이 남아 있다)
view 마스크  = semantic_top_k>0 / structured_top_k>0 / enable_entity_swap   # 기존 필드
(first_found · *_only 는 제안 대상에서 제외)
```

**중복 표현을 없애는 것이 핵심 이유 하나다.** `keyword_only`와 "깊이를 전부 0으로 둔 `rrf`"가
같은 상태를 가리키면 LLM이 같은 수를 두 경로로 제안하고, `attempt_history`가 그것을
**같은 수로 인식하지 못해** 거부 기록이 우회된다.

**`*_only`는 이 분해와 별개의 일이다 (2026-11-16 정정).** 한동안 "`*_only`를 지우려면
−0.010을 먼저 설명해야 한다"고 적었는데, **둘은 다른 질문이고 소관 축도 다르다.** 셋으로
갈라 적는다.

| | 질문 | 소관 | 판정 방법 |
|---|---|---|---|
| **A** | `*_only`가 `fusion_mode`의 값인가 | **view 마스크 축** | **측정 불필요.** R1로 이미 판정됐다 — `*_only`는 `retrieve_multiview`에서 **어느 view를 실행할지**를 게이팅하고, 그것은 합치는 방법이 아니다. 설계 결정이다 |
| **B** | `keyword_only`와 "보조 깊이 0 + fusion" 이 **같은 랭킹**인가 | **표현의 중복**(R6) | **점수 측정이 아니라 run 파일 동일성 비교.** 두 설정의 doc id 리스트가 같은지만 보면 된다 |
| **C** | view가 하나일 때 rrf가 왜 −0.010인가 | **rrf의 거동** | 별개 측정. 그리고 **출처가 불명이다** — 이 문서에만 있고 JOURNAL에 대응 항목이 없다 |

A는 지금 결론이 났고, C는 `*_only`와 무관하다(단일 view에서 rrf가 무엇과 비교해 −0.010
이었는지조차 기록이 없다). 남는 것은 B 하나이고, 그것은 **채점이 아니라 비교**다.

B를 재는 방법도 둘을 함께 건드리지 않게 설계한다 — 마스크를 `{keyword}`로 고정한 채
`rrf`와 `weighted_sum` **양쪽**에서 `keyword_only`와 run 파일을 비교한다. 둘 다 같으면
B는 참이고 `*_only`는 중복 표현이므로 접을 수 있으며, **그때 C가 혼자 남아** rrf만의
질문으로 분리된다.

**보고 형식.** fusion의 주효과 열은 **만들지 않는다.** view 조합 8가지 × fusion 2값 =
**16셀**로만 보고한다. 8조합은 깊이 100으로 이미 측정돼 있으므로(§8) 추가 비용은 한 번의
2배다.

**`rrf_k`는 여전히 하드코딩(60)이다.** 즉 `rrf`는 **측정된 적 없는 1-파라미터 족의 한
점**이다. @30만 보고하는 동안은 "고정, 미탐색"으로 명시하고, 깊은 지표를 한 줄이라도 쓰면
차원으로 올려야 한다(위 "깊은 지표를 보고하려면" 참조).

#### `first_found` — 제거했다 (코드 반영 완료, 2026-11-16)

고치지 않고 **막았다.** 라운드로빈으로 고치면 "각 view의 상위를 섞는다"가 되는데 그것은
`rrf`가 이미 하는 일이고, 그러면 한 축에 같은 일을 하는 값이 둘이 된다 — 분해의 목적과
반대다. 네 곳을 동시에 막는다.

| 어디 | 무엇 |
|---|---|
| `legal_retriever.search()` | `fusion_mode == "first_found"`면 **`ValueError`**. `max_context` 단정과 같은 자리이고, **그보다 먼저** 검사한다 — 기본 설정은 두 조건을 모두 어기는데 더 근본적인 쪽을 보여야 한다 |
| `legal_evolve.setup()` | `diagnosis.MemoryDiagnostics.ENUM_VALUES["fusion_mode"]`에서 제거. `set_action_fields`는 **어느 필드**를 제안할 수 있는지만 정하고 **어떤 값**인지는 말하지 않는다 — 그 틈이 이 필드에서만 문제가 된다 |
| `recipe_learning.ENUMS` | 카드가 실어 나를 수 있는 값에서 제거 |
| `diagnosis_prompts.py` (en/zh/ko) | 값 목록에서 빼되 **이름은 남기고 거부 이유를 적는다** — 목록에서 조용히 사라지면 LLM이 "왜 안 되는지"를 모른 채 다시 제안한다 |

`verify.py`가 두 검증기(`ENUM_VALUES`·`ENUMS`)에 `first_found`가 없음을 **게이트에서
단정**한다. 세 곳이 서로 합의해야 하는데 아무도 검사하지 않고 있었다.

**벤더는 포크하지 않았다.** `multi_retriever._fuse`의 `first_found` 분기는 그대로 있고
(LoCoMo가 쓴다), 우리 쪽에서 **도달 불가**로 만들었다 — §2 "seam"의 원칙 그대로다.
과거 실행 기록을 확인했다: `runs/`·`evolution_*/`의 설정 460건 중 `first_found`는
**0건**이다(keyword_only 91 / rrf 233 / weighted_sum 136). 즉 이 변경은 과거 숫자를
무효화하지 않는다.

**`RetrievalConfig`의 기본값은 여전히 `first_found`다**(벤더 값). 바꾸지 않은 이유는
그것이 벤더 dataclass이기 때문이고, 그래서 **기본 생성자로 만든 설정은 이제 예외로
떨어진다** — 조용히 −0.115를 찍던 자리가 실패하는 자리가 됐다.

> **아직 남은 것**: `*_only` 세 값은 **그대로 둔다** — 위 B(run 파일 동일성)가 미착수이기
> 때문이고, −0.010과는 무관하다. 그리고 **step-cap 회계 예외**(§3 B)와 묶음/제약
> 분리(§3)는 아직 코드에 없다.

**⑤ MMR 다양성 재랭킹** — 우리는 off.

**⑥ 상위 `max_context`개를 `RetrievedMemory`로 반환.**

### 바깥 루프 — 한 라운드가 하는 일

1. **채점** — train 640쿼리를 현재 θ로 검색, NDCG@30 평균 + 클래스별 성적 + 실패 사례 수집
2. **진단** — LLM 1회. 프롬프트에 들어가는 것: 성적표, 현재 설정 전체, 차원 카탈로그,
   `{available_recipes}`(§9), `attempt_history`(과거 시도와 거부 기록).
   출력은 JSON — `parameter_suggestions` 또는 `use_recipe`
3. **step-cap** — 최대 2개 필드만 반영. 레시피는 **1칸**
4. **가드** — θ'로 다시 채점. `+0.003` 미만이면 **θ로 되돌림**(`elitist=True`)

**4번이 왜 중요한가.** 서로 의존하는 설정 5개를 켜야 이득이 나는 경우, 2개씩 나눠 가면 중간
상태가 θ₀보다 나빠 전부 거부되고 라운드마다 원위치한다. 입력이 같으니 `temperature=0`인 LLM은
같은 제안을 반복한다 — **무한 루프다.** 레시피 pool이 비어 있던 4회 실행이 전부 R1에서 멈춘
메커니즘이고, 레시피가 그 5개를 1칸으로 묶자 R0에서 바로 출발할 수 있었던 이유다(§9·§10).

### fusion 코드에서 고친 것

**(a) RRF 분기의 죽은 루프를 지웠다.** `agg` dict를 채우는 첫 루프의 결과가 한 번도 읽히지
않았고, 그 루프는 조건이 `agg[i][0] < contrib`라 **이미 누적된 값이 새 기여보다 크면 더하지
않는** 틀린 구현이기도 했다. 주석 `# collapse: sum contributions (second pass to be correct)`가
그 고백이었다. 버려지는 쪽이 틀린 쪽이었으므로 **결과는 처음부터 정확했다.** 지금은
`rrf`와 `weighted_sum`이 누적 함수 하나를 공유한다.

**(b) `source` 귀속이 틀려 있었다 — 우리에게 직접 영향이 있었다.** docstring은
*"attribution = top view"*라 적혀 있었으나 코드는 **처음 삽입한 view**를 유지했고,
`view_results`의 삽입 순서가 `semantic → keyword → structured → swap`으로 고정이라
**semantic이 켜져 있으면 BM25가 1위로 올린 문서까지 전부 `semantic`으로 표기**됐다.

우리는 `h.source`에 doc_id와 법률 메타데이터를 붙여 **진단 LLM의 실패 로그로 보낸다**
(`legal_evolve.py`). 즉 LLM이 view 귀속을 읽을 때 체계적으로 편향된 정보를 받고 있었다.

**고친 뒤 단일 라벨로는 부족하다는 것이 드러났다.** "최대 기여 view"로 바꾸자 RRF에서
`keyword 99.8%`가 나왔는데, 기여가 `w/(60+rank)`라 `weight_keyword=1.0` vs
`weight_semantic=0.3`이면 **같은 순위에서 semantic이 이길 수 없기 때문**이다. 먼저 삽입된
view는 순서 편향, 최대 기여 view는 가중치 편향이다. 그래서 **기여한 view 전부**를 기여도 순으로
적는다(`keyword+semantic+swap`). view를 진단하려면 승자가 아니라 **집합**이 필요하다.

### seam

EvolveMem의 `retrieve_multiview`를 **포크하지 않는다**. 진화 엔진이 튜닝할 fusion·시간감쇠·
per-category override 로직이 그 안에 있기 때문이다. 법률에서 깨지는 지점만 seam에서 교체한다.

| seam | 원본의 전제 | 교체 |
|---|---|---|
| BM25 토크나이저 | `content.lower().split()` | jieba / mecab / spaCy — 중국어는 문서 전체가 1토큰이 됐다 |
| 쿼리 분석 | 대문자로 엔티티 탐지 (영어 전용) | 사전 추출한 쿼리 메타데이터 주입 |
| pool 마스킹 | 없음 | `index.set_pool()` 하나로 세 view 전부 — **기본 off**, LeCaRD 공식 프로토콜 재현용 opt-in (§5) |
| 구조화 view | persons ∩ entities + 상수 1.0 점수 | 일치 **개수** 점수, persons 기본 비활성 (아래) |
| `RetrievedMemory` | 식별자 없음 | 레포에 `doc_id` 필드 추가 (2줄, LoCoMo는 `None`) |

### `persons` / `entities`가 무엇인가

두 필드는 EvolveMem이 메모리마다 갖는 메타데이터 슬롯이고, **우리가 `legal_meta.py`의
규칙 기반 추출로 채운다**. 문서와 쿼리를 **같은 추출기**로 처리하므로 양쪽이 같은 어휘를 쓴다.

| 필드 | 내용 | 예시 |
|---|---|---|
| `entities` | **법조문 인용 + 죄명 + 정규화 토큰** | `형법 제347조` / `中华人民共和国刑法第一百三十三条` / `s.2(1) Immigration Act` / `交通肇事罪` / `art:133` |
| `persons` | **사건 당사자 이름** | `张宝玉` / `Campbell` / `A` |

`art:NNN`은 우리가 추가한 정규화 토큰이다. 같은 조문이 문서에서는 한자(`第二百六十三条`),
쿼리에서는 아라비아 숫자로 나타나 매칭률이 0%였기 때문이다(§4).

채움률(문서 / 쿼리):

| 데이터셋 | entities | persons |
|---|---|---|
| LeCaRDv2 | 99.9% / 100% | 99.8% / 95.0% |
| COLIEE | 87.3% / 100% | 89.7% / **19.6%** |
| LEGAR | 100% / 100% | 61.6% / 73.6% |

### 구조화 view — 무엇을 어떻게 바꿨나

원본 `structured_search`(`multi_retriever.py:367`)는 **persons와 entities의 교집합**을 구하고
걸린 문서에 **상수 1.0**을 준다. 대화 메모리에서는 맞다 — 화자 이름이 반복 등장하고,
"그 사람이 말한 그 주제"가 곧 검색 조건이다.

판례에서는 두 군데가 동시에 깨진다.

1. **교집합이 결과를 비운다.** 당사자 이름은 사건마다 다르다. 쿼리 사건의 피고인이 다른
   판례에 등장할 일이 없으므로 persons 집합과 entities 집합의 교집합은 거의 항상 공집합이다.
2. **상수 1.0은 fusion에 줄 정보가 없다.** 조문 하나가 겹친 문서와 셋이 겹친 문서가 같은 점수를
   받는다. 가중합에서 이것이 −9.0 pp의 직접 원인이다(§8) — 수천 건이 전부 같은 값을 받아
   BM25 순위를 평탄하게 만든다.

→ **교집합을 합집합으로, 상수를 일치 개수로** 바꿨다(`legal_retriever.py:226`).
entities 일치 1건당 1.0, location 일치 0.25, persons 일치 0.25이며 **persons는 기본 비활성**이다.

### persons를 매칭에 쓰지 않는 이유 — 데이터셋마다 다르다

"비활성"은 한 줄짜리 설정이지만 근거는 세 가지로 서로 다르다.

**LeCaRDv2 — 너무 고유해서 쓸모없다.** 고유 인명 135,009개 중 **88%가 코퍼스 전체에서 단 한 번**
등장한다. 두 사건이 같은 피고인을 공유하는 일이 사실상 없으므로, 매칭해도 자기 자신 말고는
아무것도 데려오지 않는다.

**LEGAR — 익명화되어 변별력이 없다.** 한국 판결문은 당사자를 `A`·`B`·`C`로 익명화한다.
`A`는 **문서의 9.5%(1,384건)**에 등장한다. `A`로 매칭하면 서로 무관한 1,384건이 딸려 온다.

**COLIEE — 쿼리에 없다.** 쿼리의 **19.6%**만 인명이 추출된다. 나머지 80%에서는 view 자체가 비어 있다.

여기에 공통으로 **추출 노이즈**가 얹힌다. 정규식이 조사·연결어를 인명으로 잡는다 —
LEGAR의 `에게`(6,374건)·`들은`(772건), LeCaRDv2의 `的行为`(3,627건)·`的犯罪`(2,458건)이
상위 "인명"이다. 이것들로 매칭하면 사실상 무작위 문서를 가져온다.

### 같은 `persons`가 metadata view에서는 꺼져 있고 party-strip에서는 쓰인다

쓰는 방향이 반대라서 **오탐의 대가**가 다르다.

| | metadata view | party-strip view |
|---|---|---|
| persons로 하는 일 | **문서를 찾는다** | 쿼리에서 **지운다** |
| 오탐 하나의 대가 | 무관한 문서를 상위에 올림 | 쿼리에서 글자 몇 개가 더 지워짐 |
| `에게`가 섞이면 | 6,374건이 딸려 옴 | 조사 하나가 지워짐 — BM25에서 어차피 불용어 |

여기까지는 맞다. **그런데 이 표는 오탐만 가격을 매긴다.** 더 중요한 질문이 빠져 있었다 —
**이름이 쿼리에 남아 있는 것 자체의 대가는 얼마인가.** `A사`·`B씨`가 BM25 키워드로
걸리면 그 이름을 공유하는 무관한 사건들과 점수를 나눠 갖는다. 그것이 미탐(추출되지
않은 이름)의 대가이고, 위 표에는 그 칸이 없다.

#### party-strip은 그 대가를 치우지 않는다 — 네 번째 후보 목록을 **더할** 뿐이다

주 BM25 view가 받는 것은 `analysis["keywords"]`이고, 우리 어댑터는 그것을
`[query]` — **이름이 포함된 원문 전체** — 로 돌려준다(`legal_retriever.py:147`).
party-strip은 이름을 지운 쿼리로 **별도 검색**을 한 뒤 fusion에 합친다
(`multi_retriever.py:751`). 즉 **주 ranking의 이름 오염은 그대로 남고**, 그 옆에
이름 없는 후보 목록이 하나 추가된다. "이름을 지운다"는 서술이 **주 view에도
적용되는 것처럼** 읽히지만 아니다.

#### 그래서 쿼리 자체에서 이름을 지우면 어떻게 되나 — 쟀다 (2026-11-12)

한 번도 측정한 적이 없는 개입이다. §16이 잰 것은 party-strip **view의** 삭제
목록을 넓히는 것이었고, 주 쿼리는 건드리지 않았다. `row["text"]`에서 이름을 지워
**모든 view가 이름 없는 쿼리를 보게** 하고 θ₀(BM25 단독)에서 쌍체로 쟀다.

삭제 범위를 둘로 나눠 쟀다 — 당사자명만, 그리고 **당사자명 + 조직 + 지명 + 법원**.

| 데이터셋 | train n | θ₀ | 당사자명만 | **+ 조직·지명·법원** | 넓은 목록 적용률 |
|---|---|---|---|---|---|
| LeCaRD | 79 | 0.6637 | **−0.0457** [−0.0724,−0.0236] 악화 | **−0.0451** [−0.0723,−0.0194] 악화 | 78/79 |
| LeCaRDv2 | 576 | 0.3885 | −0.0032 [−0.0075,+0.0011] 구분 불가 | **−0.0170** [−0.0229,−0.0110] **악화** | 571/576 |
| COLIEE | 1,148 | 0.2413 | −0.0003 [−0.0011,+0.0005] 구분 불가 | +0.0007 [−0.0018,+0.0032] 구분 불가 | 1,022/1,148 |
| LEGAR_stricter | 4,228 | 0.0516 | +0.0002 [−0.0003,+0.0008] 구분 불가 | +0.0006 [−0.0002,+0.0013] 구분 불가 | 3,741/4,228 |

**어디서도 이름을 지우는 것이 도움이 되지 않는다.** 가장 좋은 경우가 "구분 불가"다.

읽을 것 셋.

**1. LeCaRD에서는 이름이 도움이 되고 있었다.** Δ가 그 fold의 noise floor(0.0240)를
넘으므로 잡음이 아니다. 지명·조직을 더 지워도 달라지지 않는다(−0.0457 → −0.0451) —
대가를 치르게 하는 것은 당사자명이고 지명·조직은 이 데이터셋의 BM25 점수에 거의
기여하지 않는다.

**2. 지명이 단독 최대 원인이고, 과삭제 때문이 아니다.** 넓은 목록을 네 성분으로
쪼개 각각 재면(`strip_components.py`, LeCaRDv2 train):

| 삭제 대상 | Δ | CI95 | 판정 |
|---|---|---|---|
| parties | −0.0032 | [−0.0075,+0.0011] | 구분 불가 |
| parties + court | −0.0035 | [−0.0077,+0.0008] | 구분 불가 |
| parties + org | −0.0060 | [−0.0106,−0.0015] | 악화 |
| **parties + place** | **−0.0136** | [−0.0191,−0.0078] | **악화** |
| 넷 다 | −0.0171 | [−0.0230,−0.0111] | 악화 |

처음에는 **중국어 정규식의 과삭제**를 원인으로 봤다. 실제로 결함이 둘 있었고
고쳤다(§legal_meta): 탐욕 수량자가 `严坪村村民`을 `严坪村` + 접미사 `村`으로 파싱해
`严坪村村`을 지우고 `民严某2家`를 남겼고, 지시어 검사가 **첫 글자만** 봐서
`要求全县`·`安排全县`(동사구)과 `各乡镇`("각 읍면", 일반명사)이 통과하고 있었다.

**그런데 고친 뒤 숫자가 그대로다** — −0.0134 → −0.0136. 남은 span은 전부 진짜
지명이다(`涡阳县`·`邢台市`·`平乡县`·`寻召村`·`大莫古镇`·`广州市`·`番禺区`). 결함은
실재했지만 **손실의 원인이 아니었다.**

원인은 데이터 쪽이다. 정답 문서가 쿼리의 지명을 공유하는 비율을 재면:

| | 정답과 공유 | 무작위와 공유 | 차이 |
|---|---|---|---|
| LeCaRD | 57% | 20% | **+37 pp** |
| LeCaRDv2 | 62% | 11% | **+51 pp** |

당사자명의 +16/+17 pp보다 **훨씬 강하다.** 즉 이 두 벤치마크에서 **지명은 관련성과
강하게 상관한다.**

**개념적으로는 지명을 지우는 쪽이 옳다** — 같은 시·군에서 재판했다는 사실이 두
사건을 유사하게 만들지 않는다. 그러나 이 폴드에서는 지우면 점수가 내려가고, 그 폭이
noise floor(0.0050)의 2.7배다. **이것은 과제 정의와 벤치마크 라벨이 어긋나는
지점이고, 우리가 고를 수 있는 것은 어느 쪽을 보고하느냐뿐이다.** 판례 수집이
관할별로 이뤄지고 LeCaRD 코퍼스가 쿼리별 후보 pool의 합집합이라는 구성이 그
상관의 유력한 출처다 — 확인하지는 않았다.

**3. 영어·한국어는 둘 다 null이다.** LEGAR는 당사자가 `A`·`B`·`C`로 익명화돼 있어
**이름 자체에 어휘적 변별력이 없다** — 지워도 아무 일이 일어나지 않는다. `A사`·`B씨`
형태가 검색을 방해한다는 걱정은 이 데이터셋에서는 성립하지 않는다. 방해도 하지
않기 때문이다. COLIEE도 같다.

그리고 COLIEE가 **적용률만으로는 설명되지 않는다**는 점이 LeCaRDv2와의 대비를
만든다. 넓은 목록이 COLIEE 쿼리의 **89%**(1,022/1,148)에 적용되는데도 null이고,
LeCaRDv2는 **99%**(571/576)에서 −0.0170이다. 즉 "많이 지워서 나빠진 것"이 아니라
**중국어 지명·조직 삭제가 지워서는 안 되는 글자를 가져간다**는 쪽이다 — §16이 적어 둔
중국어의 경계 문제(띄어쓰기가 없어 `来到重庆市`에서 `到重庆市`를 물어 온다)가 그
후보다.

왜 이름이 도움이 되는지 직접 셌다 — 정답 문서가 쿼리의 당사자명을 포함하는 비율
(상위 20건 기준, 같은 수의 무작위 문서를 대조군으로):

| | 정답 문서와 공유 | 무작위 문서와 공유 | 차이 |
|---|---|---|---|
| LeCaRD | 58% | 42% | **+16 pp** |
| LeCaRDv2 | 53% | 36% | **+17 pp** |

대조군 수치가 높은 것은 중국어 인명이 짧아 부분 문자열로 우연히 걸리기 때문이고
(이 측정은 부분 문자열 포함 검사다), 그래도 **차이는 양쪽에서 일관된다.**
LeCaRD 코퍼스는 107개 쿼리의 공식 후보 pool을 합친 것이므로 정답 문서가 쿼리와
당사자를 공유하는 일이 실제로 있고, 이름이 그 사건을 **특정하는 데** 쓰이고 있었다.
유사 판례 검색의 목적에서 보면 그것은 **원하는 신호가 아니라 벤치마크의 성질**이다 —
그러나 측정은 측정이고, 이 fold에서 이름을 지우면 점수는 내려간다.

(당사자명만 지우는 쪽의 적용률은 LeCaRD 76/79 · LeCaRDv2 545/576 ·
LEGAR 3,090/4,228 · **COLIEE 198/1,148**이다. COLIEE의 17%는 §8의 인명 추출률
19.6%와 일치한다.)

#### 정리

- **오탐에 관대하다**는 것은 party-strip view에 대해 맞다.
- **이름 오염을 치운다**는 것은 party-strip view가 하는 일이 **아니다** — 주 BM25는
  이름이 든 원문을 계속 읽는다.
- 그 오염을 실제로 치우는 개입(주 쿼리에서 삭제)은 **네 데이터셋 중 둘에서 해롭고
  둘에서 무효다. 어디서도 양수가 아니다.** 이름이 검색을 방해한다는 가설은 재 보니
  성립하지 않았다.
- **지명이 단독 최대 원인**이고(LeCaRDv2 −0.0136), **정규식 결함 때문이 아니다** —
  결함 둘을 고쳐도 숫자가 그대로다. 원인은 정답 문서가 쿼리의 지명을 공유한다는
  것이다(+51 pp vs 무작위). **개념적으로 지워야 할 신호를 벤치마크가 보상한다.**

party-strip이 새 코퍼스에서 죽은 것(§8: ⑥ RRF+swap30이 +0.0001 / **−0.0347** /
−0.0005 / +0.0004)과도 일관된다. 이름 없는 후보 목록을 **더하는** 것으로는 주 view의
오염을 되돌릴 수 없고, 되돌릴 필요가 있는지 자체가 데이터셋마다 다르다.

다만 party-strip도 persons **추출률**에 의존한다. COLIEE 쿼리의 19.6%에서만 발동하므로
그 데이터셋에서는 쿼리 재작성이 같은 역할의 대안이 된다(§8).

---

## 3. action space (45차원 → 제안 가능한 것은 12개)

> `RetrievalConfig`는 45필드이지만, 이 파이프라인에서 **순위를 실제로 바꾸는 것은
> 12개**다. 나머지 33개는 2026-11-10에 제안 대상에서 제외했다(JOURNAL §26). 그리고
> rewrite 캐시가 없는 데이터셋에서는 `enable_query_rewrite`도 빠져 **11개**가 된다
> (JOURNAL §37). 아래 "45차원"은 설정 구조의 크기이고, LLM 앞에 놓이는 것은 그중
> 12(또는 11)개다.


### action space ≠ 레시피 — 층이 다르다

혼동하기 쉬운 지점이라 먼저 못박는다. **레시피는 action space가 아니다.**

그리고 이 둘 **위에 한 층이 더 있다** — view가 무엇인지를 정하는 결정들이고, LLM은
그것을 볼 수도 바꿀 수도 없다(**§0.3**). `enable_entity_swap`(② 설정, LLM이 켠다)과
`STRIP_SCOPE`(① 정의, 그 view가 **무엇을 지우는지**)의 차이가 그 경계다.

| | action space | 레시피 |
|---|---|---|
| 정체 | `RetrievalConfig`의 **필드 45개** | 그 필드 **몇 개를 묶어 둔 꾸러미** |
| 비유 | 조작 가능한 **필드 전체** | "이 필드들을 이 값으로" 적힌 **카드** |
| 누가 만들었나 | 레포 (+우리가 3개 추가) | **우리가 측정 결과로 작성** (§9) |
| step-cap 계산 | 1필드 = **1칸** | 몇 필드를 바꾸든 **1칸** |

> **이 표는 현재 코드의 상태이고, 2026-11-16에 폐기하기로 했다.** 오른쪽 열의
> *"이 필드들을 **이 값으로**"* 가 혼동의 근원이다 — 레시피가 값을 담으면 타입이
> 왼쪽 열과 같아지고, 그러면 두 층이 아니라 한 층의 두 입력 형식이다. 새 설계는
> 아래 **"레시피와 action space의 분리"**에 있다. 그 전까지의 서술(아래 두 소절)은
> 현재 코드를 설명하므로 그대로 둔다.

레시피는 action space **위에서 움직이는 단축키**다. 레시피가 바꾸는 것도 결국 저 45개 필드다.

```
라운드당 예산 = 2칸 (max_changes_per_round)

  A:  fusion_mode 변경(1칸) + weight_semantic 변경(1칸)      = 2칸
  B:  use_recipe: lexical_first_light_semantic              = 1칸
      └ 실제로는 5개 필드가 한꺼번에 바뀐다
      + 추가 필드 1개                                        = 총 2칸
```

#### 한 장이 실제로 건드리는 필드

당시 `lexical_first_light_semantic`은 45개 중 **5개**를 한꺼번에 설정했다.

```
fusion_mode      = "weighted_sum"     # ← 당시 (2026-11-11 현재는 "rrf")
semantic_top_k   = 100
keyword_top_k    = 100                # ← 제거됨 (θ₀가 이미 100)
weight_keyword   = 1.0                # ← 제거됨 (θ₀가 이미 1.0)
weight_semantic  = 0.3                # ← 제거됨 (크기는 LLM이 고른다)
```

**현재는 2개다** — `fusion_mode="rrf"`, `semantic_top_k=100`. 셋을 뺀 이유가 다르다:
`keyword_top_k`·`weight_keyword`는 θ₀가 이미 그 값이라 아무것도 바꾸지 않았고,
`weight_semantic`은 **크기이므로 LLM이 고를 몫**이다(§9 머리). 다만 그 셋째가
아래 논증을 약화시키지는 않는다 — 카드가 `weight_semantic`을 **설정하지 않는
대신 "같은 라운드에 반드시 네가 정하라"고 요구**하기 때문이다. 의존 관계는 그대로
남아 있고, 누가 값을 고르는지만 바뀌었다.

#### 왜 필드를 직접 돌리면 도달하지 못하는가

**이 차이가 R1 정체를 풀었다.** 위 5개를 필드 단위로 바꾸면 2칸씩이므로 **3라운드**가 걸린다.
그런데 이 5개는 **서로 의존한다** — 예컨대 `semantic_top_k=100`만 켜고 `weight_semantic`을
기본값에 두면 가중치가 과해 **θ₀보다 나빠진다.** elitist는 나빠진 라운드를 되돌리므로
(`delta > acceptance_threshold`가 아니면 revert) **중간 상태가 전부 거부되어 목적지에
도달할 수 없다.** 레시피는 그 5개를 **1칸**으로 넘겨 중간 상태를 건너뛴다
(§2 "바깥 루프", §9).

따라서 정확히 말하면: 레시피 없이 45개 필드만으로도 같은 지점에 **도달은 가능**하다.
다만 의존하는 필드 bundle은 step-cap과 elitist가 겹쳐 **실질적으로 도달 불가**다.
레시피는 표현력을 넓히는 것이 아니라 **경로를 뚫는 것**이다.

#### 한 줄 요약

> **45개는 어휘이고, 레시피는 그 어휘로 쓴 검증된 문장이다.**
>
> 단 우리 과제에서는 그 문장이 거의 없다 — 손으로 쓴 5장 중 4장이 **단어 하나**이고,
> 그래서 레시피를 LLM이 제안하게 한다(§9).

LLM은 둘 다 쓸 수 있다 — 필드를 직접 제안하거나(각 1칸), `use_recipe`로 장을 고르거나(1칸).

### 레버 등록부와 등록 규칙 (2026-11-16)

2026-11-16에 찾은 문제 다섯 — `fusion_mode`가 축이 아니었고, `first_found`가 벤더
기본값으로 숨어 있었고, 레시피와 action space의 타입이 같았고, rewrite와 party-strip이
**같은 연산**이었고, fusion의 Δ가 단면에서 나왔다 — 은 서로 다른 버그가 아니라 **레버를
등록하는 규칙이 없다**는 한 가지 결함의 다섯 증상이다. 규칙을 먼저 적고, 그 규칙으로
현재 레버를 다시 등록한다.

#### 레버의 좌표 — 등록부는 코드다 (`registry.py`)

모든 검색 레버를 세 항목으로 등록하되, **동일성은 (입력 × 채점)** 으로 판정한다.

| 항목 | 값 | 동일성에 들어가나 |
|---|---|---|
| **입력** | 질의를 무엇으로 바꿔 넣는가 — `original` / `strip` / `expand` | **예** |
| **채점** | 무엇으로 점수를 내는가 — `bm25` / `embedding` / `metadata` | **예** |
| **결합** | 결과를 어떻게 합치는가 — `view`(별도 후보 → fusion) / `in-place`(질의 텍스트에 합침) | **아니오 — 기록만** |

**`결합`을 동일성에서 뺀 것이 핵심이다.** 같은 입력을 같은 채점 함수에 먹이는 두 레버는
**같은 정보**를 나르고, 별도 리스트로 융합하느냐 질의에 섞느냐는 그 정보가 **어떻게
들어오는가**일 뿐이다 — R2의 *"구현 방법은 좌표가 아니다"* 가 바로 그것을 말한다.
처음에 셋 전부를 동일성에 넣었더니 party-strip `(strip, bm25, view)` 과
rewrite `(strip, bm25, in-place)` 가 **서로 다른 레버로 통과했다** — 규칙이 잡아야 할
바로 그 쌍이다.

같은 이유로 `INPUTS`에 **`strip+compress` 같은 값은 두지 않는다.** rewrite가 삭제에
더해 하는 압축을 좌표로 올려 주면 레버가 **자기 부수 효과를 이름으로 삼아 R2를
빠져나간다.** 압축만 따로 측정되기 전까지 rewrite의 입력은 `strip`이고, 게이트는 매번
"알려진 중복"으로 출력한다.

이 좌표계가 §2의 *"세 개의 채점 함수 + 질의 변환 하나"* 를 대체한다. view는 축이 아니라
**(입력 × 채점) 격자의 한 칸**이다.

> **표가 아니라 코드다.** 등록부는 `registry.py`의 `LEVERS` 상수이고 `verify.py`의
> 세 번째 검사가 그것을 건다. 문서의 아래 표는 그 코드의 사본이다 — 어긋나면 코드가 맞다.
>
> **2026-11-22: `levers.py`와 `recipes.py`를 `registry.py` 하나로 합쳤다.** 두 쪽이
> 불변식 하나를 공유하기 때문이다 — Rule이 자기가 만드는 레버 좌표를 선언하고, 그
> 좌표는 등록부에 있어야 한다. 파일이 둘이면 그것은 **규약**이고, 하나면 **조회**다.

#### 일곱 규칙

**R1 — 축의 동질성.** 한 필드의 값들은 **교환 가능**해야 한다. 값을 바꾸는 것이 *다른
질문을 하는 것*이 되면 그 필드는 축이 아니라 **선택자**이고, 분해한다.
→ 위반: `fusion_mode`(view 마스크 + 캐스케이드 + 진짜 fusion이 한 enum에). §2에서 분해.

**R2 — 좌표의 유일성.** 같은 좌표에 레버가 둘이면 하나를 접는다. 구현 방법(정규식 vs LLM)은
좌표가 아니다.
→ 위반: party-strip `(이름삭제, BM25, view)` 과 rewrite `(이름삭제+압축, BM25, in-place)`.
`결합`만 다르고 **입력의 주된 연산이 같다** — 측정이 이미 말했다(§5: corr +0.524,
Jaccard 0.463). 해소는 두 길뿐이다: 하나를 없애거나, **입력 좌표를 실제로 갈라
놓는다**(party-strip = 삭제, rewrite = 확장). 지금 rewrite 프롬프트의 1번 줄("이름·지명·
날짜·금액 제거")이 좌표를 겹치게 만드는 문장이다.

**R3 — 기본값 위에서 판정하지 않는다.** 비교하는 모든 축은 **명시적으로 설정한 값**에서
재고, 비교에 들어가는 모든 다른 축도 그렇다. 측정하지 않은 기본값에서 나온 판정은
"그 기본값에 대한 판정"이지 레버에 대한 판정이 아니다.
→ 위반 넷(전부 같은 원인): metadata view "해롭다"(깊이 10 = 엔진 기본),
party-strip "죽었다"(깊이 30 = 옛 코퍼스 잔존), `weight_structured` 0.5(엔진 기본),
`first_found`(`RetrievalConfig` 기본). JOURNAL §47가 앞의 셋을 뒤집었다.

**R4 — Δ에는 조건을 붙인다.** 상호작용하는 축을 고정한 채 **단독 주효과를 보고하지
않는다.** 모든 Δ는 〈활성 view 조합 · 깊이 · fold · 데이터셋〉과 함께 적는다. 표에
주효과 열을 만들면 읽힌다.
→ 위반: fusion "실질적으로 무관"(view 조합 고정 단면). §5가 *"sweep 설계가 결론을
만든다"* 로 이미 적어 둔 함정을 결론 표가 반복했다.

**R5 — 층이 다르면 타입이 다르다.** "다른 층"이라는 주장은 **타입 수준에서** 확인
가능해야 한다. 같은 타입을 내놓고 같은 예산을 쓰면 한 층이다.
→ 위반: `Recipe.proposal`과 `parameter_suggestions`가 둘 다 `dict[field, value]`.
해소는 이 절 다음의 "레시피와 action space의 분리".

**R6 — 중복 표현 금지.** 같은 상태에 도달하는 경로가 둘이면 하나를 막는다. 경로가 둘이면
`attempt_history`가 같은 수를 **같은 수로 인식하지 못해** 거부 기록이 우회된다.
→ 위반 후보: `keyword_only` 와 "보조 깊이를 전부 0으로 둔" 마스크. **동치인지 미확인**.
확인은 **채점이 아니라 run 파일 비교**이고(§2의 B), fusion 축이나 §2의 −0.010과는
무관한 별개의 일이다 — 확인 전에는 둘 다 남긴다.

**R7 — 쓰지 않는 값은 도달 불가로.** 레버나 값을 접기로 했으면 **네 곳을 동시에** 막는다:
① 실행 초크포인트(`legal_retriever.search`)에서 예외, ② 제안 검증기
(`diagnosis ENUM_VALUES`), ③ 카드 검증기(`recipe_learning.ENUMS`), ④ 프롬프트 —
**이름은 남기고 거부 이유를 적는다**(조용히 사라지면 LLM이 왜 안 되는지 모른 채 다시
제안한다). 그리고 `verify.py`가 ②③의 합의를 게이트에서 단정한다.
→ 적용 완료: `first_found`(§2).

#### 현재 등록부

| 레버 | 입력 | 채점 | 결합 | 필드 | 상태 |
|---|---|---|---|---|---|
| keyword | 원문 | BM25 | view | `keyword_top_k`, `weight_keyword` | θ₀. 항상 켬. **sweep 미실시**(JOURNAL §25) |
| semantic | 원문 | 임베딩 | view | `semantic_top_k`, `weight_semantic` | 유일하게 일관된 양수 레버 |
| metadata | 원문 | 메타데이터 | view | `structured_top_k`, `weight_structured` | 보완재. `weight_structured` sweep 중 |
| party-strip | 이름삭제 | BM25 | view | `enable_entity_swap`, `entity_swap_keyword_k` | 깊이 100에서 중립. **가중치는 `weight_semantic`과 공유**(아래) |
| rewrite | 이름삭제+압축 | BM25 | **in-place** | `enable_query_rewrite`, `query_rewrite_repeat` | **R2 위반** — party-strip과 입력 좌표가 겹친다. LeCaRDv2에만 캐시 |
| — | 이름삭제 | 임베딩 | view | `entity_swap_semantic_k` | **은퇴**(0 고정) — 주 semantic과 중복(§44) |
| — | 확장 | BM25 | ? | 없음 | **빈칸.** 법률 요소(죄명·조문·유의어)를 질의에 **더하는** 레버가 없다 |

**빈칸이 지금 가장 큰 발견이다.** 네 레버 중 질의를 바꾸는 둘이 **모두 삭제**이고,
확장이 하나도 없다. BM25는 문서에 있는 용어만 매칭하므로 **삭제만으로는 재현율이
구조적으로 오르지 않는다** — rewrite가 +0.0009 CI∋0인 것은 레버가 약해서가 아니라
삭제형 재작성을 원문에 덧붙이는 것이 **tf 재가중 이상을 할 수 없기** 때문이다. 비교 대상인
"When Rules Learn"의 규칙은 전부 확장이다(유의어·법률용어 변형).

법률 요소 자체는 이미 시스템 안에 있다 — `build_cases.build_queries`가 쿼리의 `Crime`을
읽어 `legal_refs`에 넣고, **metadata view가 그것으로 점수를 낸다.** 즉 법률 요소는
*채점 함수*로만 쓰이고 *질의 용어*로는 쓰이지 않는다. 그 칸을 채울 때의 유보 둘:

- **중복 계상.** 같은 `legal_refs`를 BM25 질의에도 넣으면 metadata view와 같은 신호를 두
  번 센다 — R4가 말하는 상호작용이 아니라 R2가 말하는 좌표 충돌에 가깝다.
- **누출.** `Crime`은 **어노테이션**이지 쿼리 텍스트에서 추출한 것이 아니다. gold 관련성이
  같은 죄명·조문으로 정의돼 있다면 그것을 질의에 넣는 것은 누출에 가깝고, **metadata view가
  이미 그 상태**라는 사실이 문서 어디에도 적혀 있지 않았다. 확장 레버는 **쿼리 텍스트에서만**
  뽑은 용어로 만들어야 누출이 없다.

#### 등록부가 드러낸 구현 사실 둘 (코드 확인, 문서에 없었다)

- **party-strip에는 독립 가중치가 없다.** `multi_retriever.retrieve_multiview`가
  `"swap": cfg.weight_semantic`로 묶는다 — `weight_semantic`을 올리면 swap도 같이
  올라간다. §3 B의 "가중치 4개"는 **실질 3개**이고, party-strip의 깊이별 측정
  (JOURNAL §47)은 전부 이 묶임 아래에서 나온 값이다.
- **party-strip view 내부가 `sw_sem + sw_kw` 단순 연결이다.** 정렬 없이 이어붙이므로
  RRF의 `enumerate` rank가 **BM25 절반에 대해 `entity_swap_semantic_k`만큼 밀린 채**
  시작한다. 지금은 그 값이 0이라 영향이 없지만, 켜는 순간 어휘 절반이 조용히 순위
  페널티를 받는다. §44가 "중복 기여" 이유로 닫아 둔 것이 결과적으로 이 함정도 막고 있다.

#### 게이트 (R2·R7 반영 완료, R3·R4 미착수)

규칙은 지켜지지 않으면 규칙이 아니다. `verify.py`가 이제 셋을 센다 — `--only levers`로
단독 실행되고 데이터셋에 무관하다(등록부는 tier ① 기술이라 벤치마크마다 같다).

| 검사 | 상태 | 무엇 |
|---|---|---|
| **R2** | ✅ | 활성 레버의 `(입력, 채점)`에 중복이 없다. **waiver 없는 충돌은 실패**, waiver가 있으면 "알려진 중복"으로 매 실행 출력 |
| 등록 누락 | ✅ | 제안 가능한 12필드가 **정확히 한 번씩** 레버에 귀속되거나 `NON_LEVER_FIELDS`에 있다. 등록되지 않은 필드는 실패 |
| 등록 잔존 | ✅ | 등록됐는데 제안 불가한 필드(=철 지난 등록부 항목)는 실패 |
| 소유 중복 | ✅ | 한 필드를 두 레버가 **소유**하면 실패. 남의 필드를 읽는 것은 `shares`로 선언한다 |
| 드리프트 | ✅ | 등록된 필드가 `RetrievalConfig`에 실제로 있다 |
| **R7** | ✅ | 접기로 한 값이 두 검증기에서 빠져 있다(`first_found`) |
| **R3·R4** | ⬜ | 보고되는 Δ의 조건 블록(조합·깊이·fold)은 아직 기계 검사가 없다 |

**waiver라는 장치.** R2 충돌을 무조건 실패로 두면 *"이 레버가 무엇이어야 하는가"* 라는
연구 질문이 풀릴 때까지 모든 실행이 막히고, 그러면 게이트가 꺼진다. 그래서 충돌 쌍 중
한쪽에 `coord_waiver`(사유 문자열)가 있으면 **실패 대신 매 실행 출력**이다. 지금 출력되는
줄은 하나다.

```
registry   levers=7 active=5 fields=9+3 non-lever = 12 of 12 proposable
  R2 waived (known duplicate): party-strip ~ rewrite ('strip', 'bm25')
  vacant cell: legal-element expansion (expand, bm25, view)
```

**빈칸도 매번 출력된다.** 채워지지 않은 격자 칸은 누락이 아니라 **발견**이고(질의에
법률 용어를 더하는 레버가 없다), 그래서 조용히 두지 않는다.

### 레시피와 action space의 분리 — 세 타입 (2026-11-16)

앞의 "층이 다르다" 표는 **의도**를 적은 것이고, 지금 코드에서는 그 층이 **타입 수준에서
분리돼 있지 않다.**

```python
# evolution_cookbook.py
@dataclass
class Recipe:
    proposal: dict = field(default_factory=dict)   # ← 부분 RetrievalConfig
```

진단 LLM이 내는 `parameter_suggestions`도 `dict[field, value]`다. **두 채널이 같은 타입을
내놓고, 같은 예산(칸)을 쓰고, 같은 프롬프트 메뉴에 나란히 선다.** 그러면 그것은 두 층이
아니라 **한 층의 두 입력 형식**이다.

그리고 `Recipe` 하나가 **서로 다른 세 가지 일**을 담으려 하고 있다 — 값을 정하는 일,
어느 필드가 함께 움직이는지 정하는 일, 그리고 **질의를 어떻게 다시 쓸지 정하는 일**.
셋째는 `RetrievalConfig`에 표현할 자리가 아예 없어서 지금은 `query_rewrite_repeat` 같은
정수 하나로 눌려 있다. 세 타입으로 가른다.

#### 세 타입

| | ② action space | ③ Bundle | ③′ Rule |
|---|---|---|---|
| 무엇인가 | **노드** — 어떤 필드를 **어떤 값**으로 | **간선** — 어떤 필드가 **함께** 움직이는가 | **텍스트** — 질의를 어떻게 다시 쓰는가 |
| 타입 | `dict[field, value]` | `frozenset[field]` + 술어 | 자연어 규칙 + 포맷 |
| 값을 담나 | 담는다 | **담지 않는다** | 해당 없음(필드가 없다) |
| 무엇을 바꾸나 | 설정 | 한 라운드에 함께 바꿀 수 있는 범위 | **레버의 입력 좌표**(§3 등록부) |
| 예산 | 칸을 **쓴다** | 칸을 쓰지 않고 **cap을 연다** | 칸 밖 — **바깥 축**(아래) |
| 누가 고치나 | LLM (라운드당 2칸) | 사람·LLM (루프 밖에서 승인) | 사람·LLM (루프 밖에서 승인) |
| 승인 기준 | elitist + 수용 임계값 | **2×2 상호작용 항 ≠ 0** | **효과 자체**(CI 하한 > 0 · floor 초과 · 종점 > θ₀) |

세 타입이 서로 겹치지 않으므로 **혼동할 수 없다.** 그리고 ③과 ③′는 **승인 경로가 다르다** —
Bundle은 필드가 둘 이상이라 상호작용을 정의할 수 있지만, Rule은 필드가 없어 상호작용을
정의할 수 없다. 지금 `admit()` 하나가 둘을 같은 기준으로 재려 한 것이 설계 오류다.

#### ② ↔ ③ : 레시피는 값을 담지 않는다

```python
@dataclass(frozen=True)
class Bundle:                    # ③ — 간선. 값이 없다
    name: str
    fields: frozenset[str]       # 함께 움직여야 하는 필드, 반드시 2개 이상
    trigger: Callable[[dict, dict, list], bool]
    rationale: str               # 왜 함께인가 + 근거 측정 참조

@dataclass(frozen=True)
class Constraint:                # ③ — 금지 상태
    name: str
    forbids: Callable[[dict], bool]
    repair_hint: str             # 어느 필드를 함께 움직이면 빠져나오나
```

`proposal`과 `expected_lift_pp`가 **둘 다 사라진다.** 값이 없으면 lift도 정의되지 않으므로
JOURNAL §48의 "lift는 조건부이지 가산적이지 않다" 문제와 §31의 숫자 오독이 **잴 대상
자체가 없어져** 닫힌다.

분리에서 따라 나오는 것 셋:

1. **순환이 타입 수준에서 불가능해진다.** 카드가 값을 담을 수 없으므로 `0.3`
   (= LeCaRDv2 train 최적값)을 쥐여주는 일이 **문법적으로 안 된다**. JOURNAL §33이
   사후에 발견해 지운 사고가 구조적으로 재발하지 않는다.
2. **예산 경쟁이 사라진다.** `use_recipe`는 제안이 아니라 **cap 변경**이다.
3. **데이터셋 전이가 가능해진다.** pool을 데이터셋별로 격리해야 했던 이유(JOURNAL §34)는
   카드가 **값**을 실어 날랐기 때문이다. 간선은 코퍼스 독립, 값은 아니다.

**`Constraint`가 가드를 대신한다.** 지금 `lexical_first_light_semantic`은 semantic을
켜면서 동시에 점수 fusion으로 전환시켜 **우연히** −0.115 함정의 가드 노릇을 한다(§2).
분리하면 그것은 부수 효과가 아니라 명시적 하드 제약이고, 위반은 **칸을 소비하지 않고**
거부·재제안시킨다. `attempt_history`에도 실패가 아니라 **무효 제안**으로 기록한다.

#### ③′ Rule — 질의 재작성 규칙

**이것이 비교 대상 논문의 구조다.** "When Rules Learn"(Tao et al. 2026)이 진화시키는 것은
설정이 아니라 **재작성 규칙 텍스트**이고(Figure 1: 유의어·법률용어 변형 확장), 사람이 쓴
규칙 1장을 seed로 준다. 우리 레시피 5장은 그 자리에 **설정 카드**를 넣어 둔 것이라 논문과
같은 모양이 아니었다.

규칙으로만 표현되는 것들이 있다. 둘 다 지금 action space에 자리가 없거나 눌려 있다.

| 쓰고 싶은 것 | 지금 | 규칙이면 |
|---|---|---|
| 같은 질의를 여러 번 반복해 가중을 준다 | `query_rewrite_repeat` 정수 하나(×4 → −3.1 pp) | **어느 부분을** 몇 번 — 세밀도가 다르다 |
| 법률 요소(죄명·구성요건·유의어)를 **더해** 검색한다 | **표현할 방법이 없다** | 등록부의 빈칸 `(expand, bm25, view)`를 채운다 |

```python
@dataclass(frozen=True)
class Rule:                      # ③′ — 텍스트. 필드가 없다
    name: str
    text: str                    # 재작성 LLM에게 가는 지시 (코퍼스 언어)
    output_format: str           # "[용어 A], [용어 B], ..." 같은 형식 제약
    coordinate: tuple[str, str]  # 이 규칙이 만드는 입력 좌표, 예: ("expand", "bm25")
    rationale: str
```

**이 타입이 앞의 문제 둘을 동시에 푼다.**

- **"정리하면 ②의 공간이 줄어든다"**(§10.0) — 규칙은 필드가 아니라 텍스트라 공간이
  유한하지 않다. LLM이 규칙을 쓰면 **등록부에 새 입력 좌표가 생긴다.** ②가 ①을 넓히는
  합법적 통로는 이것뿐이다.
- **등록부의 빈칸** — 확장 레버는 필드로는 만들 수 없다. 규칙으로만 만들어진다. 그리고
  §3이 적은 대로 질의를 바꾸는 레버 둘이 **모두 삭제**인 지금, 확장이 비어 있는 것은
  이 타입이 없었기 때문이다.

**비용 구조가 ②·③과 다르다 — 바깥 축이다.** 규칙을 바꾸면 **재작성 캐시를 다시 만들어야
한다**(쿼리당 LLM 1회, LeCaRDv2 800건). 라운드당 레버가 될 수 없고, `DOC_FIELD_OF`와
같은 성격의 **데이터셋당 몇 번 도는 바깥 축**이다. `learn_recipes.py`가 루프 **밖에서**
도는 구조가 여기에 맞는다 — 규칙 승인은 그 자리에서 하고, 루프 안에서는 "그 규칙을
쓸지"만 ②가 정한다.

**비교 가능성은 어떻게 지키나.** 규칙이 새 입력 좌표를 만든다는 것은 LLM이 tier ①을
건드린다는 뜻이고, §0.3이 ①을 닫아 둔 이유가 여기 걸린다. 해소는 하나다 — **규칙 pool은
네 데이터셋에 같은 것을 주고, 쓸지 말지를 ②가 정하게 한다.** 그러면 시스템은 하나로
남고, 갈리는 것은 설정뿐이다. 데이터셋별로 **승인된 규칙이 달라지는 것**은 허용하되
(그것이 "데이터셋마다 다른 값에 도달한다"의 일부다), **pool 자체는 공유**한다.

#### 마이그레이션 — 손으로 쓴 5장

| 지금 | 분리 후 |
|---|---|
| `lexical_first_light_semantic` | **Bundle**{`fusion_mode`, `semantic_top_k`} + **Constraint**(semantic on & `first_found` 금지). 값 `0.3`·`100`은 버린다 |
| `entity_swap_shallow` | **삭제** — 깊이 스칼라. trigger는 `state_notes()`로 |
| `query_rewrite_append_once` | **Rule로 이전** — 지금 이 카드가 켜는 `enable_query_rewrite`는 "삭제형 재작성을 한 번 덧붙인다"이고, 그 재작성 지시문이야말로 ③′다(§3 등록부의 R2 waiver) |
| `undo_structured_view` | **삭제** — 단일 필드 되돌리기 |
| `undo_chapter_filter` | **삭제** — 기본값이 이미 `False`였다(JOURNAL §48의 오독) |

남는 것은 **Bundle 1 + Constraint 1 + Rule 1(기존 재작성 프롬프트)**이다. 그리고 Rule
쪽이 비로소 늘어날 수 있는 자리다 — 확장 규칙이 들어갈 곳이 거기다.

#### 그러면 Bundle은 여전히 필요한가 — 2×2로 판정

Bundle의 유일한 기능은 **cap을 여는 것**이므로, cap을 그냥 열어 주는 것과 구별되는지만
보면 된다. (Rule에는 이 질문이 걸리지 않는다 — 대체물이 없다.)

|  | `step_cap = 2` | `step_cap` 완화 |
|---|---|---|
| Bundle 있음 | ① | ② |
| Bundle 없음 | ③ | ④ |

`①−③`이 Bundle의 명목 효과, `④−③`이 **제약 완화만으로** 얻는 효과.
**`①−③ ≈ ④−③`이면 Bundle의 독립 기여는 0**이다 — 그것은 버그가 아니라 보고할 결과다:
*"행동 공간이 작고 필드가 대체로 분리 가능하면 번들 추상화는 제약 완화로 대체된다."*

순서는 **타입 분리 → pool 마이그레이션 → 2×2**다. 먼저 정리하지 않으면 ①이 단일 필드
카드 넷에 희석된다.

#### 코드 반영 상태 (2026-11-16)

**벤더는 포크하지 않았다.** 세 타입은 우리 쪽 `registry.py`에 있고, 벤더 `Recipe`는
LoCoMo용으로 그대로 둔다.

| | 어디 | 상태 |
|---|---|---|
| **Rule** | `registry.RULES` → `query_rewrite.PROMPTS`가 **상수가 아니라 조립 결과** | ✅ 연결됨 |
| **Constraint** | `registry.CONSTRAINTS` → `legal_retriever.search()`가 순회하며 거부 | ✅ 연결됨 |
| **Bundle** | `registry.BUNDLES` → shim + cap seam | ✅ 연결됨 (2026-11-22). 아래 |
| 마이그레이션 | 5장 → Bundle 1 + Constraint 1 + Rule 1 | ✅ |
| 게이트 | `verify.py` | ✅ 아래 넷 |

#### Bundle이 루프에 닿는 방법 — shim + cap seam (2026-11-22)

Bundle을 벤더 `Recipe`로 **변환**하는 것은 원리적으로 불가능하다 — `Recipe.proposal`이
바로 Bundle이 담기를 거부하는 값이다. 그런데 벤더 경로에 우리가 원하는 성질이 **이미**
하나 있다: 호출된 레시피의 필드는 step cap에서 면제된다. 그 성질만 빌린다.

**① shim** — `registry.as_vendor_shims()`가 Bundle마다 `proposal={}`인 벤더 `Recipe`를
만든다. 엔진이 이름을 해석해 `report.invoked_recipe`에 넣어 주므로 **어느 번들을 골랐는지
알 수 있고**, 병합할 값이 없으므로 **값은 하나도 주입되지 않는다.** `_already_rejected({})`는
구조상 `False`다(스칼라 집합이 비면 즉시 반환) — 번들은 수가 아니므로 "이미 시도해 거부됨"이
성립하지 않는다는 뜻이고, 의미가 정확히 맞는다.

**② cap seam** — 벤더는 호출된 레시피에 **칸 하나를 청구한다.** 아무것도 설정하지 않는
shim이 칸을 먹으면 번들이 없느니만 못하다. 그래서 `LegalEvolutionEngine`이
`self.diagnostics.suggest_config_update`를 감싼다.

```
cap' = cap + |{LLM이 이번 라운드에 제안한 필드} ∩ {번들의 필드}|
```

**번들 밖 필드는 원래 cap을 그대로 두고 경쟁한다.** 번들을 호출했다고 라운드가 무제한이
되면 다음 진단이 무엇 덕분이었는지 귀속할 수 없다. 그리고 `invoked_recipe`는 벤더 호출
동안만 비우고 **끝나면 되돌린다** — `evolution.py`가 그 뒤에 라운드 기록에 적기 때문이고,
번들이 기록에서 사라지면 실행을 읽을 수 없게 된다.

**벤더는 포크하지 않았다.** 감싸는 대상은 우리 객체에 붙은 바인드 메서드다.

실측(격리 테스트, cap=2, 제안 = `fusion_mode`·`semantic_top_k`·`weight_semantic`):

| | 적용된 필드 |
|---|---|
| 번들 없음 | `fusion_mode`, `weight_semantic` — **`semantic_top_k`가 잘린다** |
| 번들 호출 | 셋 전부 (cap 2 → 4) |
| 번들 호출 + 번들 밖 1필드 추가 | 넷 전부 (cap 2 → 4: 번들 2 면제 + 기본 2) |

첫 줄이 번들이 존재하는 이유 그 자체다 — fusion만 바뀌고 깊이가 잘린 중간 상태는 θ₀보다
나쁘고, elitist가 되돌린다.

**실행 arm**: `--recipes bundles`. 손으로 쓴 pool을 **대체**한다(합치지 않는다) — 둘은
같은 프롬프트 자리를 두고 다투는 다른 타입이라, 섞으면 "번들이 도왔는가"를 답할 수 없다.

**타입이 규칙을 강제한다.** `Bundle.__post_init__`이 `len(fields) < 2`를 거부하므로
**단일 필드 카드는 생성 자체가 안 된다** — JOURNAL §33이 기준을 '스칼라'로 좁게 적어
불리언 하나짜리를 남긴 누락이 "기억해야 할 규칙"이 아니라 **없는 필드**가 됐다.

**게이트 넷** (`verify.py --only levers`, 음성 테스트 전부 통과):

| 검사 | 실패 조건 |
|---|---|
| Bundle의 필드 | 액션 공간에 없는 필드를 묶으면 실패 — 못 움직이는 필드에 cap을 열어 봐야 소용없다 |
| Bundle shim | shim이 **값을 담으면 실패**, 그리고 **코퍼스 언어 번역이 없으면 실패** — `install_recipe_text`는 인덱스를 다 만든 뒤 **실행 중에** 예외를 던진다 |
| cap seam 생존 | `LegalEvolutionEngine._install_bundle_cap_seam`이 없으면 실패 — 그 경우 번들이 칸만 먹고 아무것도 설정하지 않는다 |
| Constraint 생존 | θ₀를 거부하면 실패(실행 불가), **벤더 기본값을 아무도 거부하지 않아도 실패** — −0.115 함정이 다시 열린 것이다 |
| Rule의 좌표 | `coordinate`가 등록부에 없으면 실패. 규칙은 칸을 만들 수 있지만 **등록부가 그것을 기록해야 한다** |
| 캐시 귀속 | 조립된 재작성 프롬프트가 seed의 원문과 **바이트 단위로** 다르면 실패 |

마지막 것이 조용한 사고를 막는다. `cache/rewrite/*.json`은 **쿼리 id로 키가 걸려 있어**
프롬프트를 바꿔도 캐시가 무효화되지 않는다 — 그러면 옛 재작성본이 **새 규칙의 산물로
기록된다.** 그래서 seed 규칙은 과거 상수와 바이트 동일하게 렌더되고(세 언어 확인),
새 pool을 쓸 때 캐시 옆에 남길 식별자로 `query_rewrite.cache_stamp()`를 뒀다
(`strip_normalize|zh|a6f52e2653df` 형태).

> **남은 반영**: `diagnosis_prompts.py`의 블록 분리(선택지 / 규칙 — 지금 번들 설명은
> 레시피 자리에 그대로 들어간다), `recipe_learning.admit`의 두 경로 분기(Bundle=상호작용
> 검정 · Rule=효과 검정). **`measure_recipes.py`는 2026-11-22에 Rule 전용으로
> 축소됐다**(207 → 151줄, `rule_lifts.json`을 쓴다).
> 그리고 **step-cap 회계 예외**(§3 B의 `fusion_mode` 2칸)는 아직 없다 — 지금 번들이
> 그 의존을 cap 쪽에서 우회하고 있을 뿐이다.

### "레시피 pool"이란

벤치마크별 레시피 목록. 코드상으로는 딕셔너리 하나다.

```python
RECIPE_POOLS = {
    "locomo":          RECIPES_LOCOMO,   # 14장 (레포 원본)
    "legal_retrieval": RECIPES_LEGAL,    #  5장 (손으로 작성) + 학습된 카드
}
pool = RECIPE_POOLS.get(benchmark, [])
```

어댑터 이름(`legal_retrieval`)으로 자기 pool을 고른다. §7의 **"법률 pool이 비어 있었다"**는 이
dict에 우리 항목이 없어 `[]`가 반환됐다는 뜻이고, 그래서 진단 프롬프트의
`{available_recipes}` 자리가 매 라운드 비어 있었다.

LoCoMo 14장을 그대로 쓸 수 없는 이유는 **측정 근거가 다른 도메인**이기 때문이다. 정체 탈출
카드만 봐도 LoCoMo는 "top-k를 키워라", 판례는 **반대 방향**(더 얕게)이다(§8).

### pool 전체가 매 라운드 보이는 것은 아니다

```
pool (손 5장 / 학습된 n장 / 둘 다 / 빈 pool — `--recipes`)
  → trigger(현재 설정 · attempt_history)를 통과한 것만
  → 이미 거부된 적 있는 수는 제외           ← match_recipes의 공통 가드(§9)
  → {available_recipes} 로 프롬프트에 노출
  → LLM이 use_recipe로 고르거나 무시        ← 강제가 아니라 제시
```

θ₀에서는 `lexical_first_light_semantic` 한 장만 뜬다(§9의 trigger 맵). 정체 시 뜨던
`escape_plateau_shallower_swap`은 §33에서 제거됐다 — 제안 전체가 깊이 스칼라였다.

### 45개 필드 전부

★ = 우리가 추가한 차원. θ₀는 `evaluate.weak_config()`가 설정하는 값.

**A. view 깊이 — 각 view가 후보를 몇 개 뽑는가 (4)**

| 필드 | 레포 기본 | θ₀ | 측정 |
|---|---|---|---|
| `keyword_top_k` | 8 | **100** | BM25. **미측정** — sweep한 적 없다(JOURNAL §25) |
| `semantic_top_k` | 20 | **0** | |
| `structured_top_k` | 5 | **0** | metadata view. **10이 최적**, 그 이상 해로움 |
| `max_context` | 25 | **100** | fusion 후 반환 개수. **RRF 깊이의 상한**(§2) |

**B. fusion (4)**

| 필드 | 값 | 측정 |
|---|---|---|
| `fusion_mode` | 받는 값 **6 → 5** (`first_found` 차단, 2026-11-16). 목표는 `weighted_sum`/`rrf` **2값** — `*_only` 셋은 §2의 B 이후 | **단독 Δ를 내지 않는다** — view 조합과의 상호작용이 주효과보다 크다(−0.010 ↔ +0.0154). (view 조합 × fusion) 16셀로만 보고. `ws → rrf` +0.0072는 **그때의 view 조합 아래에서**의 값 |
| `weight_keyword` | 1.0 | 진화가 0.8로 낮춘 적 있음 — **일화이고 sweep이 아니다**(JOURNAL §25) |
| `weight_semantic` | 1.0 | **0.3 근처만 유효**, 1.0↑ −2.4 pp |
| `weight_structured` | 0.5 | 0.8 → −9.0 pp |

`first_found`와 `*_only` 세 값은 **제안 대상에서 뺐다** — 전자는 캐스케이드, 후자는 A·C의
깊이·스위치 필드로 이미 표현되는 view 마스크다. 분해의 근거·선행 검증·코드 반영 상태는
§2 "fusion_mode는 차원이 아니다". **필드 수는 그대로 12개**이고, 줄어든 것은 한 필드가
갖는 값의 수다 — **지금은 6 → 5**(`first_found`만 차단), `*_only` 셋까지 옮기면 2가 된다.

**step-cap 회계 예외.** `fusion_mode` 변경은 **2칸**으로 센다(혹은 그 모드에서 비로소
의미가 생기는 필드 하나와 묶어 1칸). 이 필드만 다른 필드의 의미를 재정의하므로, 1칸으로
세면 LLM이 "싼 수"로 오인하고 elitist가 그 중간 상태를 되돌린다 — §2의 4번 이유.

**C. party-strip view (3)**

| 필드 | 기본 | 측정 |
|---|---|---|
| `enable_entity_swap` | True | **유일한 스위치**(게이팅 제거, §4) |
| `entity_swap_keyword_k` | 10 | **30이 최적**(+1.1 pp), 100은 +0.4 |
| `entity_swap_semantic_k` | 15 | 동일 |

**D. 쿼리 변형 — 검색 전에 쿼리를 바꾼다 (9)**

| 필드 | 측정 |
|---|---|
| ★`enable_query_rewrite` | **+1.3 pp** (repeat=1) |
| ★`query_rewrite_repeat` | ×1 최적, ×4 → −3.1 pp |
| `enable_query_decomposition` · `decomposition_max_subqs` · `decomposition_merge_top_k` | 판례엔 쪼갤 "홉"이 없다 |
| `enable_intent_planning` · `intent_max_subqs` · `intent_merge_top_k` · `intent_final_context` | **−8.8 pp ~ +0.6 pp로 불안정** |
| `enable_coverage_reflection` | 미측정 |

**주의**: `intent_planning`은 **쿼리마다 LLM을 호출한다.** 켜지면 "LLM 0회/쿼리"라는 우리
구조(§2)가 깨지고 라운드 시간이 급증한다.

**E. 후처리 (5)**

| 필드 | 상태 |
|---|---|
| `reflection_rounds` | 검색 차원. `category==5` 예외 제거함(§4) |
| `coverage_max_rounds` · `coverage_gap_max_queries` | 미측정 |
| `mmr_diversity_weight` · `mmr_candidate_pool` | **미탐색** — 다양성 재랭킹 |

**F. 필터 · 시간 (4)**

| 필드 | 측정 |
|---|---|
| ★`enable_chapter_filter` | **−8.3 pp** (해로움) |
| `time_decay_half_life_days` · `reference_date` | 미측정 |
| `per_category_overrides` | test +0.0027 → 레버 늘자 0 (§6) |

**G. KG 확장 (5) — 전부 미시도**

`enable_kg_expansion` · `kg_expand_top_k` · `kg_seed_top_k` · `kg_min_score` · `kg_keep_strategy`

**H. 답변 생성 (11) — 검색에서 완전히 무효**

`answer_style` · `enable_answer_verification` · `verification_style` · `answer_model` ·
`answer_model_ensemble` · `locomo_cat1~5`

`_generate_answer`를 doc_id 반환으로 덮었으므로 **아무 효과가 없다.** 그대로 두면 진단 LLM이
매 라운드 이것만 제안하고 매 라운드 거부당한다(실측: delta 정확히 +0.0000 반복).
`PROMPT_EXCLUDE`로 프롬프트의 차원 카탈로그에서 제거한다.

**요약**

| 그룹 | 개수 | 상태 |
|---|---|---|
| A view 깊이 | 4 | 측정 완료 |
| B fusion | 4 | **최대 레버** |
| C party-strip | 3 | 측정 완료 |
| D 쿼리 변형 | 9 | 재작성 ✓ / 분해 ✗ / intent 불안정 |
| E 후처리 | 5 | **MMR 미탐색** |
| F 필터·시간 | 4 | chapter ✗ / 시간감쇠 미측정 |
| G KG | 5 | **전부 미시도** |
| H 답변 생성 | 11 | **무효 — 프롬프트에서 제외** |

**실질 탐색 공간은 34개**(45 − 11)이고, 그중 **측정한 것은 11개**다.

### action space에 없는 것 — 차원이 아닌 구조적 결정

진화 엔진이 **제안할 수 없는** 값들이다. 전부 코드에 상수로 박혀 있다.

| | 현재 | 왜 문제인가 |
|---|---|---|
| `first_found`의 view 순서 | 하드코딩 `(semantic, keyword, structured, swap)` | 판례에선 BM25가 최강인데 순서가 거꾸로다(§2) |
| RRF의 `k=60` | 하드코딩 | 깊은 지표에서 유효 하이퍼파라미터가 된다(§2) |
| BM25 `k1` / `b` | `rank_bm25` 기본값(1.5 / 0.75) 고정 | 판례는 길이 편차가 극단적(COLIEE `Fact` 최대 116,864자) → **길이 정규화 `b`가 유망** |

셋 중 **`b`가 가장 아깝다. 한 번도 건드리지 않았다.**

### 우리가 추가한 차원

논문 §3.3의 self-expanding 주장과 같은 성격(진단이 원래 없던 차원을 제안할 수 있다).
둘 다 기본값이 꺼짐이라 기존 동작은 불변이다.

| 차원 | 내용 | 실측 |
|---|---|---|
| `enable_chapter_filter` | 쿼리와 다른 형법 장의 판례를 점수 계산 전에 제외 | **−8.3 pp** (해로움) |
| `enable_query_rewrite` + `query_rewrite_repeat` | LLM 재작성본을 원쿼리 뒤에 n회 덧붙임 | **+1.3 pp** (repeat=1) |

재작성은 **사전 계산해 캐시**하므로 진화 중 LLM 호출이 추가로 들지 않는다. 이것이 없으면
우리 최고 레버를 진화 엔진이 도달할 수 없다.

## 4. 도메인 적응 — 무엇을 왜 바꿨나

논문 구현에는 LoCoMo 전제가 하드코딩된 곳이 여럿이다. 각각 **최소 수정 + 기본값 불변**으로
주입 가능하게 바꿨다(LoCoMo 동작은 그대로).

| 하드코딩 | 문제 | 조치 |
|---|---|---|
| entity-swap이 `category == 5`에서만 발동 | LoCoMo의 적대적 split 전용. 판례에서 당사자 이름은 **모든 쿼리에서** 무관하다 | **게이팅 자체를 제거**했다 — `enable_entity_swap`이 유일한 스위치 (아래) |
| `diagnosis.py`가 카테고리를 `{1:"SingleHop", 5:"Adversarial"}`로 명명 | 중국어 판례 검색이 **대화 QA로 라벨링되어** 진단 LLM에 전달됐다 | `set_category_names()`로 법률 클래스명 주입 |
| 프롬프트 차원 카탈로그에 `locomo_cat*` 포함 | 무효한 레버를 제안하게 유도 | `set_prompt_exclude()` |
| 실패 샘플이 질문 90자 + 예측 80자 | 예측이 doc_id 숫자뿐이라 **왜 틀렸는지 알 수 없다** | `set_sample_widths()` — 검색된 판례의 죄명·법조문·법원·날짜 노출 |

→ 조치 후 진단이 법률 용어로 말하기 시작했다
(*"what crime was committed?", "was there coercion?"*). **다만 레버 선택은 바뀌지 않았다.**

### `category == 5`를 전부 제거했다 (2026-10-29)

레포는 `category`를 **두 가지 뜻으로** 쓴다 — per-category override의 키이자, LoCoMo
적대적 split의 번호다. 우리 `category`는 법률 클래스이므로 후자는 성립하지 않는데,
세 군데가 5번을 특별 취급하고 있었다.

| 위치 | 하던 일 | 영향받던 클래스 |
|---|---|---|
| `multi_retriever.py` | entity-swap을 `category == 5`에서만 발동 | — (우회 중이었음) |
| `evolution.py` | `category == 5`면 intent planning·decomposition **건너뜀** | **57건** |
| `evolution.py` | `category == 5`면 reflection **건너뜀** | 동일 |

하필 걸린 클래스는 **등록 순서라는 우연**으로 정해졌다:

| 데이터셋 | category=5 | 쿼리 |
|---|---|---|
| LeCaRDv2 | A8_탐오뇌물 | 57건 |
| LEGAR | **A_살인** | 168건 |
| COLIEE | A_노동연금 | 39건 |
| LeCaRD | A3_시장질서 | 13건 |

세 곳을 모두 제거했다. 그 결과 **우회도 함께 사라졌다** — 전에는 "게이팅을 풀려고 5를 넘기고,
그러면 override가 깨지니 래퍼에서 미리 해석한다"는 2단 우회였는데, 이제
`enable_entity_swap`만으로 켜지고 `category`는 본래 뜻 하나만 갖는다. 랭킹은 불변임을
세 설정(θ₀ 0.3948 / `ws_sem0.3_swap30` 0.4057 / `rrf+swap30+str10` 0.4101)에서 확인했다.

남은 `category == 5`는 전부 **답변 생성 경로**이고, `_generate_answer`를 doc_id 반환으로
덮었으므로 도달하지 않는다.

---

## 5. 평가 프로토콜

- **주 지표 NDCG@30.** 기존 `compute_query_metrics`를 그대로 import (재구현하지 않음) —
  기존 BM25 결과와 같은 열에 놓기 위해.
- **run 파일**(`{query_id: [doc_id,...]}`)을 항상 남긴다. 어떤 방법이든 재검색 없이
  동일 기준으로 재채점할 수 있다.
- **train / val / test 3단**(2026-10-30). 진화는 **train**에서만 돌고, 확인과 오류 수정은
  **val**에서 하며, **test는 마지막에 한 번** 본다. 이전에는 train → test뿐이었는데,
  실행 사이에 test를 들여다보면 그것이 사실상 검증셋이 된다.
- **공식 split을 쓴다.** 네 벤치마크 모두 `result/{ds}/query_structured.json`에 train/val/test가
  있고 우리 쿼리와 100% 매칭된다. 우리가 만든 층화 20% 분할은 폐기했다.

| 데이터셋 | train | val | test |
|---|---|---|---|
| LeCaRDv2 | 576 | 64 | 160 |
| LeCaRD | 79 | **8** | 20 |
| COLIEE | 1,151 | 127 | 400 |
| LEGAR | 4,339 | 429 | 1,192 |

  **LEGAR는 두 소스의 split이 다르다.** `output/`에도 val이 있는데(476건) `result/`의
  val(429건)과 **완전히 배타적**이다 — 한쪽의 val이 다른 쪽에서는 전부 train이고, **test
  1,192건만 동일**하다. 둘 다 재분할본이고 어느 쪽이 더 공식이라 할 근거가 없어,
  **네 데이터셋을 한 소스(`result/`)로 맞추는 쪽**을 택했다. `assign_fold()`는 매칭되지 않는
  쿼리가 하나라도 있으면 **예외를 던진다** — 조용히 폴백하면 두 실행이 서로 다른 held-out을
  쓰게 된다.

  **LeCaRD val은 8건이라 선택 기준이 될 수 없다**(θ₀가 train 0.6683 / val 0.5104 /
  test 0.6427). 이 데이터셋은 val 기준 선택을 참고용으로만 다룬다.

### θ₀ (BM25 단독, NDCG@30, 공식 fold)

| 데이터셋 | train | val | test |
|---|---|---|---|
| LeCaRDv2 | 0.4024 | 0.4180 | 0.3948 |
| LeCaRD | 0.6683 | **0.5104** | 0.6427 |
| COLIEE | 0.1839 | 0.1817 | 0.2000 |
| LEGAR | 0.0363 | 0.0366 | 0.0371 |

COLIEE·LEGAR는 세 fold가 거의 같고, LeCaRD만 fold 간 편차가 크다(8건 때문).

### 라운드 선택은 val에서 한다 (`select_on_val.py`)

진화 루프 자체는 **train만** 본다(누출 없음). 엔진은 train 최고 라운드를 고르는데, 그것이
전이되는 라운드라는 보장이 없다. 라운드별 설정이 `round_N.json`에 전부 남아 있으므로
**재검색 없이** 사후에 val로 재채점한다.

> 선택 규칙: **train과 val 양쪽에서 θ₀를 넘긴 라운드 중 val이 가장 높은 것.**
> 둘 다 못 넘기면 θ₀를 유지한다.

train 조건을 같이 거는 이유는 val이 작기 때문이다 — 한두 쿼리로 순위가 뒤집히는 크기에서
val만 보면 운 좋은 라운드를 고르게 된다. 두 기준의 결과를 `best_config.json`(train)과
`best_config_val.json`(val)으로 **나란히 남겨** 갈리는지 본다.
- 네 데이터셋 **동일 전처리**(`Fact`+`Reason`+`Result`를 색인, 세 필드 보유 문서만).
  2026-11-10에 `Fact`만 색인하던 것을 세 필드 전부로 바꿨다(`DOC_FIELD_OF = "all"`). **메타데이터 추출(`legal_refs`·`parties`)은 처음부터 세 필드 전부를 훑었으므로** (`build_cases.py:82`, `--meta-fields` 기본값) 이 변경으로 추출 결과는 달라지지 않았다 — 달라진 것은 BM25·semantic이 읽는 본문뿐이다. 바꾸기 전에는 **메타데이터 view가 BM25가 볼 수 없는 `Reason`·`Result`의 조문까지 매칭**하는 비대칭이 있었고, 지금은 사라졌다.
  데이터셋마다 입력 텍스트가 다르면 프레임워크 비교가 교란된다.
- **pool 마스킹을 쓰지 않는다**(2026-10-29 결정). LeCaRD의 공식 과제는 쿼리당 후보 100개
  재랭킹이지만 나머지 셋은 전체 코퍼스 검색이다. 한쪽만 pool을 쓰면 **서로 다른 과제의 숫자를
  같은 표에 올리게 된다** — LeCaRD θ₀가 전체 코퍼스 0.6427에서 subpool 0.8254로 뛴다.
  `evaluate.py` · `sweep.py` · `legal_evolve.py` 모두 `--use-pool` **기본 off**이고,
  켜면 공식 프로토콜을 재현한다(참고용).

### noise floor — 카드가 아니라 설정 쌍에서 (2026-11-22)

수용 임계값은 **fold의 불확실성**을 넘어야 한다. 넘지 못하면 루프가 "아무것도 안 한 것과
구분되지 않는" 라운드를 수락하고, elitist가 그 상태를 최고 기록으로 들고 간다.

검색은 결정적이다 — 같은 (쿼리, 설정)은 항상 같은 점수다. 불확실한 것은 **이 fold가
쿼리의 한 표본**이라는 점이고, 그래서 재는 방법은 쿼리 단위 paired bootstrap이다.
바뀐 것은 **무엇을 쌍으로 삼는가**다.

| | 옛 방식 (~2026-11-22) | 지금 |
|---|---|---|
| 쌍의 출처 | `measure_recipes.py`가 측정한 **레시피 카드** | **고정된 설정 쌍 13개** (`noise_floor.py`) |
| 의존 | pool — 카드가 들고 나면 임계값이 움직인다 | 없음 |
| 표본 수가 늘면 | pool이 줄수록 **나빠진다** | 쌍을 늘릴수록 **좋아진다** |
| 편향 | 카드는 "도움이 되리라 본 수"라 몇 필드에 몰린다 | 두 기반(θ₀ · 융합 상태) × 한 필드 이동 |

**옛 숫자가 유령으로 돌아가고 있었다.** 살아 있던 임계값을 만든 카드 8장 중 3장은 이미
pool에 없었고(§33에서 제거), 2026-11-22에 3장이 더 지워졌다. LeCaRD의 0.0228은 카드
**둘**의 중앙값이었고 그중 하나가 §48이 오독이라 판정한 `undo_chapter_filter`(반폭 0.048)다.

실측(LeCaRD train n=79, 비퇴화 쌍 11개): **0.0228 → 0.0103.** 옛 값은 이상치 둘이
중앙값을 끌어올린 결과였다.

쌍은 **고정 목록**이고 무작위 표집이 아니다 — 같은 코퍼스에서 두 번 돌리면 같은 임계값이
나와야 한다. 스스로 움직이는 임계값이 이 파일이 없애려는 문제다. 반폭 0인 쌍은
버린다(그 데이터셋에서 작동하지 않는 레버는 **아무것도 아닌 것의 잡음**을 잰다).

> **폴백과 게이트**: `recipe_lifts.json`에서 읽는 옛 경로는 남아 있되 출처 문자열이
> `FALLBACK ...`으로 시작하고, **`verify.py`가 그 상태에서 실행을 막는다.** 읽히는
> 폴백은 유지되는 폴백이다.

### 검증의 위계

측정값을 믿기 전에 세 단계를 거친다. 이 프로젝트에서 **단계를 건너뛴 지표는 모두 틀렸다**.

1. **재현** — 래퍼가 기존 baseline을 소수점까지 재현하는가. (네 데이터셋 전부 통과)
2. **train → test 전이** — train에서 고른 설정이 test에서도 유효한가.
   (per-category override: +0.0072 → +0.0027로 **63% 증발**)
3. **레버가 쌓인 뒤에도 남는가** — 다른 레버 추가 후 재측정.
   (per-category override: 레버 추가 후 **0.0000**)

### sweep 설계가 결론을 만든다 (실패 사례)

수동 sweep으로 `rrf_hybrid`(0.3926) < `ws_sem0.3`(0.4047)를 재고 **"RRF는 유망하지 않다"고
접었다.** 그 sweep은 entity-swap을 **끈 채** fusion 모드만 비교했다. swap을 켜고 재면 순서가
뒤집힌다 — RRF+swap이 `weighted_sum`+swap을 test에서 **+0.0044** 앞선다.

이유는 구조적이다. 가중합은 점수 스케일이 다른 view(BM25 수천 점, semantic 0~1, 구조화
일치 개수, swap)을 정규화해 섞어야 하는데, **RRF는 순위만 쓰므로 그 문제가 없다.**
view가 늘수록 RRF가 유리해진다. 즉 **fusion 모드는 단독 차원이 아니라 활성 view 수에 의존하는
조건부 차원**이고, view를 고정한 채 비교하면 그 의존성이 보이지 않는다.

→ 한 차원을 평가할 때는 **그 차원이 상호작용하는 축을 고정하지 말 것.**
이 함정은 진화 엔진이 R1에서 `weighted_sum → rrf`로 바꿔 +0.0072를 얻으면서 드러났다.

### 레버가 독립적인지 보는 법 (2×2 요인 설계)

평균 비교만으로는 "합쳐서 얼마"만 알 수 있다. 같은 쿼리 집합에 네 조건을 돌린다.

```
효과(A) = ②−①,  효과(B) = ③−①,  관측 = ④−①
상호작용 = 관측 − (효과A + 효과B)
```

더 중요한 것은 **쿼리 단위 분석**이다 — `corr(Δ_A, Δ_B)`와 도움받은 쿼리 집합의 겹침.
entity-swap과 쿼리 재작성에 적용한 결과 **corr +0.524, Jaccard 0.463, 해를 입은 쿼리도 90건 겹침**
→ 같은 조작임이 드러났다(재작성 프롬프트가 "성명 제거"를 지시하므로 swap을 내포한다).
평균만 봤다면 "합쳐서 +0.0063"만 알고 **왜**는 몰랐을 것이다.

---

## 6. 쿼리 분류 (per-category override)

### 왜 하는가 — 측정 결과가 아니라 요구사항이다

**실제 판례 검색은 죄종별로 들어간다.** 사기 사건을 찾는 사람과 교통사고를 찾는 사람이
같은 검색 설정을 쓸 이유가 없다. 그러니 시스템은 **카테고리별 설정을 지원해야 하고,
이것은 깨서는 안 되는 제약**이다 — 성능이 오르면 하고 안 오르면 마는 선택지가 아니다.

그리고 **카테고리는 데이터셋마다 다르다.** 각 법제가 자기 분류 체계를 갖고 있으므로
중국 형법의 장(章) 체계에 한국·캐나다 데이터를 맞추지 않는다. 데이터셋별로 **그
벤치마크가 스스로 정의한 체계**를 쓴다(아래 표).

측정은 이 요구가 **헛수고가 아님을 확인한 것**이다. 같은 한 수(`semantic 0.3`)가
A2에서 +0.0119, A3에서 −0.3421로 **부호가 뒤집히므로** 전역 설정 하나는 원리적으로 둘을
동시에 만족시킬 수 없고, 클래스별 조립이 val에서 +0.0079 대 전역 +0.0012였다
(JOURNAL §15). 순서를 뒤집어 읽으면 안 된다 — **요구가 먼저이고 측정이 뒤따랐다.**

### 어떤 체계로 접는가

클래스당 쿼리가 충분해야 설정을 적합할 수 있다. 죄명(LeCaRD 57종/107쿼리)은 너무 잘다.
**각 벤치마크가 스스로 정의한 체계**로 접는다 — 임의로 만든 축은 근거가 없다.

| 데이터셋 | 근거 | 클래스 |
|---|---|---|
| LeCaRD / v2 | 중국 형법 분칙 장 (조문 번호에서 **결정적** 유도) | 7 / 9 |
| LEGAR | **벤치마크 공식 d1** (`Crime` 필드) | 8 |
| COLIEE | 법 분야 (벤치마크 태그가 데이터와 불일치해 본문에서 추출) | 7 |

**폐기한 Axis B** — BM25 점수 첨예도로 난이도를 3분할했으나 버렸다. 교훈이 분명하다:

- `gap` ↔ NDCG 상관 **+0.147**로 약했다 (질의 시점에 알 수 없는 정답 개수가 +0.353)
- **난이도를 예측하는 것과 "다른 설정이 필요하다"를 예측하는 것은 다르다.** 전역적으로 좋은
  레버가 생기자 per-class 이득이 0이 됐다
- `gap`은 BM25 자신의 점수 분포인데 그것으로 "BM25에 무엇을 섞을지"를 정하는 구조였다

→ **분류 축은 외부 근거(공식 분류 체계)가 있어야 한다.** 성능에서 유도한 축은 순환이다.

---

## 7. self-evolution 루프

엔진의 루프(diagnose → propose → guard → revert)는 **건드리지 않는다**. 세 seam만:

1. `evolution.MultiViewIndex` → 미리 만든 `LegalIndex` 반환 팩토리
2. `_generate_answer` → 답변 대신 **doc_id 반환**. 답변 생성을 루프에서 제거해
   질의당 LLM 호출이 0이 됐다 — 논문이 "라운드당 15~20분"이라 한 비용이 사라진다.
   **다만 "라운드당 1회"는 2026-11-11까지 틀린 서술이었다**: Phase 7b의
   meta-analyzer가 매 라운드 두 번째 호출을 하고 있었고(temp 0.2), elitist가 그
   결과를 전부 버렸다. 껐다 — JOURNAL §34. 지금은 라운드당 1회(진단)다.
3. 어댑터가 doc_id를 qrels로 채점 → NDCG@30

`_verify_answer`도 no-op으로 덮어야 했다. 검증기가 doc_id 목록을 산문으로 재작성해 랭킹을
파괴했고(0.4048 → 0.0002), 프롬프트가 판례 100건을 담아 **256,000 토큰**을 요청했다.
**revert 가드는 정상 작동해 이 붕괴를 되돌렸다.**

### 엔진의 3층 구조 — 진단과 현재

진화 엔진은 한 겹이 아니다. 초기 네 번의 실행에서 실제로 돈 것은 **첫 번째 층뿐이었고**,
그것이 R1 정체의 원인이었다. 2층을 채워 해결했다.

| 층 | 모듈 | 보는 범위 | 초기 4회 실행 | **현재** |
|---|---|---|---|---|
| per-round 진단 | `diagnosis.py` | 이번 라운드 스냅샷 | 작동 | 작동 |
| 레시피 (capability card) | `evolution_cookbook.py` | 실패 패턴 → 검증된 설정 bundle | **pool이 비어 있음** | **`RECIPES_LEGAL` 투입 → 해결**(§9). 당시 8장, 현재 5장이고 그중 4장은 단일 필드다 |
| meta 분석 | `meta_analysis.py` | 라운드 **이력 전체** | 호출되나 제안 폐기 | **변동 없음**(`elitist=True` 유지) |

**결과**: 2층을 채우자 R1 정체가 풀렸다 — R0에서 레시피 발동, R1·R2 연속 ACCEPT,
train 0.4184 / test 0.4101로 **수동 튜닝(0.4057)을 처음 이겼다**(§10).

3층은 의도적으로 그대로 둔다. 판단 근거는 **"레시피가 정체 시 발동하면 그 자체가 탐색이고,
진단 경로를 타므로 step-cap 안에서 작동한다"** — 즉 2층이 3층의 역할을 대신한다. 레시피
투입 후에도 정체하면 그때 3층을 연다(§9 "남은 두 지점").

> 아래 세 절(레시피 차단 / meta 분석 / 정체의 구조)은 **해결 전 진단 기록**이다.
> 현재 상태가 아니라 "왜 막혀 있었는가"의 분석으로 읽어야 한다.

### 레시피 — 법률 pool이 비어 있었다 (해결됨)

```python
# evolution_cookbook.py:383  ← 수정 전
pool = RECIPES_LOCOMO if benchmark == "locomo" else []
```

우리 어댑터 이름은 `legal_retrieval`이라 **15개 레시피가 전부 차단**됐다. 진단 프롬프트의
`${available_recipes}` 슬롯이 매 라운드 비어 있었다.

**조치** — 벤치마크 이름으로 pool을 고르는 구조로 바꾸고(2줄, LoCoMo 불변), 우리 측정 결과로
법률 레시피를 새로 작성했다(§9). 당시 8장이었고 지금은 5장이다.

```python
RECIPE_POOLS = {"locomo": RECIPES_LOCOMO, "legal_retrieval": RECIPES_LEGAL}
pool = RECIPE_POOLS.get(benchmark, [])
```

LoCoMo 15장을 그대로 쓰지 않은 이유는 대화 메모리 전제이기 때문이다. 예컨대 정체 탈출
레시피가 LoCoMo에서는 `escape_plateau_scalar_sweep`("top-k를 키워라")인데, 판례에서는
**탈출 방향이 아래쪽**(swap을 얕게, 가중치를 낮게)이라 정반대다(§8).

레시피는 단순 제안이 아니라 **실패 패턴 → trigger 조건 → 검증된 설정 bundle → 기대 이득**을
묶은 것이다. 그중 `escape_plateau_scalar_sweep`은 `_consec_noaccept(h) >= 2`로 발동하는
**정체 탈출 장치**다. 우리 R2~R6이 정확히 그 상태였는데 법률 레시피가 없어 아무것도 발동하지 않았다.

→ 앞서 "레포가 논문 Eq.4의 explore 분기를 빠뜨렸다"고 본 진단은 **부정확하다**.
빠진 것이 아니라 **벤치마크별 레시피로 구현**되어 있고, 법률 pool이 없었다.

### meta 분석 — 호출되지만 결과를 쓸 수 없다

`META_PROMPT`의 규칙이 우리 상황을 정확히 겨냥한다:

> *"If latest round's primary − first round's primary < 1pp after 3 rounds, set
> `decision_type=explore` and pick ONE field to randomly perturb"*
> *"If all knobs have been tried and nothing moves primary, set `decision_type=propose_new_dim`"*

그런데 `elitist=True`(우리 설정)에서:

```python
if not self.config.elitist and meta_plan.parameter_suggestions:   # 건너뜀
if not self.config.elitist and meta_plan.per_category_overrides:  # 건너뜀
if meta_plan.new_dimension_proposal:                              # 파일에 기록만
```

`explore`는 섭동값을 `parameter_suggestions`에 담는데 그것이 적용되지 않는다. 레포가 이유를
주석에 적어 두었다 — 메타 제안이 step-size 상한과 `attempt_history`를 우회해 단조 진화를
깬다는 것. 즉 **단조성과 탐색이 맞교환되어 있고 레포는 단조성을 택했다.**

추가로 `META_PROMPT`의 action space는 **하드코딩된 LoCoMo 부분집합**이다. 검색에서 무효한
`answer_style`은 들어 있고, 우리가 측정한 유일한 유효 레버의 필드인
`entity_swap_semantic_k`/`keyword_k`와 `enable_chapter_filter`는 **빠져 있다**.
`diagnosis.py`에 적용한 `PROMPT_EXCLUDE`·`CATEGORY_NAMES` 교체가 여기에는 미적용이다.

### 정체의 구조 (해결 전 진단)

초기 4회 실행(4B·30B, 메타데이터 주입 전후) **모두 R1에서 멈췄다**. 아래가 그 메커니즘이고,
**레시피 투입으로 해결됐다** — 원인은 모델이 아니라 "의존하는 설정들을 한 번에 넘길 수단이
없었다"는 것이었다(§2 "바깥 루프", §9 "레시피가 실제로 하는 일").

```
거부 → 설정이 R1으로 복원 → 진단 입력이 동일 → temperature=0 → 출력 동일 → 같은 제안 → 거부
```

세 층 중 둘(레시피·meta)이 비활성이므로 **탈출 경로가 모두 막힌 상태**였다(현재 2층은 열렸다). 4B는
`enable_answer_verification`을, 30B는 `enable_intent_planning`(−0.088)을 무한 반복했다.
`attempt_history`는 정상 전달된다(`Round N: {...} — delta=-0.0874 REJECTED`). 모델이 무시한다.

### 관측 불가 구간

`meta_analysis.analyze()`는 실패 시 `logger.warning` 후 `None`을 반환하고, 호출부는 그것을
그대로 받는다. **성공했는지 실패했는지가 결과물에 남지 않는다.** `meta_proposals.jsonl`이
비어 있는 것이 (a) 호출 실패인지 (b) 모델이 제안하지 않은 것인지 지금 구분할 수 없다.

의심되는 원인 하나: 메타 호출도 `max_tokens=2048`이다(진단 경로는 `MIN_COMPLETION_TOKENS=8192`로
덮었으나 메타는 안 덮었다). **단 "reasoning 모델이 `<think>`에 예산을 쓴다"는 설명은 우리
설정에 해당하지 않는다** — `Qwen3-30B-A3B-Instruct-2507`은 thinking 모델이 아니다.
남는 가능성은 단순히 JSON이 2048을 넘겨 잘리는 것이고, **아직 미확인이다.**

> 관련 함정(2026-10-29 실측): 이 비-thinking 모델에 `--reasoning-parser qwen3`을 붙이면
> 답변 전체가 `reasoning` 쪽으로 분류되어 `message.content`가 `None`이 된다.
> `make_llm_call`은 `content or ""`를 반환하므로 **매 라운드 빈 제안**이 되고, 7라운드가
> 전부 θ₀에 머문다. 서버 기동 시 이 플래그를 붙이지 말 것.

→ **다음 실행 전에 메타 호출의 성공/실패와 `decision_type`을 라운드 기록에 남겨야 한다.**

### 비 evolution 모듈 (조사 완료, 우리와 무관)

`self_upgrade`(876) · `replay`(558) · `policy_optimizer`(173) · `candidate`(127) ·
`promotion`(44) — 2,573줄 중 **2,215줄이 §3.1 메모리 저장소와 운영 중인 메모리 서비스의
온라인 정책 튜닝**에 속한다. `MemoryPolicyState`(검색 모드·주입 예산)를 다루지
`RetrievalConfig`를 다루지 않으며 `evolution.py`가 import하지 않는다. 진화 루프와 무관하다.

## 8. 검색 지형에 대해 밝혀진 것 — 다음 진화의 전제

> **이 절의 숫자는 새 코퍼스(`Fact`+`Reason`+`Result`, 임베딩 8192) train 기준이다.**
> 옛 코퍼스(`Fact`만, 512 토큰)에서 잰 측정과 그때의 결론 넷이 뒤집힌 경위는
> **JOURNAL §52**로 옮겼다.

### 새 코퍼스의 실측 (2026-11-11, train, Δ vs θ₀)

아래 두 표가 현재 유효한 지형이다. 그 아래의 옛 절들은 역사 기록이다.

**fusion 7설정** (`sweep.py --grid fusion --split train`)

| 설정 | LeCaRDv2 | LeCaRD | COLIEE | LEGAR |
|---|---|---|---|---|
| θ₀ (BM25 단독) | 0.3885 | 0.6637 | 0.2413 | 0.0516 |
| **② RRF+semantic** | **+0.0140** | **+0.0134** | **+0.0041** | **+0.0062** |
| ③ RRF+str10 | −0.0035 | −0.0223 | −0.0069 | −0.0011 |
| ④ RRF+sem+str10 | +0.0126 | +0.0013 | +0.0011 | +0.0051 |
| ⑤ RRF+swap30+str10 | −0.0014 | −0.0349 | −0.0063 | +0.0002 |
| ⑥ RRF+swap30 | +0.0001 | −0.0347 | −0.0005 | +0.0004 |
| ⑦ ws_sem0.3+swap30 | +0.0117 | −0.0113 | +0.0013 | +0.0063 |

**`weight_semantic`** (rrf + semantic만, 다른 view off). ◀ = 그 데이터셋의 최적.

| w_sem | LeCaRDv2 | LeCaRD | COLIEE | LEGAR |
|---|---|---|---|---|
| 0.1 | — | +0.0152 | **+0.0059** ◀ | — |
| 0.2 | +0.0127 | **+0.0209** ◀ | +0.0046 | +0.0047 |
| 0.3 | **+0.0140** ◀ | +0.0134 | +0.0041 | +0.0062 |
| 0.5 | +0.0116 | −0.0072 | +0.0012 | +0.0073 |
| 1.0 | −0.0059 | −0.1067 | −0.0055 | +0.0100 |
| 2.0 | — | — | — | **+0.0121** ◀ |
| 5.0 | — | — | — | +0.0110 |

(전체 grid와 경계 확인은 JOURNAL §28.)

**지금 말할 수 있는 것 넷.**

1. **semantic view를 켜는 것이 유일하게 일관된 양수 레버다** — ②가 네 데이터셋 전부 양수.
   옛 코퍼스에서 ②는 중국어 둘에서 음수(−0.0101 / −0.1072)였다.
2. ~~**구조화 view는 네 데이터셋 전부 음수다.**~~ **깊이 10에서만 참이었다(2026-11-13).**
   깊이 100에서 단독(③)은 여전히 셋에서 음수지만, semantic과 함께면(④) COLIEE +0.0061·
   LEGAR +0.0064로 ②를 앞선다. **단독으로는 해롭고 보완재로는 기여한다.**
3. ~~**swap view는 죽었다**~~ **깊이 30에서만 참이었다(2026-11-13).** off/10/30/100에서
   LeCaRD +0.0134/+0.0139/+0.0172/**+0.0191** — 단조 증가이고 k=100에서 끄는 것보다 높다.
   Δ가 floor 안이므로 정직한 결론은 **"깊이 30 판정은 무효, 100은 중립"**이다. → JOURNAL §47
4. **fusion 모드는 실질적으로 무관하다** (②와 ⑦이 LeCaRDv2 0.4025/0.4002,
   LEGAR 0.0578/0.0579). 차이를 만드는 것은 **semantic view를 켜는가** 하나다.

그리고 **값 하나가 축 전체를 결정한다**: LeCaRD는 `weight_semantic`만 맞추면
+0.0209로, fusion 7설정 grid의 최고치(+0.0134)를 넘는다. 최적값은 데이터셋마다
0.1~2.0으로 **20배** 벌어지고, θ₀의 기본값 1.0에서 semantic을 켜면 LeCaRD가 **−0.1067**이다
— 그래서 (fusion + semantic 켜기 + 가중치)는 분리 불가능한 삼중이고, 카드가 가중치를
박아 두면 어느 데이터셋에서는 반드시 손해다.

---

## 9. 법률 보완 — RECIPES_LEGAL

> ### 현재 상태 (2026-11-22)
>
> **이 절의 본문은 pool이 8장이던 시기에 쓰였다.** 추론 과정은 그대로 두되, 지금
> 돌아가는 것은 아래가 맞다. 본문에서 "8장"을 읽으면 **"그때는 8장"**으로 읽어야 한다.
>
> | | 그때 (11-11) | 지금 |
> |---|---|---|
> | pool | 손으로 쓴 카드 8장 → 5장 | **세 타입**: Bundle 1 + Constraint 1 + Rule 1 (§3) |
> | 카드가 담는 것 | 증상 → trigger → 설정 bundle → 기대 이득(pp) | Bundle은 **필드 집합만**, Rule은 **텍스트만**. 값도 기대 이득도 담지 않는다 |
> | 기대 이득 | 항상 `(expected +N pp)` | **Rule만** 측정 대상이고(`rule_lifts.json`), 그 데이터셋 train 측정이 있을 때만 찍는다 |
>
> 왜 줄었나 — 세 단계였다. (1) **스칼라 크기뿐인 카드 넷**을 뺐다. 그런 카드는 구조를
> 담지 않고 step cap 이득도 없으며(1필드 = 1칸), 남는 것은 상수다. `undo_semantic_overweight`
> 의 `0.3`은 **LeCaRDv2의 train 최적값**이었고 LEGAR에는 **반대 방향**을 가리켰다 —
> 정답을 카드가 쥐여주고 있었다(JOURNAL §33). (2) 남은 5장 중 **4장이 필드 하나짜리**로
> 드러났다. (3) 2026-11-16에 기준을 필드 수가 아니라 **상호작용**으로 다시 쓰자 단일 필드
> 카드는 **정의상** 탈락했고, `Recipe` 하나가 담으려던 세 가지 일이 세 타입으로 갈렸다.
>
> 그 결과가 §3의 "레시피와 action space의 분리 — 세 타입"이고, 이 절의 나머지는 **그
> 분리 이전의 메커니즘 설명**으로 읽어야 한다.

레포 docstring이 레시피의 정체를 명시한다:

> *"Each one is a **compressed form of an experiment we ran**, checked in as code so the
> framework can autonomously access it **rather than requiring the human to re-discover it
> per-run**."* / *"every recipe must be defensible as: when symptom X is observed,
> capability Y is known to help by ~Z pp"*

즉 **사람의 실험 결과를 레시피로 압축해 넣는 것이 설계상 정상 경로**다. LoCoMo 레시피 15장도
그렇게 만들어졌다. 우리가 측정한 11개 레버가 정확히 이 형식이므로, 수동 sweep은 버려지는 작업이
아니라 **레시피의 재료**였다.

> **이 절은 메커니즘만 담는다.** 2026-10-29~11-11의 레시피 수정 기록 268줄
> (카탈로그 "작성한 8장", RRF 이후 수정, 실행에 붙여 보며 고친 것, TODO 체크리스트
> 경쟁, 남은 두 지점)은 **[JOURNAL §9-구](JOURNAL.md)로 옮겼다.** 날짜 붙은 발견과
> 실수는 일지에 있어야 하고, 그 분리가 세 파일로 나눈 이유다(README).
>
> **지금 상태**: 손으로 쓴 풀은 **5장**이고 그중 **4장은 action space 필드 하나짜리**
> 제안이다 — 진단 LLM이 카드 없이 직접 쓸 수 있으므로 묶음의 의미가 없다. 유일한
> 2필드 묶음(`lexical_first_light_semantic`)은 데이터셋마다 부호가 뒤집힌다
> (LEGAR +0.0101 / LeCaRD −0.1068). **행동 공간이 작고(12개) 필드가 대체로 독립적이면
> 레시피의 역할이 없다** — 이것이 이식 과정에서 나온 결과이고, 그래서 **LLM이 직접
> 제안하는 쪽**으로 간다(아래 "다음 (C)", `learn_recipes.py`).
>
> 묶음이라는 장치 자체는 action space 안에 이미 있다 — `per_category_overrides`는
> 조건 + 여러 필드 묶음 + 1칸이라는 점에서 레시피와 구조가 같고, 조건이 **실행 상태**가
> 아니라 **쿼리의 카테고리**다. 그쪽이 실제로 더 많이 움직였다(val +0.0079 대 전역
> +0.0012). 레시피의 자리가 좁은 이유 하나가 이것일 수 있다.

### 용어 — "레시피"란 무엇인가

**레시피 = `Recipe` 데이터클래스의 인스턴스 1개.** 필드 6개를 가진 파이썬 객체다
(`evolution_cookbook.py:31`).

레포 docstring이 이것을 프롬프트에 노출할 때 *"available capability **cards**"*라고 부르므로
"카드"라는 말이 섞여 쓰였으나 **같은 객체의 다른 이름**이다. 이 문서는 **"레시피"로 통일**한다.

| 필드 | 타입 | 뜻 |
|---|---|---|
| `name` | str | 식별자. LLM이 이 문자열로 호출한다 |
| `description` | str | LLM에게 보여줄 설명문 |
| `trigger` | **함수** | 이번 라운드에 이 레시피를 노출할지 판정 |
| `proposal` | dict | **실제로 적용될 `RetrievalConfig` 필드 bundle** |
| `expected_lift_pp` | float | 우리가 측정한 기대 이득(%p) |
| `benchmark` | str | 어느 어댑터용인가 (우리는 `"legal_retrieval"`) |

**`trigger`를 뺀 전부가 하드코딩된 리터럴이다.** 런타임에 계산되는 값이 아니다.
실제 소스(`evolution_cookbook.py:384`):

```python
Recipe(
    name="lexical_first_light_semantic",
    description=(
        "Foundation. Keep BM25 dominant and blend the semantic view in at a "
        "low weight. Semantic alone scores far below BM25 on case law "
        "(0.28 vs 0.40); it only helps at weight ~0.3, and degrades "
        "monotonically above it. Required before entity-swap, which does "
        "nothing without score fusion."
    ),
    trigger=lambda r, c, h: (
        c.get("fusion_mode") in ("keyword_only", "first_found")
        or c.get("semantic_top_k", 0) == 0
    ),
    proposal={
        "fusion_mode": "weighted_sum",
        "semantic_top_k": 100,
        "keyword_top_k": 100,
        "weight_keyword": 1.0,
        "weight_semantic": 0.3,
    },
    expected_lift_pp=0.1,
    benchmark="legal_retrieval",
),
```

- `description`의 `0.28 vs 0.40`, `weight ~0.3` — **우리 수동 sweep 측정값을 손으로 적은 것**
  (§5의 sem_only 0.2814 vs θ₀ 0.4040). 생성되지 않는다
- `expected_lift_pp=0.1` — 같다. 측정값 상수
- `proposal`의 5개 필드 — **적용되는 유일한 부분**
- `trigger` — 유일한 함수. 매 라운드 호출된다

### 발동 메커니즘 — 4단계

함수가 도는 것은 **①뿐**이다.

**① trigger 평가** (`match_recipes()`, `evolution_cookbook.py:542`) — 라운드당 1회

```python
pool = RECIPE_POOLS.get(benchmark, [])      # "legal_retrieval" → 우리 pool
for r in pool:
    if r.trigger(report_fields, current_config, attempt_history):
        matched.append(r)
```

`trigger(r, c, h)`의 세 인자는 **(이번 라운드 성적 요약, 현재 설정 dict, 과거 시도 이력)**이다.
위 레시피는 `c`만 본다 — "fusion이 꺼져 있거나 semantic top-k가 0이면". 즉 **설정 상태에 대한
순수 조건식**이고 성능을 보지 않는다. 우리 레버가 전부 설정 기반이라 QA 전용 신호
(`abstention_count` 등)에 의존하지 않고 trigger를 쓸 수 있었다.

**② 프롬프트 문자열로 렌더링** (`format_recipes_for_prompt()`, `:577`)

발동한 레시피만 이 모양이 된다:

```
• `lexical_first_light_semantic` (expected +0.1 pp)
    Foundation. Keep BM25 dominant and blend the semantic view in at a low weight. ...
    Fields set: fusion_mode, semantic_top_k, keyword_top_k, weight_keyword, weight_semantic
```

이것이 진단 프롬프트의 `{available_recipes}` 자리(`diagnosis.py:104`)에 들어간다. 발동한 것이
없으면 `(no recipes triggered this round)` — **법률 pool이 비어 있던 이전 4회 실행이 매 라운드
이 문장이었다.**

`Fields set:`은 `proposal` dict의 **키 이름만** 뽑은 것이다 (`:583`):

```python
prop_keys = ", ".join(r.proposal.keys())     # ← 값이 아니라 키만
```

| `proposal` (코드에 박힌 값) | → | LLM이 보는 것 |
|---|---|---|
| `fusion_mode: "weighted_sum"` | | `fusion_mode,` |
| `semantic_top_k: 100` | | `semantic_top_k,` |
| `weight_semantic: 0.3` | | `weight_semantic` |

**값은 LLM에게 보여주지 않는다.** LLM은 "이 레시피가 fusion 모드와 view 깊이와 가중치를 건드린다"는
것만 알고 호출하고, 구체적 숫자는 ④에서 코드가 넣는다. 설계 의도는 **LLM이 검증된 숫자를 임의로
바꾸지 못하게** 하는 것이다. 이 이름들은 전부 `RetrievalConfig`의 필드명이므로, `Fields set:`은
"action space 45차원 중 이 레시피가 쓰는 칸들"을 가리킨다.

**③ LLM이 선택** — 강제가 아니라 제시다. JSON에 `"use_recipe": "<name>"`을 넣으면 선택이고,
무시할 수도 있다.

**④ `proposal`을 원자 적용** (`diagnosis.py:977`)

```python
recipe = get_recipe_by_name(recipe_name.strip(), benchmark=benchmark_name)
for k, v in recipe.proposal.items():
    report.parameter_suggestions[k] = v     # 레시피가 충돌 시 우선
```

**여기가 핵심이다.** 5개 필드가 한꺼번에 들어가고 step-cap(`max_changes_per_round=2`)은
이것을 **1칸으로 센다.** 레시피 없이 LLM이 직접 제안하면 5개를 바꾸는 데 3라운드가 걸리고,
그동안 각 중간 상태가 θ₀보다 나빠 거부된다. **이것이 이전 4회 실행이 R1에서 멈춘 이유이고,
레시피를 넣자 R0에서 바로 출발할 수 있었던 이유다.**

### 레시피가 실제로 하는 일

새로운 탐색 능력이 아니라 **묶어서 한 번에 넘기는 장치**다. 셋을 한다.

1. **원자성** — 서로 의존하는 설정 N개를 step-cap 1칸으로 전달 (**가장 큰 효과**)
2. **사전 지식 주입** — 사람의 측정 결과를 LLM이 읽을 수 있게
3. **조건부 노출** — 관련 없는 상황에서는 프롬프트에 안 보임

### 어느 레시피가 언제 쓰이는가 — 선택 규칙

카드를 다 읽지 않아도 되도록, **무엇이 언제 노출되는가**만 한 표로 모았다. trigger는
전부 `current_config`에 대한 조건식이고(하나만 이력을 본다), **성능을 보지 않는다.**
그래서 "지금 설정이 어떤 상태인가"로 노출 여부가 결정된다.

실제 pool은 **5장**이고, 데이터셋별로 4장이 된다(`enable_query_rewrite`가 rewrite
테이블이 있는 LeCaRDv2에서만 산다). 원칙은 **카드 하나는 메커니즘 하나**다.

| 지금 설정 상태 | 노출되는 레시피 | 제안 | 메커니즘 |
|---|---|---|---|
| fusion 꺼짐(`keyword_only`) 또는 `semantic_top_k==0` | `lexical_first_light_semantic` | `fusion_mode: rrf`, `semantic_top_k: 100` | **전제** (fusion+semantic 켜기) |
| fusion 켜짐 + strip 꺼짐 | `entity_swap_shallow` | `enable_entity_swap: True` | party-strip 켜기 |
| fusion 켜짐 + strip 꺼짐 + 재작성 꺼짐 | `query_rewrite_append_once` | `enable_query_rewrite: True` | 재작성 |
| `structured_top_k>0` | `undo_structured_view` | `structured_top_k: 0` | metadata 끄기 |
| `enable_chapter_filter` 켜짐 | `undo_chapter_filter` | `enable_chapter_filter: False` | 장 필터 끄기 |

**다섯 장 중 네 장의 제안이 필드 하나다.** 그러면 진단 LLM이 카드 없이 그 필드를 직접
쓸 수 있으므로 묶음의 의미가 없다 — 이것이 §33의 삭제 기준("제안 전체가 action space에
이미 있는 필드")에 똑같이 걸리는데, 당시 기준을 '스칼라'로 좁게 적어 불리언 하나짜리를
남겼다. 유일한 2필드 묶음은 데이터셋마다 부호가 뒤집힌다(LEGAR +0.0101 / LeCaRD −0.1068).

**제거된 카드 (§33, 2026-11-11)**: `swap_depth_shallow`, `undo_semantic_overweight`,
`undo_structured_overweight`, `escape_plateau_shallower_swap` — 제안 전체가 action space
필드의 스칼라 크기였다. `undo_semantic_overweight`의 0.3은 LeCaRDv2의 train 최적값이고
LEGAR(최적 2.0)에는 **반대 방향**을 가리켰다.
`undo_query_decomposition`도 제거됐다 — 그것이 끄던 세 레버가 action space에서 빠져
**trigger가 영구 거짓**이 됐고, 재측정이 네 데이터셋 전부 +0.0000 / CI[0,0]으로 확인했다.
`trigger_base()`는 카드가 trigger가 받는 상태에서 측정되는지는 검사하는데, 그 상태가
**도달 가능한지**는 아무도 검사하지 않았다.

**카드가 갖지 않는 것: 크기.** `weight_semantic`·깊이 같은 값은 LLM이 고르고, 카드는
**구조**(어느 필드가 함께 움직여야 효과가 있는가)만 말한다. 근거는 §8의
`weight_semantic` 표다 — 최적값이 0.1~2.0으로 벌어지므로 **한 상수는 어딘가에서 반드시
손해**다.

### 많아지면 프롬프트가 터지지 않는가 — 실측 0~2장

trigger가 상태 가드이므로, **pool이 커져도 한 라운드에 노출되는 수는 따라 커지지 않는다.**
LeCaRDv2 클래스별 실행의 실제 라운드 설정으로 `match_recipes()`를 다시 돌려 센 값:

| 라운드 상태 | 발동 수 | 어느 것 |
|---|---|---|
| θ₀ | 1 | `lexical_first_light_semantic` |
| A2 r1 | 2 | `entity_swap_shallow`, `query_rewrite_append_once` |
| A2 r2 | **0** | — |
| A2 r3 | 1 | ~~`undo_query_decomposition`~~ (이후 제거) |
| A8 r3 | 2 | `entity_swap_shallow`, ~~`undo_query_decomposition`~~ |
| A8 r4~r6 | **0** | — |

> **이 표는 2026-11-02 pool(8장)로 센 것이고 제거된 카드를 포함한다.** 지금 pool(5장)에서는
> 발동 수가 더 적다. 숫자가 아니라 **"pool이 커져도 노출 수는 따라 커지지 않는다"**는
> 성질을 읽을 표다.

**최대 2장**이고 0장인 라운드가 흔하다. trigger들이 서로 배타적으로 설계된
결과다 — `lexical_first_light_semantic`은 fusion이 꺼진 상태에서만, `entity_swap_shallow`는
켜진 상태에서만 열린다. 켜기 레시피가 전제를 요구하므로 **순서가 상태로 강제된다.**

0장인 라운드가 중요하다. 그때 LLM은 **원시 필드만으로** 제안해야 하고, §15에서 본
반복 제안과 정체가 거기서 난다. 즉 레시피가 없는 것이 아니라 **지금 상태에 해당하는
레시피가 없는 것**이다.

### 여러 장이 동시에 열리면

| 단계 | 규칙 |
|---|---|
| 정렬 | **선언 순서 유지.** `expected_lift_pp`로 정렬하지 **않는다** |
| 제시 | 발동한 전부를 프롬프트에 나열 |
| 선택 | LLM이 `use_recipe`로 **최대 1장**. 무시해도 된다 |
| 중복 차단 | 이미 거부된 `proposal`은 노출 단계에서 제외(`_already_rejected`) |

**lift로 정렬하지 않는 이유가 코드 주석에 남아 있다.** 기대 이득이 큰 장이 자신의
전제보다 앞에 오면, 낮은 temperature의 LLM이 **순서가 틀린 수**를 고른다. LoCoMo에서
`cat1_cat4_paraphrase_killer`(+8 pp)가 RRF(+6 pp)보다 먼저 선택돼 R2에서 0.29로
퇴행한 기록이 그것이다. 우리 pool도 같은 구조였다 — `undo_*` 카드의 기대 이득이 8~9 pp로 가장 컸는데, 그것들은
**해로운 설정이 켜져 있을 때만** 의미가 있다. (그 8~9 pp 자체가 §31에서 문제가 됐다:
머리말 숫자가 중요도와 **반대로** 정렬돼 있었고, 이제 해당 데이터셋 train 측정이 있을
때만 찍는다.)

선언 순서는 그래서 **의존 순서**다: 켜기(전제→strip→재작성) → 되돌리기.

### 지금 가는 방향 — LLM이 레시피를 제안한다 (2026-11-13 결정)

손으로 쓴 pool이 위와 같이 비어 있으므로, **레시피를 사람이 쓰지 않고 LLM이 제안하게
한다.** "When Rules Learn"(Tao et al. 2026)의 구조다 — 그 논문도 사람이 쓴 규칙 1장을
seed로 주고 시작하며, Limitations에서 *"다른 벤치마크에 적용할 때는 적절한 seed 규칙을
주는 것이 좋다"*고 명시한다. 위 5장이 그 seed에 해당한다.

그리고 근거가 하나 더 있다. **LLM이 낸 카드는 묶음이었다.** 옛 코퍼스에서 승인된 4장은
전부 4~5개 필드를 함께 움직였다(예: `{fusion_mode: weighted_sum, semantic_top_k: 50,
mmr_diversity_weight: 0.2}`). 사람이 쓴 카드가 단일 필드로 환원된 것과 정반대다.

#### 언제 도는가 — 루프 밖이다

`learn_recipes.py`는 **진화 실행 전에, 따로** 돈다. 루프 안에 접으면 한 라운드의 결과가
같은 실행 중에 변한 pool에 의존하게 되고, 그러면 "어떤 pool이 이 성적을 냈나"를 말할
수 없다. 분리해 두면 **한 pool을 다른 pool과 같은 루프에서 비교**할 수 있다.

#### 승인 — 세 관문

제안 하나당 **LLM 1회 + train-dev 채점 2회**(기반과 후보)다. 그리고:

| 관문 | 내용 |
|---|---|
| 1 | paired bootstrap **CI 하한 > 0**, 수준은 같은 fold에 던진 후보 수로 보정(`ALPHA/k`) |
| 2 | 평균 > **그 데이터셋의 측정된 noise floor** |
| 3 | **종점 > θ₀** — 아무도 서지 않을 분기에서 개선해 1·2를 통과할 수 있다 |

2번이 2026-11-13에 고쳐졌다. 그때까지 `0.003`(LoCoMo 상수)이었고 **넷 중 셋에서 noise
floor 아래**였다. `admit(threshold=)`는 **키워드 전용·기본값 없음**이라 호출자가
`noise_floor(dataset)`를 넘겨야 한다.

**단 floor는 full train에서 쟀고 승인은 train-dev(20%)에서 한다.** 작은 fold가 더
시끄러우므로 floor는 **하한**이고, 실제로 묶는 것은 dev 델타로 bootstrap한 1번이다.
LeCaRD는 train 79 → dev 28이므로 **아무것도 승인되지 않을 가능성이 높다** — 그것은
버그가 아니라 그 데이터셋의 해상도다.

#### 기반은 그 카드의 trigger를 만족하는 상태다

후보는 θ₀가 아니라 **자기 trigger가 참인 설정**에서 측정한다. θ₀에서 재면 trigger가
거짓인 카드의 제안이 아무것도 바꾸지 않아 Δ=0이 된다. 다만 경계값을 쓰면 안 된다 —
`> 0`에 대해 `semantic_top_k=1`을 잡으면 기반이 0.0814로 떨어지고 후보가 **+0.3092**로
읽혀, 그 숫자가 진단 LLM에게 **측정된 사실**로 제시된다. 그래서 `REPRESENTATIVE`는
"누군가 실제로 서 있을 수 있는 설정"이다(깊이 전부 100, `weight_semantic` 0.3).

#### 데이터셋별로 격리한다

`as_recipes(dataset=)`가 그 데이터셋에서 승인된 카드만 돌려준다. **이 필터가 없던 것이
오염의 원인이었다**(JOURNAL §34) — 승인된 4장 중 3장이 LEGAR에서만 측정됐는데 LeCaRD와
COLIEE 루프가 그 4장을 전부 받고 있었다. `dataset`은 `measured_block`이 **항상 기록하고
있었고 아무도 읽지 않았다.**

#### 네 개의 arm

```
--recipes hand      손으로 쓴 5장 (위 표)
--recipes learned   그 데이터셋에서 승인된 LLM 제안 카드만
--recipes both      둘 다
--recipes none      빈 pool — LLM이 원시 필드로 같은 설정을 조립해야 하고
                    필드마다 step-cap 1칸을 쓴다
```

`learned`는 손으로 쓴 pool을 **먼저 비우고** 넣는다. 이전에는 플래그 둘
(`--no-recipes`/`--learned-recipes`)이 상호 배타여서 **"학습된 카드만"이 도달 불가**였다.

`--exclude-recipe-dev`를 함께 써야 한다. 그러지 않으면 **카드를 승인한 fold가 그 카드로
점수를 내는 fold와 같다.**

#### 프롬프트가 LLM에게 말하는 것

제안 프롬프트는 네 view와 그 입력, **깊이가 전부 100이라는 측정**, 이미 있는 카드(중복
금지), 각 필드를 몇 장이 이미 건드리는지, 거부된 후보, 그리고 **루프가 라운드당 몇 필드를
움직일 수 있는지**를 담는다.

마지막 항목이 2026-11-13에 틀린 값이었다. 상수 하나가 두 일을 하고 있었다 — 루프의
step-cap(**2**)과 카드가 담을 수 있는 필드 수(**5**)가 같은 `MAX_FIELDS`였고, 프롬프트는
"루프는 라운드당 5개 바꿀 수 있다"고 알려주고 있었다. **레시피의 논거 자체가 그 숫자다**
("한 칸으로 여러 필드"). `STEP_CAP = 2`로 분리했다.

#### 측정할 것

| 질문 | 어떻게 |
|---|---|
| LLM이 사람보다 나은 카드를 내는가 | `hand` vs `learned` vs `both`, 같은 루프 |
| 레시피가 애초에 필요한가 | `none` vs 나머지 — 행동 공간 12개에서 묶음의 값 |
| 묶음의 조건이 상태인가 쿼리인가 | 승인된 카드가 `per_category_overrides`로 대체 가능한 종류인지 |

세 번째가 열려 있는 질문이다. 묶음이라는 장치는 action space 안에 이미 있고
(`per_category_overrides` = 조건 + 여러 필드 + 1칸), 조건이 **실행 상태**가 아니라
**쿼리의 카테고리**다. 그쪽이 더 많이 움직였다(11-02 측정: val +0.0079 대 전역 +0.0012,
**옛 코퍼스·옛 임베딩 기준이므로 현재 성적과 섞어 쓸 수 없다**). 레시피의 자리가 좁은
이유 하나가 이것일 수 있다.

## 10. 현재 성적

### 10.0 이득의 귀속 — 무엇이 루프에서 왔나

이 절을 "루프가 +X를 찾았다"로만 읽으면 과대 보고가 된다. **분모가 필요하다.**
아래는 문서에 기록된 두 θ₀ 표에서 유도한 것이다 — 옛 코퍼스(`Fact`만 + 512 토큰,
JOURNAL 2026-10-29)와 새 코퍼스(세 절 전부 + 8192, §8) 양쪽 train θ₀.

| 데이터셋 | 옛 θ₀ | 새 θ₀ | Δ |
|---|---|---|---|
| LeCaRDv2 | 0.4024 | 0.3885 | −0.0139 |
| LeCaRD | 0.6683 | 0.6637 | −0.0046 |
| **COLIEE** | 0.1839 | 0.2413 | **+0.0574** (상대 **+31%**) |
| **LEGAR_stricter** | 0.0363 | 0.0516 | **+0.0153** (상대 **+42%**) |

이 Δ는 **① 두 결정의 묶음**이다(색인 절 + 임베딩 길이). §27이 둘을 같이 적용했으므로
하나로 분리할 수 없고, 그 자체가 기록해야 할 사실이다.

그리고 ②③ 루프가 찾은 것(LeCaRDv2 val, 옛 코퍼스):

| 출처 | Δ | 상대 |
|---|---|---|
| 전역 진화 7라운드 | **+0.0012** | +0.3% |
| 클래스별 조립 | +0.0079 | +1.9% |
| (참고) 수동 7설정 grid | +0.0112 | +2.7% |

**두 표를 나란히 읽어야 한다.** 사람이 내린 ① 결정 하나가 COLIEE에서 +31%이고,
루프가 찾은 최대가 LeCaRDv2에서 +1.9%다. 한 자리 차이다.

읽을 때의 유보 셋:

1. **같은 코퍼스가 아니다.** ①의 Δ는 전이 그 자체이고, 루프 숫자는 전이 **전**
   코퍼스에서 나왔다. 새 코퍼스의 루프 숫자는 아직 없다 — 진단 LLM이 막혀 있었다.
2. **루프 숫자는 val이고 ①은 train이다.** 직접 비교가 아니라 **규모의 비교**로만 읽는다.
3. **LeCaRDv2와 LeCaRD에서 ①의 Δ는 음수다.** ①이 항상 이기는 것이 아니라,
   **데이터셋에 따라 ①이 지배적이거나 무관하다.**

이 표가 말하는 것은 "루프가 쓸모없다"가 아니다. **"루프가 할 수 있는 일의 크기가
②의 공간 크기에 묶여 있고, 그 공간은 ①이 정한다"** 는 것이다. §26의 ④가 같은 것을
다른 말로 적었다 — *"가장 큰 이득은 action space 밖에 있다."*

### 10.1 LeCaRDv2 (옛 split)

> **이 절의 숫자는 공식 fold 이전의 것이다.** train 640은 우리가 임시로 나눈
> split이고, 공식 fold는 576/64/160이다(§5). 아래 test 숫자는 그 임시 split에서
> 나온 것이므로 **현재 성적으로 읽으면 안 된다.** 공식 fold의 val 숫자는
> JOURNAL §15에, test는 아직 측정하지 않았다.

LeCaRDv2, NDCG@30. 진화는 train에서만 돌리고 test는 최종 확인용.

아래는 LeCaRDv2 기준이고, **다른 데이터셋의 최적 설정은 다르다**(§8 "네 데이터셋 검증").

| | train (640) | **test (160)** |
|---|---|---|
| θ₀ (BM25 only) | 0.4040 | 0.3948 |
| 수동 sweep 최고 (`ws_sem0.3_swap30`) | 0.4155 | 0.4057 (+0.0110) |
| **진화 R2** (RRF + swap30 + 구조화 k=10) | **0.4184** | **0.4101 (+0.0154)** |

### 경고 — 이 비교는 순환이다 (2026-11-09)

위 표를 **"self-evolution가 수동 튜닝을 이겼다"로 읽으면 안 된다.** 비교의 두 항이 독립이
아니다.

```
수동 sweep 최고          ws_sem0.3_swap30          test 0.4057
진화가 쓴 레시피 2장
  lexical_first_light_semantic -> {weighted_sum, semantic_top_k:100,
                                   weight_semantic: 0.3}
  entity_swap_shallow          -> {enable_entity_swap, k: 30, rrf}
                                                  ^^^^^^^^^^^^
                                  = 수동 sweep 최고 설정과 같은 값
```

**저자가 수동 sweep의 결론을 카드로 압축해 넣었고, 루프가 그 카드를 적용했고, 그
결과를 수동 sweep과 비교했다.** 레시피의 `description`에 박힌 *"가중치 ~0.3에서만
도움"*, *"k=30이 +1.1 pp, k=100은 +0.4"*가 바로 그 sweep의 문장이다.

진화가 **실제로 더한 것은 `structured_top_k 0 → 10` 한 수**이고, 이긴 폭 +0.0044가
그것이다. 나머지 +0.0110은 sweep이 이미 찾아 둔 값이 카드를 통해 되돌아온 것이다.

그리고 위 숫자는 **test fold에서 나왔다.** 레시피 수치 자체가 test sweep 산물이므로
(§9, JOURNAL §24 이후), 이 표는 **누출된 사전 지식으로 test를 맞춘 기록**이다.

### 그래서 무엇을 주장할 수 있는가

| arm | 레시피 | 측정 상태 | 무엇을 말해 주는가 |
|---|---|---|---|
| A | 없음 | **측정됨** — 4회 실행 전부 R1 정체 | 루프는 step-cap 장벽을 **혼자 못 넘는다** |
| B | 저자 seed (당시 8장, 현재 5장) | 위 표 | **저자의 사전 지식 + 루프**. self-evolution의 증거가 아니다 |
| C | **LLM 학습분만** | **1장뿐 — arm이 성립하지 않는다** | self-evolution의 증거가 될 유일한 arm |
| D | seed + 학습분 | 미측정 | 실무상 최선 |

정직하게 쓸 수 있는 문장은 이것이다.

> 루프는 step-cap 장벽을 혼자 넘지 못한다(A). 능력 카드가 있으면 넘는다(B).
> 그 카드를 LLM이 스스로 만들 수 있다는 것은 **소규모로만 보였다**(C, 1장).

**seed를 쓰는 것 자체가 틀린 것은 아니다.** 레포 docstring이 레시피를 "사람의 실험을
코드로 압축해 프레임워크가 접근하게 한 것"으로 정의하고, "When Rules Learn"도 사람이
쓴 규칙 1장을 seed로 시작하며 새 벤치마크에는 적절한 seed를 주라고 명시한다. 빈
저장소에서 32제안·승인 0장이라는 cold start 측정도 seed의 필요를 가리킨다
(JOURNAL §23).

**틀린 것은 seed를 쓴 결과를 self-evolution의 성과로 적은 것이다.** 측정 설계가 B와 C를
분리하지 않았다.

이전 4회 실행은 레시피 pool이 비어 있어 전부 R1(0.4048)에서 멈췄다. 레시피를 넣자 R0에서
`lexical_first_light_semantic`이 다섯 필드를 **변경 1칸**으로 적용하고 출발해, R1·R2에서
두 번 연속 ACCEPT(+0.0072씩)했다. 차이는 **전제 설정을 원자적으로 넘겨받았는가**다.

진화가 찾은 두 수 중 **`structured_top_k 0 → 10`만 저자의 수동 sweep이 놓친 것**이다.
`weighted_sum → rrf`는 `entity_swap_shallow` 카드의 `proposal`에 이미 들어 있었으므로
(위 "경고" 참조) 진화가 찾은 것이 아니라 **카드가 실어 온 것**이다. 처음에는 둘 다
진화의 발견으로 적었고, 그것이 위 순환을 만든 서술이다.

## 11. 방법론상 미해결

- **진화 루프가 재현되지 않는다** — `temperature=0`인데도 두 실행의 라운드1 상태가
  **소수점 6자리까지 같은데 라운드2 제안이 갈렸다**. vLLM의 연속 배칭 + MoE 라우팅에서
  그리디 디코딩이 순서 의존이기 때문이다. 레포는 `temperature=0`을 재현성 근거로 주석에
  적어 두었으나 그 전제가 성립하지 않는다. `--seed`를 넣어 좁혔지만 닫지는 못한다.
  → **단일 실행의 차이를 "효과"라고 부를 수 없다.** 반복 실행으로 분산을 먼저 재야 하는데
  **아직 못 했다**(코드가 계속 바뀌어 고정 조건을 만들지 못했다). 이 때문에 레시피 수정
  5~8번이 점수에 준 영향은 **미측정**이다 — trigger 동작만 결정적으로 검증했다
- ~~**COLIEE·LeCaRD가 R0에서 멈춘다**~~ → **원인 규명됨 (2026-10-30). JOURNAL 참조.**
  COLIEE는 진단 LLM이 매 라운드 `""`를 반환하고 있었다. 제안이 나빴던 것이 아니라
  **제안이 아예 없었다.** LeCaRD는 여전히 train 79건의 노이즈로 설명된다
- **LEGAR 임베딩이 512 토큰** — semantic이 이긴 유일한 데이터셋인데 잘린 상태로 돌았다.
  최장 문서 45,902자. COLIEE에서 512→2048이 semantic을 63% 끌어올린 전례가 있다. 재인코딩 미착수
- **`max_seq_length`가 어디에도 기록되지 않는다** — 어느 설정으로 인코딩했는지 알려면
  다시 인코딩해 코사인을 비교해야 한다(실제로 그렇게 확인했다: COLIEE·LeCaRD 2048,
  LeCaRDv2·LEGAR 512). 사이드카로 남겨야 한다
- **per-category override의 근거** — Axis A 통제 실험(전역 vs 클래스별, 네 데이터셋 test) 미착수.
  test에서 잰 것은 한 번(+0.0027)뿐이고 그마저 조건이 바뀌었다
- **비교 가능성** — 기존 신경망 방법(BERT-aux/RankNet/Joint)은 **BM25 top-30 재랭킹**이라
  후보 집합이 다르고, `ranknet_metrics`의 NDCG는 gain이 `2**rel`이라 **오답 랭킹도 0.78**이 나온다
- **LEGAR의 과제 축소** — 공통 전처리(`Result` 채움률 14.2%) 때문에 공식 정답의 83%가
  코퍼스 밖이다. 네 데이터셋 동일 조건 비교로만 해석해야 한다
- **미개척 차원** — BM25 `k1`/`b`를 한 번도 조정하지 않았다(`k1=1.5, b=0.75` 고정).
  판례는 길이 편차가 극단적이라(COLIEE `Fact` 최대 116,864자) 길이 정규화 `b`가 유망하다.
  action space 45개 중 **측정한 것은 11개**이고, MMR(2)·KG 확장(5)·시간감쇠(2)가 미탐색이다(§3)
- **카테고리별 진화 미착수** — 지금은 데이터셋당 **전역 설정 하나**이고, 카테고리는
  `per_category_overrides` 필드로만 들어간다. 지금까지 엔진이 낸 override는 **전부 category 5**
  였는데, 그것은 레포 프롬프트의 LoCoMo 예시를 모방한 것이지 근거가 아니다.
  클래스별 독립 루프는 클래스당 train이 충분한 LEGAR(4,339)에서 먼저 시도할 만하다
- **레시피의 "안전장치"가 한 데이터셋 측정이었다** — `undo_auxiliary_overweight`에
  "보조 가중치 ≥1.0은 위험"을 넣었는데, LEGAR 진화가 찾은 최적이 **`weight_semantic=1.5`**였다.
  근거가 전부 LeCaRDv2의 `weighted_sum` 측정이었고 RRF·다른 코퍼스에서는 반대다.
  **레시피에 데이터셋 조건을 달거나 pool을 데이터셋별로 나눠야 한다**

---
