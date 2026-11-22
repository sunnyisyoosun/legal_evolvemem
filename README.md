# EvolveMem for Legal Case Retrieval

[EVOLVEMEM](https://github.com/aiming-lab/SimpleMem)(AutoResearch)의 **자가진화 검색**
구조를 법률 판례 검색에 적용한다. 네 벤치마크(LeCaRD · LeCaRDv2 · COLIEE · LEGAR),
주 지표 **NDCG@30**.

논문은 **대화 메모리 QA**(검색 → 답변 생성 → token-F1)를 전제하지만, 판례 검색은 **랭킹**이다.
논문 세 층 중 §3.3 **자가진화 엔진만** 쓰고, §3.1 메모리 저장소는 버린다(판례는 이미 검색 단위다).

**핵심 질문** — LLM 진단 루프가 **데이터셋마다 다른 검색 설정에 스스로 도달하는가**, 그리고
손으로 격자를 훑는 것보다 **싸게** 도달하는가.

**현재 답** — 아직 없다. 루프는 2026-10-31 이후 돌지 않았고, 그 사이 액션 공간·수용
임계값·레시피 구조가 모두 바뀌었다. 지금 상태는 [PROGRESS.md](PROGRESS.md)에 있다.

---

## 문서 셋 + 이 파일

| 파일 | 무엇이 있나 |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | 시스템이 **지금 어떻게 동작하는지** — 용어, 검색 구조, 45차원 액션 스페이스, 평가 프로토콜, 측정된 지형 |
| [JOURNAL.md](JOURNAL.md) | **날짜순 발견과 실수** — 왜 그렇게 정했나, 무엇이 틀렸었나. 새것부터 |
| [PROGRESS.md](PROGRESS.md) | **지금 돌고 있는 것**과 다음 순서, 실행 방법 |

세 파일로 나눈 이유: 레퍼런스와 일지가 한 파일에 섞여 있어 "지금 맞는 것"과
"그때 틀렸던 것"이 구별되지 않았다.

---

## 읽는 순서

구현 상세보다 **왜 그렇게 쟀는가**가 중요하므로 METHODOLOGY.md를 먼저 읽는 것이 좋다.

## 모듈

```
쿼리 → [메타데이터 추출] → 다중 뷰 검색 → 융합 → 상위 30
                            ├ BM25        질의 원문        (언어별: jieba / mecab / spaCy)
                            ├ semantic    질의 원문, 불변  (bge-m3, 디스크 선계산)
                            ├ metadata    법조문·죄명 일치 개수
                            └ party-strip 인명+사명+지명 제거 후 BM25 재검색
```

**view별 후보 깊이는 전부 100이다**(측정값, `measure.py grid --grid depth`). 30으로 내리면
semantic view의 이득 대부분이 날아간다. 이 100은 반환 수 30(=NDCG@30, 지표가 고정)과
무관한 **별개의 숫자**다 — view가 후보를 몇 건 내놓는가다.

`party-strip`은 새 채점 함수가 아니라 **BM25를 변환된 질의로 다시 돌린 것**이므로,
"네 개의 view"는 정확히는 **세 개의 채점 함수 + 질의 변환 하나**다. 삭제가 어휘적
행위이므로 semantic view는 무엇을 켜든 질의 원문을 읽는다(`SEMANTIC_USES_ROW_TEXT`).

| 파일 | 역할 |
|---|---|
| `legal_meta.py` | 규칙 기반 메타데이터 추출 (법조문·죄명·당사자·법원·날짜), 한자 숫자 정규화 |
| `build.py cases` | 판례·쿼리 → 검색 단위 dict (`corpus/{ds}_cases.json`) |
| `legal_retriever.py` | `LegalIndex` — 토크나이저 주입, 쿼리 확장, pool 마스킹(기본 off) |
| `legal_taxonomy.py` · `build.py classify` | 벤치마크 공식 체계로 쿼리 분류, **공식 train/val/test fold 주입** |
| `build.py embed` | 다국어 임베딩 선계산 |
| `evaluate.py` · `measure.py grid` | 단일 설정 평가 / 설정 비교 · 일반화 검증. `--grid {classes,fusion,weight,weight-edge,swap-depth,depth}` |
| `verify.py` | **실행 전 단일 관문** — 조립된 프롬프트와 semantic view 입력을 함께 단정 (`verify_prompt.py` + `derived_queries.py` 통합) |
| `registry.py` | **레버 등록부**(좌표 `(입력×채점)`) + **세 레시피 타입** `Bundle`/`Constraint`/`Rule` |
| `measure.py floor` | 데이터셋별 **수용 임계값** — 고정 설정 쌍 13개의 paired bootstrap CI 반폭 중앙값 |
| `measure.py rules` · `measure.py strip` | **Rule**별 lift 측정 / 삭제 목록 성분별 측정 |
| `learn_recipes.py` · `recipe_learning.py` | **LLM이 레시피를 제안하고 train-dev가 승인** — 측정·승인 기계와 제안 프롬프트 |
| `runs.py select` | 진화 라운드별 설정을 **val에서 재채점·선택** |
| **`legal_evolve.py`** | **진화 루프를 검색 과제로** — 어댑터 · 이음매 3개 · 진단 컨텍스트 |
| `query_rewrite.py` | LLM 쿼리 재작성 (언어별 프롬프트, 디스크 캐싱) |

**다시 만들지 말 것** — `llm_taxonomy.py`(중국어에서 규칙이 우월) · `random_arm.py`(논문
Table 6이 이미 쟀다, −9.63 F1) · `legal_recipes.py`(24장 중 16장이 스칼라를 카드에 박았다)
는 측정 후 삭제했다. 경위는 JOURNAL.

EvolveMem 레포는 **포크하지 않는다.** 진화 엔진이 튜닝할 로직이 그 안에 있으므로,
법률에서 깨지는 지점만 이음매에서 교체한다. 레포 자체 수정은 최소한이며
(`doc_id` 필드 2줄, 차원 3개, 진단 프롬프트 주입 setter) **모두 기본값 불변**이라 LoCoMo
동작은 그대로다.

## 빠른 시작

```bash
V=/home/yoosun/legal/.venv-evolve/bin/python
cd /home/yoosun/legal/evolve_legal

$V build.py cases --all                         # 1. 코퍼스 변환
$V build.py classify --all                    # 2. 분류 + 공식 fold 주입
HF_HOME=/home/yoosun/.cache/hf \
  $V build.py embed --dataset LeCaRDv2 --queries --device cuda:2   # 3. 임베딩

$V verify.py                                    # 4. 실행 전 관문 (필수)
$V evaluate.py --dataset LeCaRDv2 --split train --out runs/LeCaRDv2_bm25.json
$V measure.py floor --datasets LeCaRDv2                  # 5. 수용 임계값 (게이트가 요구한다)
$V legal_evolve.py --dataset LeCaRDv2 --split train --rounds 7 \
    --recipes bundles \
    --base-url https://openrouter.ai/api/v1 --model openai/gpt-oss-120b
$V runs.py select --dataset LeCaRDv2 --dir evolution_official --write  # val 기준 선택
```

> **test fold는 `in_split()`이 막는다.** `--split test`·`all`·`None`은 `ALLOW_TEST=1`
> 없이는 `RuntimeError`다. 문장으로 막으려 했으나 실패했고(§8의 네 데이터셋 표가
> test에서 나왔다), 이제 코드가 막는다. `evaluate.py`의 `--split` 기본값도
> `test` → `train`으로 바꿨다.

진단 LLM은 **OpenRouter**로 결정됐다(2026-11-13). 자체 vLLM을 띄우지 않으므로 다른 사용자의
GPU·포트를 건드리지 않는다. 키는 실행 시 `/home/yoosun/evolve/config.py`에서 읽고 **출력하지
않는다**. `--recipes`는 `hand|learned|both|none|bundles` 중 하나다(§3의 2×2가 쓰는 축).

## 평가 규약

- **NDCG@30**을 `multilingual_bm25_recall.compute_query_metrics`에서 **그대로 import**한다
  (재구현하지 않음). 기존 BM25 결과와 같은 열에 놓기 위해서다.
- 모든 실행이 run 파일(`{query_id: [doc_id,...]}`)을 남긴다 → 재검색 없이 재채점 가능.
- **train / val / test 3단.** 진화는 train에서만, 확인·수정은 val에서, **test는 마지막 한 번**.
  네 벤치마크의 **공식 split**을 쓴다(`result/{ds}/query_structured.json`).
- 네 데이터셋 **동일 전처리**: `Fact`+`Reason`+`Result`를 색인(`DOC_FIELD_OF="all"`, 2026-11-10), 세 필드를 모두 가진 문서만. `Fact`만 색인하던 것을 바꾼 이 결정 하나가 COLIEE θ₀를 **+31%**, LEGAR를 **+42%** 올렸다 — 루프가 찾은 최대보다 한 자리 크다.
- 네 데이터셋 **동일 과제**: **전체 코퍼스 검색**. pool 마스킹을 쓰지 않는다.

### θ₀ (BM25 단독, NDCG@30, 공식 fold)

| 데이터셋 | 코퍼스 | train | val | test |
|---|---|---|---|---|
| LeCaRDv2 | 55,192 | 0.4024 | 0.4180 | **0.3948** |
| LeCaRD | 9,088 | 0.6683 | 0.5104 | **0.6427** |
| COLIEE | 5,452 | 0.1839 | 0.1817 | **0.2000** |
| LEGAR | 14,525 | 0.0363 | 0.0366 | **0.0371** |

LeCaRD val은 **8건**이라 선택 기준으로 쓰지 않는다.

마지막 두 규약의 대가를 명시해야 한다.

**pool** — LeCaRD의 공식 과제는 쿼리당 후보 100개 재랭킹이지만 나머지 셋은 전체 코퍼스
검색이다. 한쪽만 pool을 쓰면 서로 다른 과제의 숫자를 같은 표에 올리게 된다. 공식 프로토콜로
재랭킹하면 LeCaRD θ₀는 0.6427 → **0.8254**로 뛴다. 재현하려면 `--use-pool`을 주면 되지만
(세 진입점 모두 **기본 off**), 그 수치는 다른 데이터셋과 나란히 놓을 수 없다.

**전처리** — 한국 판결문은 `Result` 채움률이 14.2%라 **LEGAR는 공식 정답의 83%가 코퍼스
밖**이다. LEGAR 수치는 "네 데이터셋 동일 조건 비교"로만 해석해야 하고 공식 LEGAR BENCH
결과와 나란히 놓을 수 없다.

## 지금까지 밝혀진 것 (요약)

아래는 **train fold** 수치다. floor를 못 넘는 Δ는 "1위처럼 보이는 자리"까지만 말한다.

> **floor 값은 이 문서에 적지 않는다.** 출처는 `noise_floor.json` 하나이고, 실행이 그 값과
> 출처를 함께 찍는다. 2026-11-22에 산출 방식을 **레시피 카드 → 고정 설정 쌍**으로 바꿨다
> (카드 기반 floor는 pool이 바뀔 때마다 움직였고, 그 숫자를 만든 카드 여덟 중 셋은 이미
> 존재하지 않았다). LeCaRD는 0.0228 → **0.0103**으로 내려갔고 나머지 셋은 재측정 중이다.

1. **최소 개입이 기본이지만, 깊이를 맞추면 view는 더 들어간다.** θ₀은 BM25 단독이고,
   8가지 view 조합을 **깊이 전부 100**으로 재면 **네 view 전부 켜는 ⑧이 세 데이터셋에서
   1위**다(LeCaRDv2 +0.0159 / COLIEE +0.0061 / LEGAR +0.0073). LeCaRD만 ②(+semantic)가
   앞선다.

| 설정 (깊이 전부 100) | LeCaRDv2 | LeCaRD | COLIEE | LEGAR |
|---|---|---|---|---|
| ① BM25만 (θ₀) | 0.3885 | 0.6637 | 0.2413 | 0.0516 |
| ② +semantic | +0.0140 | **+0.0134** | +0.0041 | +0.0062 |
| ③ +metadata | −0.0060 | −0.0306 | −0.0023 | +0.0010 |
| ④ +sem+meta | +0.0132 | −0.0071 | **+0.0061** | +0.0064 |
| ⑤ +meta+strip | −0.0034 | −0.0100 | −0.0001 | +0.0039 |
| ⑥ +strip | −0.0037 | −0.0082 | −0.0018 | +0.0009 |
| ⑦ ws+sem+strip | +0.0059 | −0.0304 | −0.0011 | +0.0062 |
| **⑧ 네 view 전부** | **+0.0159** | +0.0086 | **+0.0061** | **+0.0073** |
| noise floor | *(재측정 중)* | **0.0103** | *(재측정 중)* | *(재측정 중)* |

   **깊이 10/30에서는 ②가 네 곳 전부 1위로 보였다.** metadata는 10, party-strip은 30이던
   값이 근거 없이 흩어져 있었고, 그것을 100으로 통일하자 순위가 바뀌었다. 기본값을
   선택으로 읽으면 결론이 바뀐다는 사례가 이 프로젝트에서 세 번째다.

2. **metadata view는 단독으로는 해롭고 보완재로는 기여한다.** 단독(③)은 네 곳 중 셋에서
   음수인데, semantic과 함께면(④) COLIEE·LEGAR에서 ②를 앞선다. "네 데이터셋 전부
   해롭다"는 **단독 조건에서만** 참이다. 그 가중치(`weight_structured`)는 엔진 기본값
   0.5로 **한 번도 sweep하지 않았다** — 측정 중이다.

3. **COLIEE와 LeCaRD에서는 "어떤 view를 켤지"가 측정 불가다.** 어느 설정도 그 데이터셋의
   floor를 넘지 못한다. LeCaRD는 train 79건이라 원리적 한계다(원데이터).
   거기서 루프가 무엇을 찾아도 성과로 읽을 수 없다.
4. **최적 설정은 데이터셋마다 다르다.** `weight_semantic` 최적값만 0.1~2.0으로 **20배**
   벌어진다. 전역 기본값을 박는 대신 **데이터셋별로 엔진이 찾게 하는 것**이 이
   프레임워크의 값이다. 격자는 채점용 정답지일 뿐이다.
5. **이득의 귀속.** 사람이 내린 tier ① 결정 하나(색인 절 + 임베딩 길이)가 COLIEE θ₀를
   **+31%**, LEGAR를 **+42%** 올렸다. 루프가 찾은 최대는 **+1.9%**다. 한 자리 차이이고,
   효과가 가장 큰 축(문서 변환)이 사람 손에 있다 — 이 연구의 가장 약한 지점이고 그대로
   보고한다.
6. **벤치마크가 과제 정의와 어긋나는 지점을 쟀다.** 정답 문서가 쿼리의 당사자명을
   **+16pp**, 지명을 **+51pp** 더 공유한다. 같은 구에서 난 두 사건은 유사 사건이 아닌데
   지표는 그것을 보상한다. 그래서 `STRIP_SCOPE` 기본값을 점수가 아니라 **과제 정의**에
   맞춰 인명+사명+지명으로 두고, 두 숫자를 함께 보고한다.
7. **레시피는 한 타입이 아니라 셋이었다.** 손으로 쓴 5장 중 4장이 필드 하나짜리 제안이라
   진단 LLM이 카드 없이 직접 쓸 수 있었다. 원인을 따라가니 `Recipe` 하나가 **서로 다른 세
   가지 일**을 담고 있었다 — ② 값(`dict[field,value]`) / ③ **Bundle**(함께 움직여야 하는
   **필드 집합**, 값 없음) / ③′ **Rule**(**질의 재작성 텍스트**, 필드 없음). 셋째는
   `RetrievalConfig`에 자리가 없어 `query_rewrite_repeat` 정수 하나로 눌려 있었고, 비교 대상
   "When Rules Learn"이 진화시키는 대상이 바로 그것이다. 2026-11-22에 세 타입으로 갈라
   `registry.py`에 넣었고 5장 → **Bundle 1 + Constraint 1 + Rule 1**이 됐다. 남은 질문은
   *"Bundle이 독립 기여를 하는가, step-cap 회계의 패치일 뿐인가"*이고 2×2로 판정한다.

근거와 반례는 [METHODOLOGY.md §8](METHODOLOGY.md)에 있다 — 위 표가 그 절의 train 수치다.

## 결과를 읽기 전에 (중요)

진화 실행의 수치를 해석하기 전에 **반드시** 두 가지를 확인할 것.

### 1. 무결성 줄

`trajectory()` 바로 뒤에 찍힌다.

```
Evolution integrity: all 7 diagnosis calls OK
```

`FAILED`가 보이면 **그 실행은 라운드 수만큼의 탐색을 하지 않았다.** 진단이 실패한 다음
라운드는 제안이 없어 아무것도 시험하지 않는다. 과거에 이 줄이 없어서, 5라운드 중
2라운드가 죽은 COLIEE 실행이 "정직하게 탐색하고 아무것도 못 찾은 실행"으로 읽혔다.
죽은 라운드는 프롬프트에 `initial config [baseline]`로 렌더되기까지 했다
(JOURNAL의 해당 일자 항목).

`round_N.json`의 `llm_status`가 `ok`가 아닌 라운드는 **데이터 포인트가 아니다.**

### 2. 무효 레버 블록

진단 프롬프트에 들어간다.

```
### Levers that were ON but never acted
- query_rewrite: ON for 100 queries, acted on 0. ...
```

켜졌지만 작동할 수 없었던 필드는 **`+0.0000`을 내고 "시도했으나 실패"로 기록된다.**
그 `+0.0000`은 반증이 아니다. 특히 **rewrite 파일은 COLIEE·LeCaRD·LEGAR에 없다** —
이 세 데이터셋에서 `enable_query_rewrite`는 무효 연산이다.

### 토크나이저는 기본적으로 대체를 거부한다

`mecab`/`spaCy` import가 깨지면 베이스라인 모듈이 경고 한 줄 뒤 정규식으로 조용히
대체하고, 그 토큰이 캐시에 저장되어 의존성을 고친 뒤에도 재사용된다 — 모든 BM25 점수가
조용히 틀어진다. 지금은 기대 엔진(`jieba` / `spacy.lang.en.English` /
`mecab.mecab.MeCab`)과 다르면 **실행이 멈춘다.** 알고서 받아들이려면
`--allow-tokenizer-fallback`.

