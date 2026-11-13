# EvolveMem for Legal Case Retrieval

[EVOLVEMEM](https://github.com/aiming-lab/SimpleMem)(AutoResearch)의 **자가진화 검색**
구조를 법률 판례 검색에 적용한다. 네 벤치마크(LeCaRD · LeCaRDv2 · COLIEE · LEGAR),
주 지표 **NDCG@30**.

논문은 **대화 메모리 QA**(검색 → 답변 생성 → token-F1)를 전제하지만, 판례 검색은 **랭킹**이다.
논문 세 층 중 §3.3 **자가진화 엔진만** 쓰고, §3.1 메모리 저장소는 버린다(판례는 이미 검색 단위다).

**핵심 질문** — 사람이 하이퍼파라미터를 손으로 튜닝하는 것보다, LLM 진단 루프가 더 좋은
검색 설정을 찾아낼 수 있는가.

**현재 답** — 그렇다, 단 **레시피 풀을 채운 뒤에만**.

| LeCaRDv2 NDCG@30 | train (640) | test (160) |
|---|---|---|
| θ₀ (BM25 only) | 0.4040 | 0.3948 |
| 수동 스윕 최고 | 0.4155 | 0.4057 (+0.0110) |
| **진화 R2** | **0.4184** | **0.4101 (+0.0154)** |

진화가 찾은 두 수(`weighted_sum → rrf`, `structured_top_k 0 → 10`)는 **사람의 수동 스윕이
놓친 조합**이었다. 레시피 풀이 비어 있던 이전 4회 실행은 전부 R1에서 멈췄다.

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

| 문서 | 내용 |
|---|---|
| **[METHODOLOGY.md](METHODOLOGY.md)** | **무엇을 어떻게 재는가**, 그리고 거기서 밝혀진 것. 설계 근거·검증 위계·법률 레시피 8장 |
| [PROGRESS.md](PROGRESS.md) | 실험 기록. 원시 수치, 해결한 버그, 실행 명령 |

구현 상세보다 **왜 그렇게 쟀는가**가 중요하므로 METHODOLOGY.md를 먼저 읽는 것이 좋다.

## 모듈

```
쿼리 → [메타데이터 추출] → 다중 뷰 검색 → 융합 → 상위 30
                            ├ BM25        질의 원문        (언어별: jieba / mecab / spaCy)
                            ├ semantic    질의 원문, 불변  (bge-m3, 디스크 선계산)
                            ├ metadata    법조문·죄명 일치 개수
                            └ party-strip 인명+사명+지명 제거 후 BM25 재검색
```

**view별 후보 깊이는 전부 100이다**(측정값, `sweep.py --grid depth`). 30으로 내리면
semantic view의 이득 대부분이 날아간다. 이 100은 반환 수 30(=NDCG@30, 지표가 고정)과
무관한 **별개의 숫자**다 — view가 후보를 몇 건 내놓는가다.

`party-strip`은 새 채점 함수가 아니라 **BM25를 변환된 질의로 다시 돌린 것**이므로,
"네 개의 view"는 정확히는 **세 개의 채점 함수 + 질의 변환 하나**다. 삭제가 어휘적
행위이므로 semantic view는 무엇을 켜든 질의 원문을 읽는다(`SEMANTIC_USES_ROW_TEXT`).

| 파일 | 역할 |
|---|---|
| `legal_meta.py` | 규칙 기반 메타데이터 추출 (법조문·죄명·당사자·법원·날짜), 한자 숫자 정규화 |
| `build_cases.py` | 판례·쿼리 → 검색 단위 dict (`corpus/{ds}_cases.json`) |
| `legal_retriever.py` | `LegalIndex` — 토크나이저 주입, 쿼리 확장, pool 마스킹(기본 off) |
| `legal_taxonomy.py` · `classify_queries.py` | 벤치마크 공식 체계로 쿼리 분류, **공식 train/val/test fold 주입** |
| `build_embeddings.py` | 다국어 임베딩 선계산 |
| `evaluate.py` · `sweep.py` | 단일 설정 평가 / 설정 비교 · 일반화 검증. `--grid {classes,fusion,weight,weight-edge,swap-depth,depth}` |
| `verify.py` | **실행 전 단일 관문** — 조립된 프롬프트와 semantic view 입력을 함께 단정 (`verify_prompt.py` + `derived_queries.py` 통합) |
| `measure_recipes.py` · `strip_components.py` | 카드별 lift 측정 / 삭제 목록 성분별 측정 |
| `learn_recipes.py` · `recipe_learning.py` | **LLM이 레시피를 제안하고 train-dev가 승인** — 측정·승인 기계와 제안 프롬프트 |
| `select_on_val.py` | 진화 라운드별 설정을 **val에서 재채점·선택** |
| **`legal_evolve.py`** | **진화 루프를 검색 과제로** — 어댑터 · 이음매 3개 · 진단 컨텍스트 |
| `query_rewrite.py` | LLM 쿼리 재작성 (언어별 프롬프트, 디스크 캐싱) |
| ~~`llm_taxonomy.py`~~ | LLM 분류 실험. **중국어에서 규칙이 우월해 미채택이고, 2026-11-09에 파일을 삭제했다** — 결론은 여기 남기고 코드는 지웠다 |
| ~~`random_arm.py`~~ | 진단 LLM 대신 무작위 교란. **논문 Table 6이 이미 쟀다(−9.63 F1)** 고 §28이 보류 → 2026-11-12 삭제 |
| ~~`legal_recipes.py`~~ | 24장 "사다리" 풀. **24장 중 16장이 스칼라 크기를 카드에 박아** §33이 지운 전제였다 → 2026-11-12 삭제 |

EvolveMem 레포는 **포크하지 않는다.** 진화 엔진이 튜닝할 로직이 그 안에 있으므로,
법률에서 깨지는 지점만 이음매에서 교체한다. 레포 자체 수정은 최소한이며
(`doc_id` 필드 2줄, 차원 3개, 진단 프롬프트 주입 setter) **모두 기본값 불변**이라 LoCoMo
동작은 그대로다.

## 빠른 시작

```bash
V=/home/yoosun/legal/.venv-evolve/bin/python
cd /home/yoosun/legal/evolve_legal

$V build_cases.py --all                         # 1. 코퍼스 변환
$V classify_queries.py --all                    # 2. 분류 + 공식 fold 주입
HF_HOME=/home/yoosun/.cache/hf \
  $V build_embeddings.py --dataset LeCaRDv2 --queries --device cuda:2   # 3. 임베딩

$V verify.py                                    # 4. 실행 전 관문 (필수)
$V evaluate.py --dataset LeCaRDv2 --split train --out runs/LeCaRDv2_bm25.json
$V legal_evolve.py --dataset LeCaRDv2 --split train --rounds 7 \
    --recipes hand --model qwen
$V select_on_val.py --dataset LeCaRDv2 --dir evolution_official --write  # val 기준 선택
```

> **test fold는 `in_split()`이 막는다.** `--split test`·`all`·`None`은 `ALLOW_TEST=1`
> 없이는 `RuntimeError`다. 문장으로 막으려 했으나 실패했고(§8의 네 데이터셋 표가
> test에서 나왔다), 이제 코드가 막는다. `evaluate.py`의 `--split` 기본값도
> `test` → `train`으로 바꿨다.

진화 루프는 진단 LLM 서버가 필요하다. 전체 명령과 vLLM 기동은 [PROGRESS.md §8](PROGRESS.md) 참고.

> **서버 주의** — 우리 vLLM은 **GPU 1 / 포트 8000**이다. **포트 8002는 다른 사용자의 서버**이므로
> 요청을 보내지 말 것. 정리할 때 `pkill -f "vllm serve"`는 남의 프로세스까지 잡으므로 PID를 지정한다.

## 평가 규약

- **NDCG@30**을 `multilingual_bm25_recall.compute_query_metrics`에서 **그대로 import**한다
  (재구현하지 않음). 기존 BM25 결과와 같은 열에 놓기 위해서다.
- 모든 실행이 run 파일(`{query_id: [doc_id,...]}`)을 남긴다 → 재검색 없이 재채점 가능.
- **train / val / test 3단.** 진화는 train에서만, 확인·수정은 val에서, **test는 마지막 한 번**.
  네 벤치마크의 **공식 split**을 쓴다(`result/{ds}/query_structured.json`).
- 네 데이터셋 **동일 전처리**: `Fact`만 색인, `Reason`+`Result`를 모두 가진 문서만.
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

아래는 **train fold** 수치다. 괄호 안은 그 데이터셋의 **noise floor**(측정된, 구분 가능한
최소 Δ)이고, floor를 못 넘는 숫자는 "1위처럼 보이는 자리"까지만 말한다.

1. **최소 개입만 작동한다.** θ₀은 BM25 단독이고, 8가지 view 조합 중 **BM25+semantic 둘만**
   켜는 것이 네 데이터셋 전부에서 1위다 (LeCaRDv2 +0.0140 / LeCaRD +0.0134 /
   COLIEE +0.0041 / LEGAR +0.0062). metadata view는 네 데이터셋 **전부 음수**다.
   판례는 모든 정보가 본문에 있어 메타데이터가 BM25와 중복된다 — 논문의 multi-view 전제가
   이 도메인에서 성립하지 않는다.
2. **COLIEE와 LeCaRD에서는 "어떤 view를 켤지"가 측정 불가다.** floor가 각각 0.0052 /
   0.0240이고 어느 설정도 넘지 못한다. LeCaRD는 train 79건이라 원리적 한계다(원데이터).
   거기서 루프가 무엇을 찾아도 성과로 읽을 수 없다.
3. **최적 설정은 데이터셋마다 다르다.** `weight_semantic` 최적값만 0.1~2.0으로 **20배**
   벌어진다. 전역 기본값을 박는 대신 **데이터셋별로 엔진이 찾게 하는 것**이 이
   프레임워크의 값이다. 격자는 채점용 정답지일 뿐이다.
4. **이득의 귀속.** 사람이 내린 tier ① 결정 하나(색인 절 + 임베딩 길이)가 COLIEE θ₀를
   **+31%**, LEGAR를 **+42%** 올렸다. 루프가 찾은 최대는 **+1.9%**다. 한 자리 차이이고,
   효과가 가장 큰 축(문서 변환)이 사람 손에 있다 — 이 연구의 가장 약한 지점이고 그대로
   보고한다.
5. **벤치마크가 과제 정의와 어긋나는 지점을 쟀다.** 정답 문서가 쿼리의 당사자명을
   **+16pp**, 지명을 **+51pp** 더 공유한다. 같은 구에서 난 두 사건은 유사 사건이 아닌데
   지표는 그것을 보상한다. 그래서 `STRIP_SCOPE` 기본값을 점수가 아니라 **과제 정의**에
   맞춰 인명+사명+지명으로 두고, 두 숫자를 함께 보고한다.
6. **레시피라는 장치가 비어 있다.** 손으로 쓴 5장 중 **4장이 action space 필드 하나짜리**
   제안으로, 진단 LLM이 카드 없이 직접 쓸 수 있다. 행동 공간이 작고(12개) 필드가 대체로
   독립적이면 레시피의 역할이 없다 — 그래서 **LLM이 직접 제안하는 쪽**(`learn_recipes.py`)
   으로 간다. 묶음이라는 장치 자체는 `per_category_overrides`로 action space 안에 이미
   있고, 그쪽이 실제로 더 많이 움직였다(val +0.0079 대 전역 +0.0012).

근거와 반례는 [METHODOLOGY.md §8](METHODOLOGY.md)에 있다. **단 §8의 네 데이터셋 표는
test fold 산물이므로 교체 대기 중이다** — 그 표의 결론("⑤가 1위")은 train에서 부호가
뒤집힌다(⑤는 전부 최하위권).

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
([METHODOLOGY.md §12·§13](METHODOLOGY.md)).

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

