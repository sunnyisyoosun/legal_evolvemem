# EvolveMem for Legal Case Retrieval — 진행 상황

마지막 업데이트: 2026-10-30

논문 [EVOLVEMEM](https://github.com/aiming-lab/SimpleMem)의 자가진화 검색 구조를
법률 판례 검색(LeCaRD / LeCaRDv2 / COLIEE / LEGAR)에 적용하는 작업 기록.

---

## 이 문서는 무엇인가

**지금 무엇이 돌고 있고 다음에 무엇을 하는지**만 적는다. 시스템 구조는
[METHODOLOGY.md](METHODOLOGY.md), 날짜순 발견과 실수는 [JOURNAL.md](JOURNAL.md)에 있다.

### 지금 막힌 곳 (2026-11-11)

**진단 LLM 가중치가 디스크에 없다.** `Qwen3-30B-A3B-Instruct-2507`의 safetensors가
사라지고 설정·토크나이저 6MB만 남았다(`/home/yoosun/.cache/hf/hub/`). 그래서
**데이터셋별 진화 실행이 보류**이고, 선택은 사용자 판단 대기다.

| 선택지 | 대가 |
|---|---|
| (a) 60GB 재다운로드 | 루트 파일시스템이 **94% 사용(여유 225GB)**. 기존 모든 실행과 동일 모델이라 §17·§26 등과 비교 가능 |
| (b) 공유 캐시의 `Thinking-2507` (57GB, 추가 디스크 0) | **다른 모델.** thinking 계열이라 `<think>`를 내고, §7의 reasoning-parser 함정이 반대로 작용한다. 기존 수치와 비교가 끊긴다 |
| (c) OpenRouter (`config.py`에 설정돼 있음) | 판결문 일부가 외부로 나가고 라운드당 과금. 모델도 달라져 비교가 끊긴다 |

**공유 장비 상태**: GPU 0·1은 `yeop`의 vLLM 둘이 점유(84GB·86GB), GPU 2는 거의
비어 있음, 포트 8000도 다른 사용자 것. 우리 서버는 **여유 GPU + 별도 포트**에,
인덱스가 아니라 **UUID로 고정**해서 올려야 한다(torch와 nvidia-smi의 번호가 다르다).

### 끝난 것 (2026-11-10~11)

| | 결과 |
|---|---|
| 색인 텍스트 `Fact` → `Fact+Reason+Result` | θ₀가 중국어 둘은 하락(−0.014/−0.005), 비중국어 둘은 상승(**COLIEE +0.057**, LEGAR +0.015) |
| 임베딩 `--max-len` 512 → **8192** | 512는 결정이 아니라 **모델 비교용으로 묶어 둔 값의 잔존물**이었다. 네 데이터셋 재임베딩 + 지문 가드 `ALL_OK` |
| 융합 스윕 val + train 완주 (`sweep.py --grid fusion`) | **§8의 결론 넷이 뒤집혔다** — dense 뷰가 최대 레버, swap은 죽음, 구조화 뷰는 음수, 융합 모드는 실질 무관 |
| `measure_recipes` 네 데이터셋 | 32칸 중 주장 가능 10칸. 어느 카드도 **4개 중 2개를 넘기지 못한다** |
| 승인 게이트 수정 | `after > θ₀`와 효과크기 바닥이 **빠져 있었다** — LeCaRD에서 +15.65 pp를 주장한 카드의 종점이 θ₀보다 낮았다 |
| 카드 묶음 해체 | 7장 → **9장, 카드당 메커니즘 하나.** 상수 제거, `fusion_mode`·`weight_semantic` 분리 |
| `weight_semantic` 정답지 | 최적값이 **COLIEE 0.1 / LeCaRD 0.2 / LeCaRDv2 0.3 / LEGAR 2.0 — 20배 차이** |
| 비용 계측 | `legal_retriever.budget()` — 격자와 루프를 **같은 단위**로 비교 |
| fold 회귀 복원 | 재빌드가 `fold`·`legal_class`를 **조용히 날렸다**(폐기된 라벨을 요구하는 승계 가드). 네 데이터셋 §5 표와 일치 확인 |
| 죄명 추출 정제 | 노이즈 4종 제거(`비추어죄` 6.3%, `성범죄` 24%, `的犯罪` 9.4% 등), 복합·법률위반형 보존. **코퍼스 미반영** |

### 다음 순서 (의존 관계 때문에 순서가 고정이다)

1. **진단 LLM 결정** (위 (a)/(b)/(c)) — 이것 없이는 2~4를 못 한다
   - 프롬프트는 이미 데이터 언어별로 갈라 두었다(JOURNAL §29). 모델이 정해지면 추가 작업 없이 네 데이터셋이 각자의 언어로 돈다.
   - 측정할 것: 언어 일치 프롬프트 vs 영어 프롬프트. `diagnosis_prompts.install()` 대신 `set_prompt_lang("en")`을 강제하면 같은 코드로 영어 arm이 되므로, 비교는 한 줄 차이다.
   - **주의**: 영어 프롬프트도 새로 쓴 것이어서 COLIEE 숫자도 움직인다. 2026-11-11 이전 기준선과의 차이는 언어가 아니라 '우리 프롬프트 vs 필터링된 LoCoMo 프롬프트'를 재는 것이다.
2. **데이터셋별 진화 4회** — 상수 없는 카드, 비용 계측 포함.
   채점: 정답지의 봉우리에 **몇 번의 평가로 얼마나 근접**하는가
3. **카테고리별 진화** — §6의 요구사항. 데이터셋 간 적응의 축소판
4. 엔진 ablation: 카드 상수 유/무, step-cap 2/5, **진단 정보 제거·셔플**
   (LLM이 데이터를 읽는지 사전지식만 쓰는지 — 한 번도 재지 않았다), CoT on/off
5. 죄명 수정 반영: 재빌드(3분) → **재임베딩(57분)** → 같은 스윕 재측정 →
   지금 기준선과 비교. 순서를 지켜야 효과가 분리된다
6. `recipe_learning.py` → `legal_recipes.py` 병합
7. **test fold 측정** — 마지막 단계. 아직 한 번도 보지 않았다

보류: 무작위/탐욕 arm(`random_arm.py` 구현만 존재). LLM 엔진은 **전제**이고 검증
대상이 아니다 — JOURNAL §28.

---

## 0. 현재 위치 요약

**완료**: 데이터 변환 · 검색 래퍼 · 평가 하네스 · 쿼리 분류(Axis A) · 다국어 semantic 뷰 ·
entity-swap · 추출 품질 개선 · 쿼리 재작성 · **법률 레시피 8장 투입** ·
**RRF 네 데이터셋 비교** · 레시피 결함 3건 수정

**핵심 수치** (LeCaRDv2, NDCG@30):

| | train (640) | test (160) |
|---|---|---|
| θ₀ (BM25 only) | 0.4040 | 0.3948 |
| 수동 스윕 최고 | 0.4155 | 0.4057 (+0.0110) |
| **진화 R2** | **0.4184** | **0.4101 (+0.0154)** |

**R1 정체는 해결됐다.** 원인은 레시피 풀이 비어 있어 의존하는 설정들을 한 번에 넘길 수단이
없었던 것이고(§6), 8장을 넣자 R0에서 발동 → R1·R2 연속 ACCEPT → **진화가 수동 튜닝을
처음 이겼다**.

**진행 중**: 레시피 3건 수정 후 LeCaRDv2 7라운드 재실행, 이어서 LEGAR train
(§2026-10-26 결과대로 엔진이 swap 대신 semantic 쪽을 고르는지 확인).

> **주의** — 아래 §6·§7은 **레시피 투입 전** 기록이다. 당시 진단으로 읽을 것.
> 현재 상태와 방법론은 [METHODOLOGY.md](METHODOLOGY.md) §7·§9·§10에 있다.

## 1. 데이터

`/home/yoosun/legal/output/` 의 4개 벤치마크. 스키마는 모두 동일
(문서 `doc_id/Fact/Reason/Result`, 쿼리 `Query_id/Q/Crime/split`).

| 데이터셋 | 언어 | 문서 | 쿼리(test) | 공식 과제 | rel |
|---|---|---|---|---|---|
| LeCaRD | zh | 9,088 | 107 (20) | 쿼리당 후보 100개 재랭킹 → **전체 코퍼스로 통일** | 1/2/3 |
| LeCaRDv2 | zh | 55,192 | 800 (160) | 전체 코퍼스 검색 | 1/2/3 |
| COLIEE | en | 5,452 | 1,678 (398) | 전체 코퍼스 검색 | binary |
| LEGAR | ko | 14,525 | 5,960 (1,161) | 전체 코퍼스 검색 | binary |

### 확인 과정에서 바로잡은 것

- **LeCaRD만 pool 재랭킹.** 두 LeCaRD 모두 `candidate_pool.json`이 있고 ceiling recall이 1.0이지만,
  LeCaRDv2의 것은 **annotation pool**(판정이 pool 안에서만 매겨짐)이라 평가용이 아니다.
  ceiling recall 1.0은 pooling 데이터셋이면 구성상 자동으로 나오는 값이므로 근거가 되지 못한다.
  v2는 `candidate/`에 55,192개 파일이 있고 그것이 공식 후보 집합이다.
  쿼리당 pool로 제한하면 **라벨 누출**이다.
- **두 pool 파일 모두 공식 데이터의 포맷 변환본**이다 (내용 일치율 1.0000).
  LeCaRD는 `candidates1`(50) + `similar_case/candidates2`(57) = 107.
- **`LANG_DEFAULTS`의 `doc_field`를 믿으면 안 된다.** 설정에는 전부 `"Fact"`로 적혀 있으나,
  발표 baseline이 측정된 캐시 인덱스는 COLIEE·LEGAR가 `Fact+Reason+Result`였다
  (캐시 avgdl COLIEE 1929.7 ≈ all-fields 1883.1, LEGAR 314.1 ≈ 308.1).
  **현재 프로젝트는 4개 데이터셋 모두 `Fact`로 통일**했다. 프레임워크 비교에서 입력 텍스트가
  데이터셋마다 다른 교란을 없애기 위함이며, 대신 발표 수치와 직접 비교는 불가능하다
  (비교하려면 기존 방법들도 Fact-only로 재측정 필요).
- **LEGAR는 공식 정답의 83%가 코퍼스 밖에 있다.** 네 데이터셋 공통으로 `Reason`과 `Result`가
  모두 있는 문서만 남겼는데(`filter_reason_result.py`), 한국 판결문은 `Result` 채움률이
  **14.2%**다(`Reason`은 99.3%). 102,606건 → 14,525건. 다른 셋은 `Result`가 92% 이상이라
  손실이 작다. 공통 전처리를 유지하기로 했으므로, **LEGAR 수치는 "네 데이터셋 동일 전처리 하의
  비교"로만 해석해야 하고 공식 LEGAR BENCH 결과와 나란히 놓을 수 없다.**
- **LEGAR의 관련성은 공식 `d1/d2/d3` 일치로 정의된다.** `Stricter_version/*.json`의
  `question_id`/`evidence_id`가 쿼리와 정답이고, 같은 factor option을 공유하는 사건이 유사군이다
  (HF 카드: "102 factors, 443 options"). 정답 문서가 쿼리의 공식 죄명(d2)을 포함할 확률
  **61.6%** vs 무작위 4.5%. → d3를 문서에서 추출하는 것은 정답 기준의 역공학이므로 접었다.

---

## 2. 평가 기준

- **주 지표: NDCG@30.** `multilingual_bm25_recall.compute_query_metrics`를 그대로 import해서 사용
  (재구현하지 않음). 기존 BM25 결과와 같은 열에 놓이도록.
- 모든 실행은 `runs/{dataset}_*.json`에 run 파일(`{query_id: [doc_id,...]}`)을 남긴다.
  나중에 어떤 방법이든 재검색 없이 동일 기준으로 재채점할 수 있다.

### pool 마스킹은 쓰지 않는다 (2026-10-29 결정)

LeCaRD는 쿼리당 후보 100개를 주고 그것을 재랭킹하는 것이 공식 과제지만, **나머지 셋은 전체
코퍼스 검색**이다. 한쪽만 pool을 쓰면 서로 다른 과제의 숫자를 같은 표에 올리게 된다.
**네 데이터셋 모두 전체 코퍼스로 통일**한다.

`evaluate.py` · `sweep.py` · `legal_evolve.py` 모두 **`--use-pool` 기본 off**이고, 켜면
LeCaRD 공식 프로토콜을 재현한다(참고용).

### θ₀ (Fact-only, BM25-only, test split, **전체 코퍼스**)

| 데이터셋 | 코퍼스 | NDCG@10 | **NDCG@30** | R@30 |
|---|---|---|---|---|
| LeCaRD | 9,088 | 0.5912 | **0.6427** | 0.6876 |
| LeCaRDv2 | 55,192 | 0.4623 | **0.3948** | 0.3728 |
| COLIEE | 5,452 | 0.1678 | **0.2000** | 0.3066 |
| LEGAR | 14,525 | 0.0250 | **0.0371** | 0.0897 |

참고 — LeCaRD를 공식 subpool(100)로 재랭킹하면 NDCG@10 0.6514 / **NDCG@30 0.8254** /
R@30 0.9772다. 과제가 쉬워져 수치가 크게 오르므로 **다른 데이터셋과 나란히 놓을 수 없다.**

래퍼는 공식 조건(COLIEE/LEGAR는 all-fields)에서 기존 baseline을 **소수점까지 재현**했다:
LeCaRD 0.8254 / 0.6427, COLIEE 0.2531, LEGAR 0.0598.

### 기존 `legal/` 평가 코드에 대한 주의

`ranknet_metrics.py`와 `multilingual_bm25_recall.py`는 **같은 이름의 지표를 다르게 계산**한다.
신경망 방법(BERT-aux / RankNet / Joint)은 전자를, BM25는 후자를 쓴다.

- `ranknet_metrics.ndcg_official`은 gain이 `2**rel`이다(docstring은 "선형 gain"이라 적혀 있어 **코드와 불일치**).
  비관련 문서가 gain 1을 받으므로, **정답을 하나도 못 맞춘 랭킹도 COLIEE NDCG@30 = 0.7815**가 나온다.
  LeCaRD(rel 1/2/3)에서는 영향이 작지만 COLIEE·LEGAR(binary)에서는 해석이 달라진다.
- `recall_at_k`의 분모가 전체 정답 수가 아니라 예측 상위 30개 내 정답 수라, **Recall@30은 항상 1.0**이다.
- 신경망 방법은 **BM25 top-30 재랭킹**(`structured_candidates.json`)이라 후보 집합 자체가 다르다.

→ 한 표에 올리려면 체크포인트로 재추론해 run 파일을 만들고 동일 평가기로 재채점해야 한다. (미착수)

---

## 3. 구현

모두 `/home/yoosun/legal/evolve_legal/`. EvolveMem 레포는 수정하지 않고 import만 한다.

| 파일 | 역할 |
|---|---|
| `legal_meta.py` | 규칙 기반 메타데이터 추출(법조문·죄명·당사자·법원·날짜), 한자 숫자 정규화, `Crime` 필드 해석, `DOC_FIELD_OF` |
| `build_cases.py` | 판례/쿼리 → 검색 단위 dict (`--dataset` / `--all`). *당시 이름 `build_memories.py`* |
| `legal_retriever.py` | `LegalIndex`(토크나이저 주입·pool 마스킹·doc_id 역매핑), `build_index`, `search` |
| `legal_taxonomy.py` | 2축 분류(형법 장 / BM25 난이도), 합성 카테고리 |
| `classify_queries.py` | 쿼리에 분류 라벨 부착, 난이도 경계는 train에서만 적합 |
| `build_embeddings.py` | 다국어 임베딩 선계산 (`--model`, `--tag`) |
| `evaluate.py` | 단일 설정 평가 + run 파일 덤프 |
| `sweep.py` | 설정 비교, 클래스별 최적 탐색, `--fit-apply` 일반화 검증 |
| `legal_evolve.py` | EvolveMem 진화 루프를 검색 과제로 — 어댑터·이음매·진단 컨텍스트 |
| `query_rewrite.py` | LLM 쿼리 재작성(언어별 프롬프트) + 확장/RRF 비교 |
| ~~`llm_taxonomy.py`~~ | LLM 분류 실험. **중국어에서 규칙이 우월해 미채택이고, 2026-11-09에 파일을 삭제했다** — 결론은 여기 남기고 코드는 지웠다 |

### 설계 결정

1. **판례 1건 = 메모리 1개, LLM 추출 생략.** EvolveMem의 추출기는 대화를 사실로 증류하는 용도인데
   판결문은 이미 검색 단위이고, 55k 문서에 LLM을 돌리면 비용이 전체를 지배한다. 변환은 10~15초.
2. **`retrieve_multiview`를 포크하지 않는다.** 진화 엔진이 튜닝할 fusion·time-decay·per-category override
   로직이 그 안에 있다. 법률에서 깨지는 지점만 각자의 이음매에서 교체:
   - BM25 토크나이저 (`content.lower().split()` → jieba/mecab/spaCy, 기존 baseline과 동일)
   - 쿼리 분석 (`analyze_query`는 대문자로 엔티티 탐지 — 영어 전용)
   - pool 마스킹 (`index.set_pool()` 하나로 세 뷰 전부)
   - 구조화 뷰 (persons∩entities 교집합 + 상수 1.0 점수 → 일치 개수 점수, persons 기본 비활성)
3. **쿼리도 문서와 같은 추출기로 변환.** 이 벤치마크들은 쿼리도 사실관계 서술이라 양쪽이 어휘를 공유한다.
4. **임베딩은 디스크 선계산.** 진화 루프가 매 라운드 재인코딩하지 않도록.
   임베딩 텍스트에서 **당사자 이름 제외** — 판례에서 피고인 이름은 사건 식별 정보가 아니다.

### 해결한 버그/함정

- `rank_bm25`가 EvolveMem `requirements.txt`에 누락. 없으면 경고 한 줄 찍고 **조용히 BM25를 끈다**.
- 중국어 인명 정규식 `{2,4}` 탐욕 매칭이 동사까지 포획 (`莫新国酒后驾驶` → `莫新国酒`). entity-swap에 치명적.
- 중국어 죄명 비탐욕 매칭이 복합 죄명을 절단 (`掩饰、隐瞒犯罪` + `罪所得收益罪`).
  탐욕 매칭으로 바꾸니 변별력 lift가 1.0% → 33.9%로 상승.
- `build_cases.py`(당시 `build_memories.py`)가 쿼리 파일을 덮어써서 **분류 라벨이 지워진다**. `_carry_labels()`로 이전
  빌드의 라벨을 승계하도록 고쳤다(세 번 겪은 뒤).
- 한국어 법조문 정규식 `[가-힣]{2,25}?법`이 법 앞에 2글자를 요구해 **`형법`이 원천적으로 매칭
  불가**였다. 한국 형사판결에서 가장 많이 인용되는 법이다. LEGAR 문서 법조문 45% → 100%.
- 영어 `s. 56 of the Trade-marks Act` 형태 미지원(축약형·하이픈 법명). COLIEE 79.8% → 87.3%.
- LeCaRDv2 문서의 81%는 조문을 아예 인용하지 않는다(정규식 문제가 아님). 죄명→조문 학습 매핑으로
  35,237건 보간해 36% → 99.8%. **검색 성능에는 영향 없었다.**
- `RetrievedMemory`에 식별자가 없어 content 역매핑이 필요했다(COLIEE는 중복 content 24%).
  레포에 `doc_id` 필드를 추가해(2줄, LoCoMo는 `None`) 역매핑 코드를 전부 제거했다.
- `Crime` 필드가 리스트가 아니라 **문자열**인 경우가 있어 `[0]`이 첫 글자를 집는다. 두 번 겪었다.
- `Crime` 필드 의미가 데이터셋마다 다름: LeCaRD=죄명, **LeCaRDv2=형법 조문 번호**,
  LEGAR=죄명+쟁점 속성, COLIEE=광역 분류. v2는 문서가 한자(`第二百六十三条`), 쿼리가 아라비아 숫자라
  정규화 전 매칭률 0% → 정규화 후 800/800.

---

## 4. 쿼리 분류 (Axis A 단독)

per-category 하이퍼파라미터를 적용하려면 클래스당 쿼리가 충분해야 한다. 죄명(LeCaRD 57종/107쿼리)이나
조문 조합(v2 160종/800쿼리)은 너무 잘아서, 각 벤치마크가 **스스로 정의한 분류 체계**로 접는다.

| 데이터셋 | 클래스 | 근거 | 미분류 |
|---|---|---|---|
| LeCaRDv2 | 9 | 중국 형법 분칙 장 (조문 번호에서 결정적 유도) | 2.1% |
| LeCaRD | 7 | 동일. 죄명→조문 매핑은 코퍼스에서 학습 (검증 9/9) | 0.9% |
| LEGAR | 8 | **벤치마크 공식 d1** (`Crime` 필드의 첫 항목) | 0% |
| COLIEE | 7 | **법 분야** (난민이민·지식재산·조세·노동연금·형사·행정) | 13.7% |

**LEGAR** — 벤치마크가 `d1 / d2 / d3`(상위 카테고리 / 죄명 / 쟁점 속성) 3단계 체계를 제공하고
쿼리 `Crime` 필드에 그대로 들어 있다. 직접 만든 9개 그룹은 가장 큰 카테고리인 성범죄(45%)를
미분류로 떨어뜨렸다. 공식 체계를 쓴다.

**COLIEE** — 벤치마크의 `Crime` 태그는 형사 죄명(`fraud`, `assault`)인데 실제 데이터는 캐나다
연방법원의 난민·이민(47%)·지식재산·조세 사건이라 79%가 `civil/other`였다. 본문에서 법 분야를
읽는 방식으로 교체. 정답 문서가 쿼리와 같은 분야일 확률 63.0% vs 무작위 32.5% (lift 1.94x).

### 폐기한 Axis B (BM25 점수 첨예도)

BM25 상위 점수의 낙차 `gap = (s[0]-s[9])/s[0]`로 난이도를 3분할했으나 폐기했다.

- `gap` ↔ NDCG@30 상관이 **+0.147**로 약하다. 질의 시점에 알 수 없는 **정답 개수**가 +0.353으로 더 강하다
- 난이도를 (약하게) 예측하지만 **"다른 설정이 필요하다"는 예측은 못 한다**. entity-swap처럼 전역적으로
  좋은 레버가 생기자 per-class 이득이 **+0.0072 → 0.0000**으로 소멸했다
- `gap`은 BM25 자신의 점수 분포인데 그것으로 "BM25에 무엇을 섞을지"를 정하는 구조였다. 평탄한 분포는
  "어휘로 안 갈린다"는 뜻이지 "의미 매칭이 도움 된다"는 뜻이 아니다
- 전체 코퍼스(55k)와 pool(100)에서 같은 지표가 다른 것을 의미해 데이터셋 간 비교가 성립하지 않았다

부수 효과로 분류가 즉시 끝난다(이전엔 쿼리마다 전체 BM25를 돌려 COLIEE 448초).

**한계**: LeCaRD는 test 20건에 클래스 7개라 per-category override를 쓸 수 없다. 전역 설정만 쓴다.

## 5. 실험 결과 (LeCaRDv2 train, NDCG@30)

| 레버 | 효과 | 판정 |
|---|---|---|
| **쿼리 재작성 — 확장 x1** | **+0.0127** | 성공 (200건 표본) |
| **entity-swap (swap30)** | **+0.0114** | 성공 |
| per-category override (test) | +0.0027 → 0 | 레버가 늘자 소멸 |
| semantic 뷰 (ws_sem0.3) | +0.0007 | 미미 |
| 죄명 쿼리 확장 (LEGAR) | +0.0008 | 미미 |
| 추출 품질 개선 (조문 36%→99.8%) | 0.0000 | 무효 |
| 조문·죄명 가산점 | 가중치↑ → 단조 감소 | 실패 |
| 장(章) 하드 필터 | −0.0832 | 실패 |
| 쿼리 재작성 — 대체 | −0.1779 | 실패 |
| d3 쟁점 속성 학습 | 학습 불가 | 실패 |

**누적 최고 0.4155** (`ws_sem0.3_swap30`) vs θ₀ **0.4040**.

### 관통하는 패턴 — 최소 개입만 작동한다

| 성질 | 레버 | 결과 |
|---|---|---|
| BM25에 **구조화 신호를 더함** | 조문·죄명 (가산점/필터/확장/추출개선/속성학습) | 전부 0 또는 음수 |
| BM25 입력을 **대체** | 쿼리 재작성 단독 | −0.178 |
| BM25 입력에 **최소 개입** | entity-swap(인명만 제거), 재작성 확장 x1 | **+0.011 ~ +0.013** |

그리고 **가중치를 올리면 항상 나빠진다**: semantic 0.3→1.0, 구조화 0.1→0.8, 재작성 x1→x4,
swap30→swap100. 판례 검색에서 BM25의 원문 매칭이 지배적이고 보조 신호는 얇게 얹힐 때만 기여한다.

논문의 multi-view 설계는 **대화 메모리**(구조화 신호가 본문에 없음)를 전제한다. **판례**는 모든
정보가 본문에 있어 메타데이터가 BM25와 중복된다 — 변별력(lift 11.4x, 정답 일치율 78%)은 있지만
BM25가 이미 상위에 올린 문서를 재확인할 뿐 새 문서를 데려오지 않는다.

### semantic 뷰

bge-m3로 4개 데이터셋 84,257건 인코딩(약 8분, 182MB), 디스크 선계산.
**판례 검색에서 semantic은 보조 신호다** — 단독으로는 BM25에 한참 못 미치고
(COLIEE 0.1700 vs 0.2000, LeCaRDv2 0.2814 vs 0.4040) 가중치 0.3~1.0에서만 도움이 된다.
논문의 LoCoMo 결과(semantic이 두 번째로 큰 기여, −10.32 F1)와 대비된다.

**COLIEE의 토큰 길이**: 문서 평균 8,383자, 쿼리 중앙값 9,634자. 512 토큰 잘림이 semantic
성능의 63%를 깎고 있었다(sem_only 0.1042 → 2048에서 0.1700). `max_seq_length`는
문서 길이에 종속된 데이터셋별 차원이다.

### 임베딩 모델 비교 (LeCaRDv2 train, max_len 512 고정)

> **2026-11-10**: `max_len 512 고정`은 세 후보를 공정하게 비교하기 위한 조치였는데,
> 비교가 끝난 뒤에도 512로 계속 돌렸다. 셋 중 **bge-m3만 8192를 읽는다**(e5-large 512,
> embeddinggemma 2048) — 고른 이유였던 능력을 꺼 둔 채 모든 실행을 했다. 기본값을
> 8192로 바꿨고, 아래 표의 숫자는 **512에서 측정한 것**이다. 모델 순위 자체는 셋 다
> 512였으므로 유효하지만, bge-m3의 우위는 8192에서 더 커질 수 있다.

논문은 `BAAI/bge-base-en-v1.5`(영어 전용)를 쓴다. BGE 계열의 다국어 모델이 `bge-m3`다.

| 설정 | **bge-m3** | multilingual-e5-large | embeddinggemma-2 |
|---|---|---|---|
| ws_sem0.3 | **0.4047** | 0.3931 | 0.3848 |
| rrf_hybrid | **0.3926** | 0.3763 | 0.3573 |
| **ws_sem0.3_swap30** | **0.4154** | 0.4054 | 0.4042 |
| 파라미터 / 차원 | 568M / 1024 | 560M / 1024 | 270M / 768 |
| 55k 인코딩 | 292s | 285s | 129s |

**bge-m3 유지.** semantic이 들어가는 모든 설정에서 1위다. 체급 차이가 아니다 — e5-large는
560M/1024/285초로 사실상 동일한 체급인데도 진다. bge-m3는 BAAI가 만들어 중국어 비중이 높다.
**미검증**: 비교는 중국어에서만 했다. 영어·한국어에서 순위가 뒤집힐 수 있다.

### entity-swap

EvolveMem은 이 기능을 `category == 5`(LoCoMo 적대적 split)에 하드코딩해 둔다. 판례에서 당사자
이름은 사건 유사성과 무관하므로 모든 쿼리에 유효하다. `search()`에서 per-class override를 직접
해석하고 `category=5`를 항상 넘겨 게이팅을 풀었다(5줄, 로직 복제 없음).

**조건부 레버다**: `bm25_swap30` = 정확히 +0.0000. `keyword_only`는 `first_found` 융합이라
swap 결과가 뒤에 붙기만 한다 — **점수 융합(`weighted_sum`)과 짝지어야 작동**한다.
`swap30`(0.4154) > `swap100`(0.4081): 이름을 뺀 질의는 정밀도가 낮아 얕게 가져와야 한다.

### 쿼리 재작성 (LLM, Qwen3-30B)

사실관계 서사를 법률 용어 질의로 재작성(388자 → 74자). 언어별 프롬프트(zh/ko/en).

| 조건 | NDCG@30 | 대비 |
|---|---|---|
| **확장 x1** (원쿼리 + 재작성) | **0.4519** | **+0.0127** |
| 확장 x2 | 0.4394 | +0.0002 |
| 원쿼리 | 0.4392 | — |
| 확장 x4 | 0.4080 | −0.0312 |
| 재작성 단독 | 0.2613 | −0.1779 |

재작성 품질 자체는 좋다(인명·날짜·지명을 지우고 `预谋抢劫`·`犯罪未遂`·`如实供述`만 남김).
**대체하면 −0.178, 한 번 덧붙이면 +0.0127.** 법률 용어는 그 자체로는 특이성이 없지만
구체적 서술 위에 얇게 얹히면 보조 신호가 된다. EvolveMem의 `intent_planning`(멀티홉 분해)은
−0.088로 해로웠다 — 판례 사실관계에는 쪼갤 "홉"이 없다.

## 6. 자가진화 루프 — 4회 실행 결과 (레시피 투입 **전**)

> 이 절은 **해결 전 기록**이다. 아래의 "매 실행 R1에서 멈춘다"는 더 이상 현재 상태가 아니다 —
> 법률 레시피 8장을 넣어 해결됐다(§0, METHODOLOGY §7·§9).

엔진은 QA용이다(검색 → 답변 생성 → token-F1). 검색 과제로 바꾸되 루프 자체
(diagnose → propose → guard → revert)는 건드리지 않았다. `legal_evolve.py`의 세 이음매:

1. `evolution.MultiViewIndex`를 미리 만든 `LegalIndex` 반환 팩토리로 치환
2. `_generate_answer`가 답변 대신 **doc_id를 반환** — 답변 생성을 루프에서 제거해
   LLM 호출이 **라운드당 1회**(진단)로 줄었다
3. 어댑터가 doc_id를 qrels로 채점 → **NDCG@30**

`_verify_answer`도 no-op으로 덮어야 했다. 검증기가 doc_id 목록을 산문으로 재작성해
랭킹을 파괴했고(0.4048 → 0.0002), 프롬프트가 판례 100건을 담아 **256,000 토큰**을 요청했다.

### 궤적 (LeCaRDv2 train, 매 실행 동일)

| 라운드 | NDCG@30 |
|---|---|
| R0 | 0.4040 |
| **R1** | **0.4048** (SOFT-ACCEPT) ← 매번 여기서 멈춤 |
| R2~R6 | 정체 또는 하락 후 revert |

4B/30B, 메타데이터 주입 전/후 모두 동일. **진화 0.4048 vs 수동 0.4155 — 7%만 따라왔다.**

### 정체의 원인 (구조적)

1. 거부되면 설정이 R1 상태로 되돌아간다
2. 진단 LLM은 **temperature=0**(레포가 재현성을 위해 하드코딩)
3. 입력이 거의 동일 → **출력도 동일** → 같은 제안 → 또 거부

4B는 `enable_answer_verification`을, 30B는 `enable_intent_planning`(−0.088)을 무한 반복했다.
`attempt_history`는 정상 전달되지만(`Round N: {...} — delta=-0.0874 REJECTED`) 무시된다.

**논문 Eq.4에는 탈출구가 있다** — 두 라운드 연속 정체 시 무작위 섭동(`θ ⊕ η_exp`).
레포의 기본 모드(`elitist=True`)는 이 분기 대신 "거부 → revert → attempt_history"로 대체했고,
코드 주석이 그것을 "the default going forward"라고 설명한다. **레포 구현이 Algorithm 1과 다르다.**

### rubric이 벤치마크 독립적이지 않다

논문 §3.3은 *"The rubric is written in terms of failure patterns rather than specific
benchmarks"*라고 하지만 실제로는:

- `RetrievalConfig`에 **LoCoMo 전용 필드 5개**(`locomo_cat1_single_fact` 등)가 있고
  진단 프롬프트의 TODO 체크리스트가 "매 라운드 하나씩 켜라"고 지시한다
- `diagnosis.py`가 카테고리 번호를 LoCoMo 명칭으로 하드코딩한다
  (`{1:"SingleHop", 5:"Adversarial"}`) — 중국어 판례 검색이 대화 QA로 라벨링돼 전달됐다

**대응 (레포 최소 수정, 기본값은 불변)**: `CATEGORY_NAMES`/`set_category_names()`로 클래스 이름
주입, `PROMPT_EXCLUDE`로 검색 과제에서 무효한 차원(답변 생성 레버, LoCoMo 플래그)을 프롬프트에서
제거, `set_sample_widths()`로 실패 샘플에 **검색된 판례의 죄명·법조문·법원·날짜**를 노출.

이후 진단이 법률 도메인으로 말하기 시작했다 — *"decompose complex legal reasoning into targeted
sub-queries (e.g., 'what crime was committed?', 'was there coercion?')"*. 다만 **레버 선택은
바뀌지 않았고 궤적도 동일**했다.

### 추가한 액션 스페이스 차원

`enable_chapter_filter`(쿼리와 다른 장의 판례를 점수 계산 전에 제외). 논문 §3.3의
self-expanding 주장과 같은 성격. 기본값 `False`로 기존 동작 불변. 실측은 **−0.0832로 해로웠다**.

## 7. 남은 작업

### ~~진화 루프 정체 (미결정)~~ → 해결됨

당시 선택지는 (A) explore 분기 복원 / (B) temperature 상향 / (C) 정체를 결과로 보고 였다.
**셋 다 아니었다.** 실제 원인은 레시피 풀이 비어 있던 것이고, (A)의 전제("레포가 Eq.4를
빠뜨렸다")도 틀렸다 — explore는 벤치마크별 레시피로 구현되어 있고 법률 풀만 없었다.

### 쿼리 재작성 검증 (진행 중)

train 640 확대 → test 검증 → LEGAR(ko)·COLIEE(en) 전이. COLIEE는 entity-swap이 거의 발동하지
않던 데이터셋(쿼리 인명 추출률 19.6%)이라, 여기서 작동하면 대안 레버가 된다. 단 COLIEE 쿼리는
중앙값 9,634자인데 `max_chars=2500`으로 잘라 재작성하므로 절단 영향을 따로 봐야 한다.

### Axis A 통제 실험 (미착수)

동일 레버 집합으로 **전역 설정 vs 클래스별 설정**을 네 데이터셋 test에서 비교(`--fit-apply`).
지금까지 per-category 이득을 test에서 잰 것은 한 번(+0.0027)뿐이고 그마저 조건이 바뀌었다.

### 기타

- 기존 신경망 방법(BERT-aux/RankNet/Joint) 재평가 — 체크포인트 재추론 → run 파일 → 동일 평가기
- LEGAR 죄명 추출 26% 노이즈 (본문 정규식). 분류는 공식 d1으로 해결됐으나 메타데이터는 미정리
- 쿼리 재작성을 액션 스페이스 차원으로 추가(`enable_query_rewrite` + 반복 횟수)

## 8. 실행 방법

```bash
V=/home/yoosun/legal/.venv-evolve/bin/python
cd /home/yoosun/legal/evolve_legal

# 1. 메모리 변환 (이전 빌드의 분류 라벨을 승계한다)
$V build_cases.py --all

# 2. 쿼리 분류 (Axis A)
$V classify_queries.py --all

# 3. 임베딩 (HF_HOME 필수, GPU 지정)
HF_HOME=/home/yoosun/.cache/hf $V build_embeddings.py --dataset LeCaRDv2 --queries --device cuda:2

# 4. 단일 설정 평가 + run 파일 (pool 기본 off = 전체 코퍼스)
$V evaluate.py --dataset LeCaRDv2 --out runs/LeCaRDv2_bm25_test.json
$V evaluate.py --dataset LeCaRD --use-pool    # LeCaRD 공식 프로토콜 재현(참고용)

# 5. 설정 비교 / 클래스별 최적 / 일반화 검증
$V sweep.py --dataset LeCaRDv2 --split train --min-n 20
$V sweep.py --dataset LeCaRDv2 --fit-apply

# 6. 진단 LLM 서버 (우리 것: GPU 1, 포트 8000. 포트 8002는 다른 사용자 것 — 쓰지 말 것)
cd /home/yoosun/legal
PATH=$PWD/.venv-vllm/bin:$PATH HF_HOME=/home/yoosun/.cache/hf CUDA_VISIBLE_DEVICES=1 \
  .venv-vllm/bin/vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --port 8000 --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --served-model-name qwen --enforce-eager

# 7. 자가진화 루프 (train에서만, test 보존)
$V legal_evolve.py --dataset LeCaRDv2 --split train --rounds 7 --model qwen

# 8. 쿼리 재작성 (LLM 1회/쿼리, 디스크 캐싱)
$V query_rewrite.py --dataset LeCaRDv2 --split train --limit 640
```

**프로세스 정리 시 주의**: `pkill -f "vllm serve"`는 다른 사용자의 서버까지 잡을 수 있다.
`TaskStop`이나 PID 지정을 쓸 것. 그리고 **`$!`를 서버 PID로 믿지 말 것** — `nohup` 래퍼
PID라서 그것만 죽이면 vLLM 본체가 살아남아 새 서버가 `Address already in use`로 죽고,
계속 옛 서버와 통신하게 된다. `ss -ltnp | grep :<포트>`로 실제 점유 PID를 확인할 것.

**`--reasoning-parser qwen3`을 붙이지 말 것** (2026-10-29 실측). `Qwen3-30B-A3B-Instruct-2507`은
thinking 모델이 아니라서, 파서를 붙이면 답변 전체가 `reasoning` 쪽으로 분류되고
`message.content`가 `None`이 된다. `make_llm_call`이 `content or ""`를 반환하므로 **매 라운드
빈 제안**이 되어 7라운드가 전부 θ₀(0.4040)에 머문다. 증상이 "레시피가 한 번도 발동 안 함
(`invoked_recipe=None`)"이라 레시피 쪽 버그로 오인하기 쉽다.

**GPU/포트는 고정이 아니다.** 기록된 GPU 1 / 포트 8000은 2026-10-29 기준 다른 사용자가
점유 중이었다. 기동 전 `nvidia-smi`와 `ss -ltn`으로 빈 곳을 확인하고, 공유 GPU에서는
`--gpu-memory-utilization`을 남의 사용량을 침범하지 않는 선으로 낮출 것
(그날은 GPU 2 / 포트 8100 / util 0.72 / `--max-model-len 32768`).

디스크: `corpus/` 470MB, `embeddings/` 182MB, `cache/` 681MB(토큰화·재작성·분류 캐시),
`runs/` 1.7MB, `evolution_results/`

---
