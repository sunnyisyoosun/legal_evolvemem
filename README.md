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

## 문서 세 개

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
                            ├ BM25 (언어별 토크나이저: jieba / mecab / spaCy)
                            ├ semantic (bge-m3, 디스크 선계산)
                            ├ metadata (법조문·죄명 일치 개수)
                            └ party-strip (당사자 이름 제거 후 재검색)
```

| 파일 | 역할 |
|---|---|
| `legal_meta.py` | 규칙 기반 메타데이터 추출 (법조문·죄명·당사자·법원·날짜), 한자 숫자 정규화 |
| `build_cases.py` | 판례·쿼리 → 검색 단위 dict (`corpus/{ds}_cases.json`) |
| `legal_retriever.py` | `LegalIndex` — 토크나이저 주입, 쿼리 확장, pool 마스킹(기본 off) |
| `legal_taxonomy.py` · `classify_queries.py` | 벤치마크 공식 체계로 쿼리 분류, **공식 train/val/test fold 주입** |
| `build_embeddings.py` | 다국어 임베딩 선계산 |
| `evaluate.py` · `sweep.py` | 단일 설정 평가 / 설정 비교 · 일반화 검증 |
| `select_on_val.py` | 진화 라운드별 설정을 **val에서 재채점·선택** |
| **`legal_evolve.py`** | **진화 루프를 검색 과제로** — 어댑터 · 이음매 3개 · 진단 컨텍스트 |
| `query_rewrite.py` | LLM 쿼리 재작성 (언어별 프롬프트, 디스크 캐싱) |
| ~~`llm_taxonomy.py`~~ | LLM 분류 실험. **중국어에서 규칙이 우월해 미채택이고, 2026-11-09에 파일을 삭제했다** — 결론은 여기 남기고 코드는 지웠다 |

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

$V evaluate.py --dataset LeCaRDv2 --out runs/LeCaRDv2_bm25_test.json    # θ₀ 확인
$V legal_evolve.py --dataset LeCaRDv2 --split train --rounds 7 --model qwen
$V select_on_val.py --dataset LeCaRDv2 --dir evolution_official --write  # val 기준 선택
```

7번은 진단 LLM 서버가 필요하다. 전체 명령과 vLLM 기동은 [PROGRESS.md §8](PROGRESS.md) 참고.

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

1. **최소 개입만 작동한다.** BM25에 구조화 신호를 더하는 5가지 방식은 전부 0 또는 음수,
   쿼리를 대체하면 −0.170, 인명만 지우거나 재작성을 한 번 덧붙이면 +0.011~0.013.
   판례는 모든 정보가 본문에 있어 메타데이터가 BM25와 중복된다 — 논문의 multi-view 전제가
   이 도메인에서 성립하지 않는다.
2. **단, 융합 방식이 그 그림을 바꾼다.** 위 결론은 `weighted_sum` 조건에서만 성립한다.
   RRF로 바꾸고 뷰를 얕게 넣으면 부호가 뒤집힌다. **이 수정은 진화 엔진이 사람의 결론을
   교정한 사례**다.
3. **최적 설정은 데이터셋마다 다르다.** 네 데이터셋 비교에서 LEGAR만 swap 대신 semantic이
   이긴다. 전역 기본값을 박는 대신 **데이터셋별로 엔진이 찾게 하는 것**이 이 프레임워크의 값이다.

근거와 반례는 [METHODOLOGY.md §8](METHODOLOGY.md)에 있다.

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

