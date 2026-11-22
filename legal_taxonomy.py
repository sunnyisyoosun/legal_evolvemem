"""Query classification along two axes.

Axis A (legal): the chapter of the penal code a query belongs to. Charge names
and article numbers are too granular to fit a per-class configuration -- LeCaRD
has 57 charge names across 107 queries -- so both are folded up into the code's
own chapter structure, which is a real legal taxonomy rather than an invented
clustering, and which leaves tens of queries per class.

Axis B (difficulty, from the lexical score profile) is RETIRED: it predicted
how hard a query is, weakly, without predicting that a different
configuration would serve it better, and the per-class gain fell to zero once
a globally good lever existed (METHODOLOGY 6). Its scoring, bucketing and
fitting functions were removed on 2026-11-22; `composite` keeps its two-part
key so rebuilt category ids match the ones already on disk.
"""
from __future__ import annotations
import collections
import math, re

# ── Axis A: Chinese penal code, 分则 chapters ──────────────────────────────
# Representative offences per chapter, for prompting a classifier. A model
# told only the chapter names sorts cases by everyday intuition -- it filed
# running a gambling den under "market order" -- which does not match how the
# code is actually divided.
ZH_CHAPTER_EXAMPLES = {
    "A1_국가안전": "간첩, 국가전복",
    "A2_공공안전": "방화(114-115), 교통사고(133), 음주운전(133-1), 총기, 폭발물",
    "A3_시장질서": "밀수, 가짜상품, 금융사기(192-200), 탈세, 지식재산권, 증권",
    "A4_인신권리": "살인(232), 상해(234), 강간(236), 감금, 유괴, 명예훼손",
    "A5_재산": "강도(263), 절도(264), 사기(266), 공갈, 횡령(270), 손괴",
    "A6_사회관리질서": "공무집행방해(277), 소란(293), 도박장개설(303), 마약(347), 문서위조, 음란물",
    "A7_국방이익": "군사시설 침해",
    "A8_탐오뇌물": "횡령/독직(382-383), 수뢰(385), 증뢰(389)",
    "A9_독직": "직권남용(397), 직무유기(397), 사법권 남용",
    "A10_군인직무": "군인 직무 위반",
}

ZH_CHAPTERS = [
    (102, 113, "A1_국가안전"),
    (114, 139, "A2_공공안전"),
    (140, 231, "A3_시장질서"),
    (232, 262, "A4_인신권리"),
    (263, 276, "A5_재산"),
    (277, 367, "A6_사회관리질서"),
    (368, 381, "A7_국방이익"),
    (382, 396, "A8_탐오뇌물"),
    (397, 419, "A9_독직"),
    (420, 451, "A10_군인직무"),
]


def zh_chapter(article: int) -> str:
    if article < 102:
        return "A0_총칙"          # general provisions, not a charge
    for lo, hi, name in ZH_CHAPTERS:
        if lo <= article <= hi:
            return name
    return "A_기타"


# LEGAR BENCH ships its own three-level taxonomy, and the query's Crime field
# is "d1 / d2 / d3" -- the top category, the specific charge, and the issue
# attributes. The eight d1 categories ARE the benchmark's own partition of the
# stricter split, so they are used directly rather than being re-derived from
# charge names.
KO_OFFICIAL = ("성범죄", "상해및폭행", "사기", "교통사고및운전",
               "마약", "명예훼손및모욕", "살인", "금융및보험")


def ko_official_category(crime) -> str:
    """The benchmark's own d1 label, read off the query's Crime field."""
    if isinstance(crime, list):
        crime = crime[0] if crime else ""
    head = str(crime or "").split("/")[0].strip()
    return f"A_{head}" if head in KO_OFFICIAL else ""


# ── Axis A: LEGAR documents in the benchmark's OWN label space ───────────
# The benchmark labels only its query cases (`crime` in query_structured.json);
# the 14,525 corpus documents are a disjoint set and carry no such field. So
# doc_chapter fell back to ko_group, whose labels are the Chinese chapter
# scheme (A11_성범죄, A5_재산, ...) while queries use the official d1 names
# (A_성범죄, A_상해및폭행, ...). The two never matched as strings, so for LEGAR
# **0.0% of gold cases shared the query's class** -- the chapter filter was
# discarding the entire corpus on every query.
#
# Documents do carry `statutes`, which is far cleaner than the noisy `charges`
# extraction (`형법 제347조` is fraud, unambiguously). These rules map statutes
# and charge text into the official 8-class space. Order matters: the specific
# offence wins, so 살인(250) precedes 상해(257) and 성범죄 precedes 폭행.
#
# Validated against the 5,960 query cases that DO carry an official label:
# **92.8% reproduced exactly.** The residual is multi-offence judgments (a case
# charging both 상해 and 성범죄), where the official label names the principal
# offence and a first-match rule cannot know which that is.
KO_OFFICIAL_RULES = [
    ("A_살인",         [r"형법\s*제25[0-4]조", r"살인"]),
    ("A_성범죄",       [r"성폭력", r"형법\s*제(29[789]|30[025])조", r"강간", r"추행",
                      r"음란", r"성매매", r"청소년성보호", r"아동·?청소년",
                      r"카메라등?이용"]),
    ("A_마약",         [r"마약류", r"향정", r"대마", r"마약"]),
    ("A_교통사고및운전", [r"도로교통법", r"교통사고처리", r"도주차량", r"음주운전",
                      r"무면허운전", r"특정범죄\s*가중처벌.*제5조의3"]),
    ("A_명예훼손및모욕", [r"형법\s*제3(0[789]|1[01])조", r"명예훼손", r"모욕",
                      r"정보통신망.*명예"]),
    ("A_사기",         [r"형법\s*제3(4[789]|5[12])조", r"사기", r"특정경제범죄.*사기"]),
    ("A_금융및보험",    [r"여신전문\s*금융업법", r"자본시장", r"보험업법",
                      r"상호저축은행", r"유사수신", r"대부업"]),
    ("A_상해및폭행",    [r"형법\s*제2(5[789]|6[0-5])조", r"폭력행위등", r"상해",
                      r"폭행", r"협박"]),
]
_KO_OFFICIAL_C = [(n, [re.compile(p) for p in ps])
                  for n, ps in KO_OFFICIAL_RULES]


def ko_official_from_text(*fields) -> str:
    """Official d1 label inferred from statutes / charges / topic text."""
    joined = " ".join(
        " ".join(str(x) for x in f) if isinstance(f, (list, tuple))
        else str(f or "")
        for f in fields)
    for name, pats in _KO_OFFICIAL_C:
        if any(p.search(joined) for p in pats):
            return name
    return "A_기타"


# ── RETIRED: Korean charges forced into the Chinese chapter scheme ────────
# Kept only so an old run's labels can still be read. Nothing calls this any
# more. Its names (A5_재산, A4_인신권리, A11_성범죄, ...) are the Chinese penal
# code's chapter divisions applied to Korean charge names, which is wrong on
# its own terms -- Korea's code divides offences differently -- and mixing it
# with the benchmark's official d1 names is what produced the 0.0% match.
# Each language uses its own taxonomy: zh_chapter for LeCaRD/v2,
# ko_official_* for LEGAR, en_area for COLIEE.
_RETIRED_KO_GROUPS = [
    ("A5_재산",        ["사기", "절도", "강도", "횡령", "배임", "장물", "손괴", "공갈", "권리행사"]),
    ("A4_인신권리",    ["살인", "상해", "폭행", "유기", "체포", "감금", "협박", "약취", "유인", "과실치"]),
    ("A11_성범죄",     ["성폭력", "강간", "추행", "음란", "성매매", "청소년성보호", "카메라"]),
    ("A2_공공안전",    ["교통", "도로교통", "실화", "방화", "일수", "교통방해", "음주운전"]),
    ("A12_마약",       ["마약", "향정", "대마"]),
    ("A6_사회관리질서", ["공무집행", "도박", "위증", "무고", "범인은닉", "증거인멸", "폭력행위등"]),
    ("A8_탐오뇌물",    ["뇌물", "수뢰", "증뢰", "직무유기", "직권남용"]),
    ("A13_문서",       ["문서", "인장", "유가증권", "통화"]),
    ("A14_경제행정",   ["조세", "관세", "식품", "약사", "건축", "환경", "근로", "자본시장", "여신"]),
]


def ko_group(charges: list[str]) -> str:
    """RETIRED -- see _RETIRED_KO_GROUPS. Do not use for new labels."""
    joined = " ".join(charges)
    for name, keys in _RETIRED_KO_GROUPS:
        if any(k in joined for k in keys):
            return name
    return "A_기타"


# ── Axis A: English (COLIEE) ──────────────────────────────────────────────
# COLIEE is Canadian Federal Court case law, not criminal cases: refugee and
# immigration matters dominate, followed by intellectual property and tax. The
# benchmark's own `Crime` tags are criminal-offence labels and read
# "civil/other" for 79% of queries, so the area of law is read off the text
# instead. Patterns are tried in order and the first hit wins -- the
# administrative catch-all is last, since "judicial review" appears in almost
# every Federal Court decision.
EN_AREAS = [
    ("A_난민이민",  [r"\brefugee", r"Convention refugee", r"\bimmigration",
                   r"removal order", r"deportation", r"permanent resident",
                   r"citizenship", r"\basylum", r"persecution"]),
    ("A_지식재산",  [r"\bpatent", r"trade-?mark", r"copyright", r"infring",
                   r"\binvention", r"Commissioner of Patents"]),
    ("A_조세",      [r"Income Tax Act", r"\btaxpayer", r"Minister of National Revenue",
                   r"\bGST\b", r"reassessment", r"taxation year"]),
    ("A_노동연금",  [r"employment insurance", r"\bpension", r"Canada Pension Plan",
                   r"labour relations", r"collective agreement", r"unemployment"]),
    ("A_형사",      [r"\bcriminal", r"\bsentenc", r"\bconviction", r"\baccused\b",
                   r"\bparole", r"indictment"]),
    ("A_행정기타",  [r"judicial review", r"\btribunal", r"administrative"]),
]
_EN_AREA_RE = [(name, [re.compile(p, re.I) for p in pats]) for name, pats in EN_AREAS]
EN_AREA_SCAN_CHARS = 8000


def en_area(text: str) -> str:
    t = (text or "")[:EN_AREA_SCAN_CHARS]
    for name, pats in _EN_AREA_RE:
        if any(p.search(t) for p in pats):
            return name
    return "A_기타"


def legal_class(query_row: dict, dataset: str,
                charge_to_article: dict[str, int] | None = None) -> str:
    """Axis A label for one query."""
    if dataset in ("LeCaRD", "LeCaRDv2"):
        arts = [int(a.split(":")[1]) for a in query_row.get("articles", [])
                if a.startswith("art:")]
        # Prefer a charge article: the general provisions get cited constantly
        # (sentencing rules) and would swamp every class into A0.
        specific = [a for a in arts if a >= 102]
        if specific:
            # The principal charge is the first 分则 article cited, not the
            # lowest-numbered one: a case charging 滥用职权 (397) alongside
            # 受贿 (385) is a dereliction case, and min() would call it bribery.
            return zh_chapter(specific[0])
        if charge_to_article:
            # A query may name several charges in one string ("妨害公务罪 危险驾驶罪").
            names = [n for c in query_row.get("charges", []) for n in str(c).split()]
            mapped = [charge_to_article[n] for n in names if n in charge_to_article]
            if mapped:
                return zh_chapter(mapped[0])
        return "A_기타"
    if dataset == "LEGAR_stricter":
        # Korean stays in the Korean label space. The old fallback here was
        # ko_group, whose names are the Chinese chapter scheme -- a query
        # missing `crime` would silently get a label from a different taxonomy
        # than every other query, which is how documents ended up 0.0%
        # matched (see KO_OFFICIAL_RULES). Infer from statutes instead.
        return (ko_official_category(query_row.get("crime"))
                or ko_official_from_text(query_row.get("statutes"),
                                         query_row.get("legal_refs"),
                                         query_row.get("charges")))
    return en_area(query_row.get("text") or "")


def doc_chapter(doc: dict, dataset: str) -> str:
    """The taxonomy class of a document, mirroring legal_class() for queries."""
    if dataset.startswith("LeCaRD"):
        arts = [int(a.split(":")[1]) for a in doc.get("articles", [])
                if a.startswith("art:")]
        specific = [a for a in arts if a >= 102]
        return zh_chapter(specific[0]) if specific else "A_기타"
    if dataset == "LEGAR_stricter":
        # Same label space as the queries -- see KO_OFFICIAL_RULES.
        return ko_official_from_text(doc.get("statutes"), doc.get("charges"),
                                     doc.get("topic"))
    return en_area(doc.get("content") or "")


def build_charge_to_article(cases: list[dict]) -> dict[str, int]:
    """Learn charge name -> penal-code article from the corpus itself.

    LeCaRD queries name a charge but cite no article, while its judgments carry
    both, so the mapping is read off the documents instead of being hand-written
    for 57 charge names.
    """
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for m in cases:
        arts = [int(a.split(":")[1]) for a in m.get("articles", []) if a.startswith("art:")]
        arts = [a for a in arts if a >= 102]
        if not arts:
            continue
        for c in m.get("charges", []):
            votes[c][min(arts)] += 1
    return {c: v.most_common(1)[0][0] for c, v in votes.items() if v}


# ── Composite category ────────────────────────────────────────────────────
def composite(legal: str, difficulty: str, registry: dict) -> int:
    """Stable integer id for (axis A, axis B), as per_category_overrides keys on an int."""
    key = f"{legal}|{difficulty}"
    if key not in registry:
        registry[key] = len(registry) + 1
    return registry[key]
