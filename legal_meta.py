"""Rule-based metadata extraction for legal case documents.

Feeds EvolveMem's structured-metadata retrieval view, which reads the keys
`parties`, `legal_refs`, `topic`, `location`, `timestamp` off each memory dict
(the engine reads the first two as `persons`/`entities` — see legal_retriever).
LLM extraction is skipped on purpose: at 55k documents it would dominate cost,
and legal text carries its discriminative signals (statutes, charges, parties)
in highly regular surface forms.
"""
from __future__ import annotations
import re

# Field indexed as retrievable content. Held uniform at "Fact" across datasets
# so that framework comparisons are not confounded by the input text differing
# per benchmark. Note this is NOT how the published baselines were measured:
# their cached BM25 indices used Fact+Reason+Result for COLIEE and LEGAR, which
# scores higher (COLIEE NDCG@30 0.253 vs 0.200, LEGAR 0.060 vs 0.037). The
# baselines must therefore be re-measured under this setting before comparison.
DOC_FIELD_OF = {
    "COLIEE": "Fact",
    "LEGAR_stricter": "Fact",
    "LeCaRD": "Fact",
    "LeCaRDv2": "Fact",
}

LANG_OF = {
    "COLIEE": "en",
    "LeCaRD": "zh",
    "LeCaRDv2": "zh",
    "LEGAR_stricter": "ko",
}

_ZH_NUM = "一二三四五六七八九十百千零〇0-9"

# ── Chinese ────────────────────────────────────────────────────────────────
_ZH_STATUTE = re.compile(rf"《([^》]{{2,40}})》第([{_ZH_NUM}]{{1,10}})条")
# Many LeCaRDv2 judgments cite bare articles without naming the code.
_ZH_ARTICLE = re.compile(rf"第([{_ZH_NUM}]{{1,10}})条")
# Greedy up to the final 罪 so compound charge names stay whole:
# "构成掩饰、隐瞒犯罪所得、犯罪所得收益罪" is one charge, not two fragments.
_ZH_CHARGE  = re.compile(r"(?:构成|犯|指控|以)([\u4e00-\u9fa5、]{2,18})罪")
_ZH_CHARGE_BAD = ("认罪", "被告人", "当庭", "表示", "悔罪", "本案", "归案", "判处")
# Names run 2-3 characters, or are anonymised as 王某 / 徐某1. A plain
# {2,4} match swallows the verb that follows ("莫新国酒后驾驶" -> "莫新国酒").
_ZH_PERSON  = re.compile(r"(?:被告人|被害人|原告|被告)([\u4e00-\u9fa5]{1,2}某\d?|[\u4e00-\u9fa5]{2,3})")
_ZH_DATE    = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_ZH_COURT   = re.compile(r"([一-龥]{2,12}?(?:人民法院|人民检察院))")

# ── Korean ─────────────────────────────────────────────────────────────────
# A law name is the words immediately before "제N조". It may be two
# characters ("형법" -- the most cited statute in Korean criminal
# judgments) or a spaced phrase ("폭력행위 등 처벌에 관한 법률"), so the
# prefix is optional and leading particles are trimmed afterwards.
_KO_STATUTE = re.compile(r"((?:[가-힣]+\s+){0,5}[가-힣]*법(?:률)?)\s*(?:제\s*)?(\d{1,4})\s*조")
_KO_JOSA = ("은", "는", "이", "가", "을", "를", "의", "에", "서", "로",
            "와", "과", "도", "만", "며", "고", "여", "서는", "에서")
_KO_CHARGE  = re.compile(r"([가-힣][가-힣()·\s]{1,30}?)죄")
_KO_PERSON  = re.compile(r"피고인\s*([A-Z]{1,2}\b|[가-힣]{2,4})")
_KO_DATE    = re.compile(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.")
_KO_COURT   = re.compile(r"([가-힣]{2,10}(?:지방|고등|가정|행정)?법원)")

# ── English (COLIEE) ───────────────────────────────────────────────────────
# Canadian judgments cite as "s. 56 of the Trade-marks Act", "ss. 3-4",
# "section 2(1)", or bare "s. 3 of the Act". Act names carry hyphens
# and digits.
_EN_STATUTE = re.compile(
    r"\b(?:ss?\.|sections?|subsections?)\s*(\d+(?:\.\d+)*(?:\(\d+\))?)"
    r"(?:\s*(?:of|under)\s+the\s+([A-Z][A-Za-z\-'0-9 ]{2,45}?Act))?",
    re.IGNORECASE)
# MLB headnotes carry a topical taxonomy, e.g. "Aliens - Topic 1323.4".
_EN_TOPIC   = re.compile(r"\b([A-Z][A-Za-z]+(?:\s[A-Za-z]+){0,2})\s-\sTopic\s(\d+(?:\.\d+)?)")
_EN_ACT     = re.compile(r"\b((?:[A-Z][A-Za-z]+\s){1,4}Act)\b")
_EN_JUDGE   = re.compile(r"\b([A-Z][a-zA-Z'\-]+),\s*(?:J|JJ|C\.J|J\.A)\.")
_EN_DATE    = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})")
_EN_COURT   = re.compile(r"\b((?:Federal|Supreme|Tax|Appeal)\s+Court(?:\s+of\s+Appeal)?)")


def _ko_law_name(raw: str) -> str:
    """Drop leading words that belong to the sentence, not to the law name."""
    words = raw.split()
    while len(words) > 1 and words[0].endswith(_KO_JOSA):
        words.pop(0)
    return " ".join(words)


def _uniq(seq, cap):
    out, seen = [], set()
    for x in seq:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
            if len(out) >= cap:
                break
    return out


def extract_metadata(text: str, lang: str, cap: int = 12) -> dict:
    """Return the metadata block MultiViewIndex's structured view consumes."""
    t = text or ""
    if lang == "zh":
        statutes = [f"{m.group(1)}第{m.group(2)}条" for m in _ZH_STATUTE.finditer(t)]
        if not statutes:
            statutes = [f"第{m.group(1)}条" for m in _ZH_ARTICLE.finditer(t)]
        charges  = [f"{m.group(1)}罪" for m in _ZH_CHARGE.finditer(t)
                    if not any(b in m.group(1) for b in _ZH_CHARGE_BAD)]
        persons  = [m.group(1) for m in _ZH_PERSON.finditer(t)]
        courts   = [m.group(1) for m in _ZH_COURT.finditer(t)]
        dates    = [f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                    for m in _ZH_DATE.finditer(t)]
    elif lang == "ko":
        statutes = [f"{_ko_law_name(m.group(1))} 제{m.group(2)}조"
                    for m in _KO_STATUTE.finditer(t)]
        charges  = [f"{m.group(1).strip()}죄" for m in _KO_CHARGE.finditer(t)]
        persons  = [m.group(1) for m in _KO_PERSON.finditer(t)]
        courts   = [m.group(1) for m in _KO_COURT.finditer(t)]
        dates    = [f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                    for m in _KO_DATE.finditer(t)]
    else:
        statutes = [f"s.{m.group(1)} {(m.group(2) or 'Act').strip()}"
                    for m in _EN_STATUTE.finditer(t)]
        statutes += _uniq(_EN_ACT.findall(t), cap)
        charges  = [f"{m.group(1)} Topic {m.group(2)}" for m in _EN_TOPIC.finditer(t)]
        persons  = [m.group(1) for m in _EN_JUDGE.finditer(t)]
        courts   = [m.group(1) for m in _EN_COURT.finditer(t)]
        _M = {m: i for i, m in enumerate(
            "January February March April May June July August September "
            "October November December".split(), start=1)}
        dates = [f"{m.group(3)}-{_M[m.group(1)]:02d}-{int(m.group(2)):02d}"
                 for m in _EN_DATE.finditer(t)]

    statutes = _uniq(statutes, cap)
    charges  = _uniq(charges, cap)
    dates    = _uniq(dates, cap)
    return {
        "legal_refs": _uniq(statutes + charges, cap),
        "parties":    _uniq(persons, cap),
        "location":  (_uniq(courts, 1) or [""])[0],
        "topic":     charges[0] if charges else "",
        "timestamp": dates[0] if dates else "",
        "keywords":  statutes + charges,
        "_statutes": statutes,
        "_charges":  charges,
        "_dates":    dates,
    }


# Korean doc ids encode court, year and case class, e.g. "광주지방법원-2010고단3888".
_KO_DOCID = re.compile(r"^(.+?)-(\d{4})([가-힣]{1,3})(\d+)")


def metadata_from_doc_id(doc_id: str, lang: str) -> dict:
    """Metadata recoverable from the identifier alone, with no text scan."""
    if lang != "ko":
        return {}
    m = _KO_DOCID.match(str(doc_id))
    if not m:
        return {}
    court, year, case_class, _ = m.groups()
    return {"location": court, "case_class": case_class, "filing_year": year}


# ── Article-number normalisation ───────────────────────────────────────────
# Chinese judgments spell article numbers out ("第二百六十三条") while the
# LeCaRDv2 query metadata gives them as digits ("263"), so neither side matches
# the other until both are reduced to an integer.
_ZH_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_ZH_UNIT = {"十": 10, "百": 100, "千": 1000}
_ARTICLE_RE = re.compile(r"第([一二三四五六七八九十百千零〇0-9]+)条")


def zh_to_int(s: str) -> int | None:
    s = str(s).strip()
    if s.isdigit():
        return int(s)
    section = num = 0
    for ch in s:
        if ch in _ZH_DIGIT:
            num = _ZH_DIGIT[ch]
        elif ch in _ZH_UNIT:
            section += (num or 1) * _ZH_UNIT[ch]
            num = 0
        else:
            return None
    return section + num or None


def article_numbers(statutes: list[str]) -> list[str]:
    """Normalised article tokens ("art:263") shared by documents and queries."""
    out = []
    for s in statutes:
        m = _ARTICLE_RE.search(str(s))
        if m:
            v = zh_to_int(m.group(1))
            if v:
                out.append(f"art:{v}")
    return _uniq(out, 20)


def query_crime_metadata(crime, lang: str, dataset: str) -> dict:
    """Interpret the per-dataset `Crime` field on a query.

    Its content differs by benchmark: LeCaRD gives a charge name, LeCaRDv2 a
    list of penal-code article numbers, LEGAR a charge plus annotated issue
    attributes, and COLIEE a coarse area-of-law label.
    """
    if isinstance(crime, str):
        crime = [crime]
    items = [str(c).strip() for c in (crime or []) if str(c).strip()]
    if not items:
        return {}
    if dataset == "LeCaRDv2":
        nums = [f"art:{t}" for t in items[0].split() if t.isdigit()]
        return {"legal_refs": nums, "charges": [], "topic": nums[0] if nums else ""}
    if dataset == "LEGAR_stricter":
        head = items[0].split("[")[0]
        charges = _uniq([p.strip() for p in head.split("/") if p.strip()], 6)
        attrs = _uniq(re.findall(r"\['([^']+)',\s*\d+\]", items[0]), 20)
        return {"legal_refs": charges + attrs, "charges": charges,
                "topic": charges[0] if charges else ""}
    return {"legal_refs": items, "charges": items, "topic": items[0]}


# ── Incidentals: what the party-strip view deletes ─────────────────────────
# The party-strip view (legal_retriever's name for the engine's entity-swap)
# removes from the query the spans that identify *this* case rather than the
# kind of case it is, then searches on what is left. Party names were the only
# such span originally, but they are not the only one: a judgment names the
# company, the bank, the branch office, the city and the district, and all of
# them are as case-specific as a defendant's name. A BM25 view that keeps them
# spends its scoring mass matching "重庆市北碚区" against every other case
# tried in that district.
#
# These are deliberately NOT the same thing as `location`, which stays in the
# metadata view: the court is a useful grouping signal there (same court, same
# sentencing practice) and noise here (the facts do not depend on it).
#
# Precision matters more than recall for a strip list. A pattern that fires on
# an ordinary word deletes that word from every query, so each language's place
# pattern is anchored on a closed list of administrative names rather than on
# the suffix alone -- Korean 동/시 and English "Co." are too common as
# fragments of other words to match on the suffix.

# A span is only taken where it starts at a boundary: Chinese runs together,
# so a lazy quantifier anchored nowhere matches "谋抢劫后来到重庆市" instead of
# "重庆市". The boundary is punctuation, a motion/locative preposition, or
# another administrative suffix, which is what lets "重庆市北碚区" yield both.
_ZH_PREP = "\u4e8e\u5728\u81f3\u5230\u7ecf\u5f80\u8d74\u6765\u56de\u53bb\u4ece\u81ea\u5411"
_ZH_BOUND = (rf"(?:(?<=[{_ZH_PREP}\u7701\u5e02\u533a\u53bf\u9547\u4e61\u6751\u9053\u53f7\s\uff0c\u3002\u3001\uff1b\uff1a\uff08\uff09(),.])|^)"
             rf"(?![{_ZH_PREP}])")

_ZH_ORG = re.compile(
    _ZH_BOUND +
    r"([\u4e00-\u9fa5]{2,10}?(?:\u80a1\u4efd\u6709\u9650\u516c\u53f8|\u6709\u9650\u8d23\u4efb\u516c\u53f8|\u6709\u9650\u516c\u53f8|\u516c\u53f8|\u94f6\u884c|"
    r"\u4fe1\u7528\u793e|\u96c6\u56e2|\u5de5\u5382|\u533b\u9662|\u5b66\u6821|\u5927\u5b66|\u6d3e\u51fa\u6240|\u5206\u5c40|\u516c\u5b89\u5c40|\u5de5\u5546\u5c40|\u7ba1\u7406\u5c40|"
    r"\u59d4\u5458\u4f1a|\u4e8b\u52a1\u6240))")
_ZH_PLACE = re.compile(
    _ZH_BOUND +
    r"([\u4e00-\u9fa5]{2,3}(?:\u81ea\u6cbb\u533a|\u81ea\u6cbb\u5dde|\u81ea\u6cbb\u53bf|\u7701|\u5e02|\u53bf|\u533a|\u9547|\u4e61|\u6751|\u8857\u9053))")
# Function words cannot appear inside a name, so a span carrying one started
# mid-sentence: "的营业执照及对公账户银行" is a clause, not a bank.
_ZH_BAD_IN_NAME = "的及与和或对向把被将其此该并"

# 本市 / 该区 and similar deictic forms are not place names.
_ZH_PLACE_BAD = ("本", "该", "此", "其", "同", "外", "内", "地", "辖", "全")

_KO_REGION = (r"서움|부산|대구|인천|광주|대전|울산|세종|경기|강원|충정북도|충북|"
              r"충정남도|충남|전라북도|전북|전라남도|전남|경상북도|경북|"
              r"경상남도|경남|제주")
_KO_PLACE = re.compile(
    rf"((?:{_KO_REGION})(?:\ud2b9\ubcc4\uc790\uce58\uc2dc|\ud2b9\ubcc4\uc790\uce58\ub3c4|\ud2b9\ubcc4\uc2dc|\uad11\uc5ed\uc2dc|\ub3c4)?"
    rf"(?:\s*[\uac00-\ud7a3]{{1,6}}(?:\uc2dc|\uad70|\uad6c))?(?:\s*[\uac00-\ud7a3]{{1,6}}(?:\uc77d|\uba74|\ub3d9|\ub9ac))?)")
# 보험 / 의원 / 증워 are left out on purpose: they are also ordinary nouns
# ("자동차보험", "대의원"), and a strip list pays for a false positive in
# every query that contains the word.
_KO_ORG = re.compile(
    r"((?:\uc8fc\uc2dd\ud68c\uc0ac\s*)?[\uac00-\ud7a3A-Z]{1,15}(?:\uc8fc\uc2dd\ud68c\uc0ac|\u321c|\uc740\ud589|"
    r"\uacf5\uc0ac|\uacf5\ub2e8|\uc870\ud569|\uc9c0\ud5a5|\uc9c0\ubd80|\ub178\ub3d9\uc870\ud569|\ub300\ud559\uad50|\uace0\ub4f1\ud559\uad50|\uc911\ud559\uad50|\ucd08\ub4f1\ud559\uad50|"
    r"\ubcc4\uc6d0|\uacbd\ucc30\uc11c|\uc9c0\uc911|\uac80\ucc30\uc9c1|\uc138\ubb34\uc11c|\uad6c\uc9c1|\uc2dc\uc9c1|\uad70\uc9c1))")

# Two English patterns, because the suffixes differ in how much they carry on
# their own. "Ltd." names a company by itself; "Commission" or "Board" is an
# ordinary noun until a proper name precedes it, so the weak set requires one
# and rejects a bare article.
_EN_ORG_STRONG = re.compile(
    r"((?:[A-Z][A-Za-z&.'\-]*\s){0,4}"
    r"(?:Inc\.|Ltd\.|Limited|Corp\.|Corporation|Company|LLC|L\.L\.C\.|"
    r"Bank|University))")
_EN_ORG_WEAK = re.compile(
    r"((?!The\b)(?:[A-Z][A-Za-z&.'\-]*\s){1,4}"
    r"(?:Commission|Board|Tribunal|Agency|Ministry|Association|Authority))")
_EN_PLACE = re.compile(
    r"\b(Ontario|Quebec|Qu\u00e9bec|British Columbia|Alberta|Manitoba|"
    r"Saskatchewan|Nova Scotia|New Brunswick|Newfoundland(?: and Labrador)?|"
    r"Prince Edward Island|Yukon|Nunavut|Northwest Territories|"
    r"Toronto|Ottawa|Montreal|Montr\u00e9al|Vancouver|Calgary|Edmonton|"
    r"Winnipeg|Halifax|Canada)\b")

# Korean party extraction matches "피고인들의" and leaves the josa fragment
# "들의" as if it were a name. Deleting that from a query removes a particle
# from every sentence it appears in, so fragments beginning with 들 are
# rejected here rather than fixed at the regex, which also has to keep
# matching the single-letter anonymised names (피고인 C) the corpus uses.
_KO_NOT_A_NAME = re.compile(r"^들")


def is_party_name(name: str, lang: str) -> bool:
    """False for spans the party regexes produce that are not names."""
    if lang == "ko":
        return not _KO_NOT_A_NAME.match(name.strip())
    return bool(name.strip())


def extract_incidentals(text: str, lang: str, cap: int = 24) -> list[str]:
    """Case-specific spans to delete for the party-strip view.

    Parties are NOT included: they come from the precomputed row, so the
    caller unions the two. Returned longest-first, because the spans nest
    ("重庆市北碚区" contains "北碚区") and a shorter span removed first
    leaves the longer one as a fragment.
    """
    t = text or ""
    if lang == "zh":
        found = [m.group(1) for m in _ZH_ORG.finditer(t)]
        found += [m.group(1) for m in _ZH_PLACE.finditer(t)
                  if m.group(1)[0] not in _ZH_PLACE_BAD]
        found = [f for f in found
                 if not any(c in f for c in _ZH_BAD_IN_NAME)]
    elif lang == "ko":
        found = [m.group(1) for m in _KO_ORG.finditer(t)]
        found += [m.group(1).strip() for m in _KO_PLACE.finditer(t)]
    else:
        found = [m.group(1).strip() for m in _EN_ORG_STRONG.finditer(t)]
        found += [m.group(1).strip() for m in _EN_ORG_WEAK.finditer(t)]
        found += [m.group(1) for m in _EN_PLACE.finditer(t)]
    out = _uniq([f for f in found if len(f) >= 2], cap)
    return sorted(out, key=len, reverse=True)
