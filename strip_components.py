"""Which part of the strip list costs the -0.0170 on LeCaRDv2?

JOURNAL 41 measured two strip lists -- parties, and parties+org+place+court
-- and the wider one cost -0.0170 on LeCaRDv2 while being null on COLIEE at a
similar coverage. That left the real question open: is a place name genuinely
useful to BM25 (which would be odd -- two cases sharing a city are not alike),
or does the Chinese place regex take characters it should not?

This splits the list into its four sources and measures each on top of
parties, then prints what was actually deleted from the queries that lost the
most. A regex fault and a real signal look the same in the aggregate number
and completely different in the spans.

    python3 strip_components.py LeCaRDv2            # measure + worst spans
    python3 strip_components.py LeCaRDv2 --spans 0  # measure only
"""
from __future__ import annotations
import argparse, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legal_retriever as L
import legal_meta as LM
import recipe_learning as RL
from evaluate import EVAL_K, PRIMARY, load_fold, weak_config
from legal_meta import LANG_OF


def strip(text: str, spans) -> str:
    out = text
    for sp in sorted(spans, key=len, reverse=True):
        out = out.replace(sp, "")
    return re.sub(r"\s+", " ", out).strip(" ?.,")


def components(row, text, lang):
    """The four sources, kept apart instead of unioned.

    org/place come from `legal_meta.extract_incidentals(by_kind=True)` -- the
    real code path. An earlier version of this script re-ran the regexes
    itself and the copy kept a first-character deictic test after the module
    had moved to a whole-span one, so a fix to the extractor did not reach
    the measurement at all (JOURNAL 41).
    """
    parties = [p for p in L.parties_of(row) if L.is_party_name(p, lang)]
    kinds = LM.extract_incidentals(text, lang, by_kind=True)
    loc = str(row.get("location") or "").strip()
    court = [loc] if len(loc) >= 2 else []
    keep = lambda xs: [x.strip() for x in xs if len(x.strip()) >= 2]
    return {"parties": keep(parties), "org": keep(kinds["org"]),
            "place": keep(kinds["place"]), "court": court}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+",
                    help="one or more datasets. Absorbed strip_primary.py on "
                         "2026-11-12: its two variants ('parties only',"
                         " 'parties+org+place') are the first and last rows "
                         "of the component table below, measured from the "
                         "same theta_0, so it was a strict subset")
    ap.add_argument("--spans", type=int, default=12,
                    help="how many worst-hurt queries to print spans for")
    a = ap.parse_args()
    for ds in a.datasets:
        _one(ds, a.spans)


def _one(ds: str, n_spans: int) -> None:
    lang = LANG_OF[ds]

    index, rows, labels = load_fold(ds, "train")
    cfg = weak_config()
    comp = {q["query_id"]: components(q, q["text"], lang) for q in rows}

    def run(pick):
        """pick: which component names to delete."""
        out, used = [], 0
        for q in rows:
            sp = [s for k in pick for s in comp[q["query_id"]][k]]
            used += bool(sp)
            r = {**q, "text": strip(q["text"], sp)} if sp else q
            ranked = L.search(index, r, cfg, top_n=30)
            out.append(L.MBR.compute_query_metrics(
                ranked, labels[q["query_id"]], EVAL_K)[PRIMARY])
        return out, used

    base, _ = run([])
    thr, _ = RL.noise_floor(ds)
    print(f"\n[{ds}] train n={len(rows)}  theta_0 = {st.fmean(base):.4f}  "
          f"noise floor {thr:.4f}")
    print(f"  {'삭제 대상':28s} {'점수':>7s} {'Δ':>9s}  {'CI95':>22s}  적용률")
    results = {}
    for name, pick in [
        ("parties", ["parties"]),
        ("parties+court", ["parties", "court"]),
        ("parties+org", ["parties", "org"]),
        ("parties+place", ["parties", "place"]),
        ("parties+org+place+court", ["parties", "org", "place", "court"]),
    ]:
        v, used = run(pick)
        d = [y - x for x, y in zip(base, v)]
        lo, hi = RL.bootstrap_ci(d)
        results[name] = (v, d)
        mark = "악화" if hi < 0 else ("개선" if lo > 0 else "구분 불가")
        print(f"  {name:28s} {st.fmean(v):7.4f} {st.fmean(d):+9.4f}  "
              f"[{lo:+.4f},{hi:+.4f}]  {used}/{len(rows)}  {mark}")

    if n_spans:
        # the place-only variant is the one under suspicion
        _, d = results["parties+place"]
        order = sorted(range(len(rows)), key=lambda i: d[i])[:n_spans]
        print(f"\n  가장 많이 떨어진 {n_spans}건 — place로 지워진 span")
        for i in order:
            q = rows[i]
            pl = comp[q["query_id"]]["place"]
            if not pl:
                continue
            print(f"   Δ={d[i]:+.4f}  place={pl}")
            before = q["text"][:90].replace("\n", " ")
            after = strip(q["text"], comp[q["query_id"]]["parties"] + pl)[:90]
            print(f"      전: {before}")
            print(f"      후: {after}")
    del index


if __name__ == "__main__":
    main()
