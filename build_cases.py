"""Convert legal case datasets into EvolveMem memory units.

One case = one memory unit. EvolveMem's own LLM extractor is bypassed: it is
built to distill dialogue into facts, whereas a judgment is already the atomic
retrieval unit, and running it over 55k documents would dominate cost.

Queries are converted with the same extractor. In these benchmarks a query is
itself a case fact pattern, so both sides of the match share a vocabulary -- and
EvolveMem's built-in query analyser cannot be used here anyway, as it detects
entities by capitalisation (multi_retriever.py:482), which does not exist in
Chinese or Korean.

Usage:
    python build_cases.py --dataset LeCaRDv2
    python build_cases.py --all --out corpus/
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legal_meta import (extract_metadata, metadata_from_doc_id, article_numbers,
                        query_crime_metadata, LANG_OF, DOC_FIELD_OF)

DATA_DIR = "/home/yoosun/legal/output"
META_FIELDS = ("Fact", "Reason", "Result")


def _meta_for(text: str, lang: str, doc_id: str | None = None) -> dict:
    meta = extract_metadata(text, lang)
    if doc_id is not None:
        for k, v in metadata_from_doc_id(doc_id, lang).items():
            if v:
                meta[k] = v
    return meta


def _impute_articles(rows: list[dict]) -> int:
    """Fill in article numbers for documents that cite none.

    Most LeCaRDv2 judgments name the offence but never cite the code article
    (64% of them), which leaves the structured view blind on two thirds of the
    corpus. The charge -> article mapping learned from the documents that do
    cite both is applied to the rest, so a case is still reachable by article.
    """
    from legal_taxonomy import build_charge_to_article
    charge_to_article = build_charge_to_article(rows)
    filled = 0
    for r in rows:
        if r["articles"]:
            continue
        arts = _uniq_keep([f"art:{charge_to_article[c]}" for c in r["charges"]
                           if c in charge_to_article], 8)
        if arts:
            r["articles"] = arts
            r["legal_refs"] = r["legal_refs"] + arts
            r["keywords"] = r["keywords"] + arts
            r["imputed_articles"] = True
            filled += 1
    return filled


def _uniq_keep(seq, cap):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
            if len(out) >= cap:
                break
    return out


def build_documents(dataset: str, doc_field: str, meta_fields: tuple[str, ...]) -> list[dict]:
    lang = LANG_OF[dataset]
    docs = json.load(open(f"{DATA_DIR}/{dataset}_document.json", encoding="utf-8"))
    out = []
    for d in docs:
        did = str(d["doc_id"])
        if doc_field == "all":
            content = " ".join(str(d.get(f) or "") for f in META_FIELDS).strip()
        else:
            content = str(d.get(doc_field) or "")
        meta = _meta_for(" ".join(str(d.get(f) or "") for f in meta_fields), lang, did)
        arts = article_numbers(meta["_statutes"])
        out.append({
            # MultiViewIndex reads these keys directly.
            "content":   content,
            "keywords":  meta["keywords"] + arts,
            "parties":    meta["parties"],
            "legal_refs": meta["legal_refs"] + arts,
            "topic":     meta["topic"],
            "location":  meta["location"],
            "timestamp": meta["timestamp"],
            "articles":  arts,
            # Ours: lets a retrieved hit be mapped back to a qrels doc_id.
            "doc_id":    did,
            "statutes":  meta["_statutes"],
            "charges":   meta["_charges"],
            "imputed_articles": False,
        })
    filled = _impute_articles(out)
    if filled:
        print(f"  imputed articles for {filled}/{len(out)} documents")
    return out


def build_queries(dataset: str, query_field: str = "Q") -> list[dict]:
    lang = LANG_OF[dataset]
    queries = json.load(open(f"{DATA_DIR}/{dataset}_query.json", encoding="utf-8"))
    out = []
    for q in queries:
        text = str(q.get(query_field) or "")
        meta = _meta_for(text, lang)
        crime = q.get("Crime") or []
        cmeta = query_crime_metadata(crime, lang, dataset)
        arts = article_numbers(meta["_statutes"]) + [
            e for e in cmeta.get("legal_refs", []) if e.startswith("art:")]
        out.append({
            "query_id":  str(q["Query_id"]),
            "text":      text,
            "split":     q.get("split", ""),
            "crime":     crime if isinstance(crime, list) else [str(crime)],
            "parties":    meta["parties"],
            "legal_refs": sorted(set(meta["legal_refs"] + cmeta.get("legal_refs", []))),
            "topic":     meta["topic"] or cmeta.get("topic", ""),
            "location":  meta["location"],
            "timestamp": meta["timestamp"],
            # Order matters: the first 分则 article a judgment cites is the
            # principal charge, the rest are sentencing provisions.
            "articles":  _uniq_keep(arts, 12),
            "statutes":  meta["_statutes"],
            "charges":   sorted(set(meta["_charges"] + cmeta.get("charges", []))),
        })
    return out


# The classification labels a rebuild carries over, and the same tuple a
# previous build must already have for its labels to be worth carrying.
#
# These were two tuples. The carry list also held Axis B
# (`difficulty_class`, `profile_gap`, `profile_entropy`), which is RETIRED
# (METHODOLOGY 6) and which no builder has written since -- zero occurrences
# across all four corpus/*_queries.json, 8,545 rows. While the required list
# still named `difficulty_class`, every rebuild silently carried nothing and
# `if carried:` printed nothing either; the folds came back as the OLD
# two-way train/test split and `fusion_sweep --split val` died on
# KeyError('NDCG@30'). One tuple cannot drift from itself.


def _carry_labels(path: str, rows: list[dict]) -> int:
    """Copy classification labels from a previous build of the same queries."""
    if not os.path.exists(path):
        return 0
    try:
        old = {r["query_id"]: r for r in json.load(open(path, encoding="utf-8"))}
    except (ValueError, KeyError):
        return 0
    n = 0
    for r in rows:
        prev = old.get(r["query_id"])
        if prev and all(k in prev for k in LABEL_KEYS):
            for k in LABEL_KEYS:
                if k in prev:
                    r[k] = prev[k]
            n += 1
    return n


def _stats(name: str, rows: list[dict], keys: tuple[str, ...]) -> None:
    n = len(rows)
    parts = []
    for k in keys:
        hit = sum(1 for r in rows if r.get(k))
        parts.append(f"{k}={hit / n * 100:.1f}%")
    print(f"  {name:10s} n={n:<6d} " + "  ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LANG_OF))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--doc-field", default=None,
                    choices=["Fact", "Reason", "Result", "all"],
                    help="field used as retrievable content (default Fact, "
                         "matching the BM25 baseline)")
    ap.add_argument("--meta-fields", default=",".join(META_FIELDS),
                    help="fields scanned for metadata only")
    ap.add_argument("--out", default="/home/yoosun/legal/evolve_legal/corpus")
    args = ap.parse_args()

    targets = sorted(LANG_OF) if args.all else [args.dataset]
    if not targets or targets == [None]:
        ap.error("pass --dataset or --all")
    os.makedirs(args.out, exist_ok=True)
    meta_fields = tuple(f.strip() for f in args.meta_fields.split(","))

    for ds in targets:
        t0 = time.time()
        print(f"[{ds}] lang={LANG_OF[ds]}")
        doc_field = args.doc_field or DOC_FIELD_OF[ds]
        print(f"  doc_field={doc_field}")
        cases = build_documents(ds, doc_field, meta_fields)
        qs = build_queries(ds)
        _stats("documents", cases, ("legal_refs", "parties", "topic", "location", "timestamp", "articles"))
        _stats("queries", qs, ("legal_refs", "parties", "topic", "location", "timestamp", "articles"))
        # Classification labels live on the query rows and are produced by a
        # separate, slower pass. Rebuilding extraction must not silently drop
        # them, or the next sweep fails on a missing `fold`.
        carried = _carry_labels(f"{args.out}/{ds}_queries{'' if args.doc_field is None else f'_{args.doc_field}'}.json", qs)
        if carried:
            print(f"  carried classification labels for {carried}/{len(qs)} queries")
        if carried < len(qs):
            # Loud, because the failure mode is invisible downstream: a query
            # with no `fold` is simply absent from every split, so the sweep
            # reports on fewer queries -- or none -- rather than erroring.
            print(f"  !! {len(qs) - carried}/{len(qs)} queries have NO "
                  f"classification labels. Run `classify_queries.py "
                  f"--dataset {ds}` before any sweep or evolution run.")
        for tag, rows in (("cases", cases), ("queries", qs)):
            suffix = "" if args.doc_field is None else f"_{args.doc_field}"
            path = f"{args.out}/{ds}_{tag}{suffix}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            print(f"  -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
        print(f"  {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
