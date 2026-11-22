"""Build the corpus: documents, query labels, embeddings.

    python3 build.py cases    --all
    python3 build.py classify --all
    python3 build.py embed    --dataset LeCaRDv2 --queries --device cuda:2
    python3 build.py all      --dataset LeCaRDv2

Three scripts became one because they are ONE pipeline that must run in this
order, and nothing but a README said so: `classify` writes fold labels into
the rows `cases` produced, and `embed` reads the text `cases` chose. Running
them out of order, or forgetting the middle one, leaves a corpus that scores
without complaining -- the folds silently revert to an older split, and
`--split val` then dies on KeyError('NDCG@30') several steps later
(JOURNAL 46 is what that cost).

`all` runs the three in order for one dataset, which is the only order that
is ever correct.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, json, os, sys, time
from legal_meta import (extract_metadata, metadata_from_doc_id, article_numbers,
                        query_crime_metadata, LANG_OF, DOC_FIELD_OF)
import argparse, json, os, sys, time, collections
import legal_retriever as L
from legal_meta import LANG_OF
from legal_taxonomy import legal_class, build_charge_to_article, composite
import argparse, datetime as _dt, json, os, sys, time
import numpy as np


# ======================================================================
# build.py cases — Convert legal case datasets into EvolveMem memory units.
# ======================================================================

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


def _cli_build_cases() -> None:
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
                  f"classification labels. Run `build.py classify "
                  f"--dataset {ds}` before any sweep or evolution run.")
        for tag, rows in (("cases", cases), ("queries", qs)):
            suffix = "" if args.doc_field is None else f"_{args.doc_field}"
            path = f"{args.out}/{ds}_{tag}{suffix}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            print(f"  -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
        print(f"  {time.time() - t0:.1f}s\n")


# ======================================================================
# build.py classify — Assign each query a legal class and a composite id.
# ======================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.join(HERE, "corpus")
# Every benchmark ships a train/val/test split; the files under result/ are
# the one source that carries all three for all four. `output/` has a val for
# LEGAR only, and it is a DIFFERENT re-split: its 476 val rows and result's
# 429 are disjoint, while test is identical in both. Using one source keeps
# the four datasets comparable.
OFFICIAL_SPLIT = {
    "COLIEE": "COLIEE",
    "LeCaRD": "LeCaRD",
    "LeCaRDv2": "LeCaRDv2",
    "LEGAR_stricter": "LEGAR_BENCH",
}
SPLIT_PATH = "/home/yoosun/legal/result/{dir}/query_structured.json"


def assign_fold(qs: list[dict], dataset: str) -> dict:
    """Attach the benchmark's official train / val / test split as `fold`.

    Evolution runs on `train`, every check and repair happens on `val`, and
    `test` is scored once at the end. Reading `test` between runs would turn
    it into a validation set, which is what `val` is for.

    Raises when the official file does not cover every query: a silent
    fallback would mean two runs disagreeing about which queries are held
    out, and their numbers would not be comparable.
    """
    path = SPLIT_PATH.format(dir=OFFICIAL_SPLIT[dataset])
    with open(path, encoding="utf-8") as f:
        official = json.load(f)
    missing = [q["query_id"] for q in qs if str(q["query_id"]) not in official]
    if missing:
        raise KeyError(f"{dataset}: {len(missing)} queries absent from {path} "
                       f"(e.g. {missing[:3]})")
    for q in qs:
        q["fold"] = official[str(q["query_id"])]["split"]
    return collections.Counter(q["fold"] for q in qs)


def run(dataset: str) -> dict:
    """Label every query with its Axis-A class and a stable integer id.

    The retrieval-difficulty axis that used to sit alongside this was dropped:
    it was derived from BM25's own score spread, and measurement showed it
    predicted how hard a query is (weakly) without predicting that a different
    configuration would serve it better -- the per-class gain fell to zero once
    a globally good lever was available.
    """
    cases = json.load(open(f"{CASE_DIR}/{dataset}_cases.json", encoding="utf-8"))
    qs = json.load(open(f"{CASE_DIR}/{dataset}_queries.json", encoding="utf-8"))
    c2a = build_charge_to_article(cases) if dataset.startswith("LeCaRD") else None

    registry: dict[str, int] = {}
    for q in qs:
        q["legal_class"] = legal_class(q, dataset, c2a)
        # Axis B is retired (METHODOLOGY 6): `composite` keeps its two-part
        # key so rebuilt category ids stay identical to the ones already in
        # corpus/*_queries.json and in every saved per_category_overrides.
        q["category"] = composite(q["legal_class"], "", registry)

    assign_fold(qs, dataset)

    json.dump(qs, open(f"{CASE_DIR}/{dataset}_queries.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    meta = {"dataset": dataset, "registry": registry,
            "counts": {k: v for k, v in
                       collections.Counter(q["legal_class"] for q in qs).most_common()}}
    json.dump(meta, open(f"{CASE_DIR}/{dataset}_categories.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return meta


def _cli_classify_queries():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LANG_OF))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for ds in (sorted(LANG_OF) if a.all else [a.dataset]):
        meta = run(ds)
        qs = json.load(open(f"{CASE_DIR}/{ds}_queries.json", encoding="utf-8"))
        test = collections.Counter(q["legal_class"] for q in qs if q["split"] == "test")
        folds = collections.Counter(q.get("fold") for q in qs)
        print(f"[{ds}] classes={len(meta['registry'])}  folds="
              f"train {folds.get('train', 0)} / val {folds.get('val', 0)} / "
              f"test {folds.get('test', 0)}")
        for k, v in meta["counts"].items():
            print(f"     all {v:6d}  test {test.get(k, 0):5d}  {k}")


# ======================================================================
# build.py embed — Encode the corpus with a multilingual embedder for EvolveMem's semantic view.
# ======================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.join(HERE, "corpus")
EMB = os.path.join(HERE, "embeddings")
DEFAULT_MODEL = "BAAI/bge-m3"


def embed_text(m: dict) -> str:
    """Text handed to the encoder for one case.

    Mirrors MultiViewIndex's own concatenation of content with metadata, minus
    the party names: in case law the defendant's name is incidental, and
    EvolveMem only includes it because its conversational memories revolve
    around a handful of recurring speakers.
    """
    parts = [m.get("content", "")]
    ent = m.get("legal_refs") or m.get("entities") or []
    if ent:
        parts.append("Entities: " + ", ".join(str(e) for e in ent[:8]))
    topic = m.get("topic")
    if topic:
        parts.append(f"Topic: {topic}")
    return " | ".join(parts)


def load_model(name: str, device: str, max_len: int):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = max_len
    return model


# Retrieval models differ in how they mark the two sides of the match.
# EmbeddingGemma ships named prompts; E5 is trained on literal "query: " /
# "passage: " prefixes and loses accuracy without them; bge-m3 uses neither.
_E5_PREFIX = {"queries": "query: ", "cases": "passage: "}


def encode(model, texts, kind: str, batch_size: int, model_name: str = ""):
    prompts = getattr(model, "prompts", None) or {}
    name = {"queries": "query", "cases": "document"}[kind]
    kw = {}
    if name in prompts:
        kw["prompt_name"] = name
    elif "e5" in model_name.lower():
        texts = [_E5_PREFIX[kind] + t for t in texts]
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False, **kw)


def _cli_build_embeddings():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LANG_OF))
    ap.add_argument("--all", action="store_true")
    # "cuda", not "cuda:1". A hardcoded index picks a physical card, which on
    # a shared machine is someone else's: GPU 0 and one other here carry
    # another user's vLLM engines (84 and 53 GiB). "cuda" follows
    # CUDA_VISIBLE_DEVICES, so the caller selects the device -- and selecting
    # by UUID is safer still, because torch and nvidia-smi number the cards
    # in different orders.
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    # bge-m3's own limit, not a conservative default.
    #
    # This was 512, and 512 was never a decision -- it was the value the
    # three candidate encoders were PINNED at so the model comparison would
    # be fair (PROGRESS, "임베딩 모델 비교 ... max_len 512 고정"). bge-m3 won
    # that comparison and is the only one of the three that reads 8192; the
    # others cap at 512 and 2048. So the capability the model was chosen for
    # was then left switched off for every run after it.
    #
    # What 512 was costing, measured on 300-document samples:
    #     COLIEE    avg 6,200 tokens, 99% truncated, 512 covers 17%
    #     LeCaRD    avg 4,654 tokens, 99% truncated, 512 covers 31%
    #     LEGAR     avg 1,062 tokens, 86% truncated, 512 covers 62%
    #     LeCaRDv2  avg 1,234 tokens, 73% truncated, 512 covers 61%
    #
    # So BM25 read the whole judgment while the semantic view read a sixth of
    # it, and the conclusion drawn from that comparison -- "semantic scores far
    # below BM25 and only helps at weight 0.3" (section 8) -- is partly a
    # measurement of this setting rather than of semantic retrieval.
    #
    # Cost at 8192 is 4.0x of 512 (LeCaRDv2 55k: 16 min -> 62 min), not the
    # 16x the sequence length suggests, because most documents are far
    # shorter than the cap and batches are bounded by real length.
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--queries", action="store_true", help="also encode queries")
    ap.add_argument("--tag", default="m3", help="output file tag")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    os.makedirs(EMB, exist_ok=True)
    model = load_model(a.model, a.device, a.max_len)
    for ds in (sorted(LANG_OF) if a.all else [a.dataset]):
        targets = [("cases", embed_text)]
        if a.queries:
            targets.append(("queries", lambda q: q["text"]))
        for tag, to_text in targets:
            rows = json.load(open(f"{CASE_DIR}/{ds}_{tag}.json",
                                  encoding="utf-8"))
            texts = [to_text(r) for r in rows]
            t0 = time.time()
            vecs = encode(model, texts, tag, a.batch_size, a.model)
            path = f"{EMB}/{ds}_{tag}_{a.tag}.npy"
            np.save(path, vecs.astype(np.float16))
            # A fingerprint of the text that was encoded, beside the vectors.
            # Vector COUNT is not enough: changing which sections of a
            # judgment are indexed leaves the count identical, so BM25 would
            # read the new text while the semantic view kept vectors built from
            # the old -- a mismatch that passes every check and shows up only
            # as a worse number. The same silent-degradation shape as
            # METHODOLOGY 13.
            json.dump({"n": len(texts),
                       "chars": sum(len(t) for t in texts),
                       "model": a.model, "tag": a.tag,
                       "built": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")},
                      open(path[:-4] + ".meta.json", "w", encoding="utf-8"),
                      indent=2)
            print(f"[{ds}/{tag}] {len(texts)} texts, {time.time()-t0:.0f}s, "
                  f"dim={vecs.shape[1]} -> {path} "
                  f"({os.path.getsize(path)/1e6:.0f} MB)", flush=True)


# ── one entry point ───────────────────────────────────────────────────────
STAGES = {"cases": _cli_build_cases, "classify": _cli_classify_queries,
          "embed": _cli_build_embeddings}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in tuple(STAGES) + ("all",):
        print(__doc__)
        return 2
    stage, rest = sys.argv[1], sys.argv[2:]
    for name in (STAGES if stage == "all" else [stage]):
        if stage == "all":
            print(f"\n=== {name} ===", flush=True)
        sys.argv = [f"build.py {name}"] + rest
        STAGES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())