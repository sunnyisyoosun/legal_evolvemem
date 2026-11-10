"""Pool-aware, multilingual retrieval wrapper around EvolveMem's multi-view index.

`retrieve_multiview` carries the fusion, time-decay, per-category override and
entity-swap logic that the evolution engine tunes, so it is reused unchanged.
Three things it assumes do not hold for legal case retrieval, and each is
patched at its own seam rather than by forking the function:

1. BM25 tokenisation is `content.lower().split()`, which makes a whole Chinese
   document a single token and leaves Korean particles attached. The index is
   rebuilt here with the same per-language tokenisers the BM25 baseline uses,
   so EvolveMem and the baseline tokenise identically.
2. Query analysis detects entities by capitalisation (multi_retriever.py:482),
   which does not exist in Chinese or Korean. `analyze_query` is swapped for one
   that reads the metadata precomputed by build_cases.py.
3. There is no notion of a per-query candidate pool, which LeCaRD's official
   re-ranking protocol requires. Every view honours an index-level mask.
"""
from __future__ import annotations

import hashlib
import json
import importlib.util
import os
import pickle
import sys
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

EVOLVEMEM = "/home/yoosun/legal/SimpleMem/EvolveMem"
LEGAL_DIR = "/home/yoosun/legal"
sys.path.insert(0, EVOLVEMEM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evolvemem.multi_retriever as MR  # noqa: E402
from evolvemem.multi_retriever import MultiViewIndex, RetrievalConfig  # noqa: E402
from legal_meta import LANG_OF, is_party_name  # noqa: E402
import legal_meta as LM  # noqa: E402

_BM25_LANG = {"zh": "zh_v1", "ko": "ko", "en": "en"}


def _baseline_module():
    """The project's existing BM25 script, reused for tokenisers and metrics."""
    spec = importlib.util.spec_from_file_location(
        "mbr", f"{LEGAL_DIR}/multilingual_bm25_recall.py")
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(LEGAL_DIR)          # its stopword paths are relative
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


MBR = _baseline_module()
_STOPWORD_PATH = {"LeCaRD": "data/LeCaRD/others/stopword.txt",
                  "LeCaRDv2": "data/LeCaRDv2/others/stopword.txt"}


# jieba caches its prefix dict in tempfile.gettempdir(), where /tmp/jieba.cache
# already belongs to another user at mode 0600 -- unreadable and unwritable for
# us. jieba then rebuilt the dict from dict.txt on every process start (0.42s)
# and raised a PermissionError traceback into every log. Segmentation was always
# correct, but that traceback is noise of exactly the kind that hid a real
# BadRequestError once (METHODOLOGY 12), so point jieba at a directory we own.
# Fed to the live tokenizer so its identity lands in the token-cache digest.
# It carries Chinese, Korean and English so the lemmatiser, the noun extractor
# and a regex fallback all produce visibly different output.
# mecab and spaCy fall back to a regex tokenizer when they fail to import,
# printing one warning and carrying on. The closure the baseline module hands
# back names the engine unambiguously, so we can refuse to run on a fallback
# instead of silently producing (and caching) different tokens.
_EXPECTED_ENGINE = {"zh_v1": "jieba", "zh_v2": "jieba",
                    "ko": "mecab.mecab.MeCab",
                    "en": "spacy.lang.en.English"}


# Set by --allow-tokenizer-fallback. Off by default: a degraded tokenizer must
# be chosen, never inherited from a broken import.
ALLOW_TOKENIZER_FALLBACK = False


def _engine_of(tok) -> str:
    if "jieba" in tok.__code__.co_freevars:
        return "jieba"
    for c in tok.__closure__ or ():
        t = type(c.cell_contents)
        if t is not set:
            return f"{t.__module__}.{t.__name__}"
    return "unknown"


def _write_engine_side(side: str, engine: str) -> None:
    try:
        with open(side, "w", encoding="utf-8") as f:
            json.dump({"engine": engine}, f)
    except OSError:
        pass

_JIEBA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".cache", "jieba")


def _set_jieba_cache() -> None:
    import jieba
    os.makedirs(_JIEBA_CACHE, exist_ok=True)
    jieba.dt.tmp_dir = _JIEBA_CACHE


def get_tokenizer(dataset: str):
    lang = LANG_OF[dataset]
    if _BM25_LANG[lang] in ("zh_v1", "zh_v2"):
        _set_jieba_cache()
    cwd = os.getcwd()
    os.chdir(LEGAL_DIR)
    try:
        tok = MBR._get_tokenizer(_BM25_LANG[lang], _STOPWORD_PATH.get(dataset))
    finally:
        os.chdir(cwd)
    want = _EXPECTED_ENGINE[_BM25_LANG[lang]]
    got = _engine_of(tok)
    if got != want and not ALLOW_TOKENIZER_FALLBACK:
        raise RuntimeError(
            f"{dataset}: expected the {want} tokenizer, got {got}. The "
            f"baseline module falls back to a regex tokenizer on import "
            f"failure with only a printed warning, which silently changes "
            f"every BM25 score. Fix the dependency, or pass "
            f"--allow-tokenizer-fallback to accept the fallback knowingly.")
    return tok


# ── Query analysis override ────────────────────────────────────────────────
_CURRENT: dict = {}


def _legal_analyze_query(query: str) -> dict:
    """Replacement for MR.analyze_query, fed by precomputed query metadata.

    `keywords` is handed back as the single untokenised query string; the
    index's keyword_search tokenises it with the language-aware tokeniser.
    """
    row = _CURRENT.get("row") or {}
    return {
        "keywords": [query],
        # Engine-side key names; see "Our vocabulary vs the engine's".
        "persons": parties_of(row),
        "entities": legal_refs_of(row),
        # What the party-strip view deletes. Separate from `persons` on
        # purpose: the metadata view scores party overlap and must keep
        # seeing names only, while the strip list is wider than names.
        "incidentals": incidentals_of(row, query, _CURRENT.get("dataset", "")),
    }



# ── Our vocabulary vs the engine's ────────────────────────────────────────
# EvolveMem's memory dicts carry `persons` and `entities` — names from
# conversational memory, where an "entity" is a topic someone mentioned.
# Ours hold court-case metadata, so this layer speaks its own names and the
# engine's two appear only in the adapters below, which is the single place
# the vocabularies meet.
#
#   parties      당사자 — defendant / victim / plaintiff names  (engine: persons)
#   legal_refs   법조문·죄명 참조                                (engine: entities)
#
# View names follow the same split:
#   metadata view     — scores by shared legal_refs      (engine: "structured")
#   party-strip view  — deletes parties, re-searches     (engine: "entity-swap")
#
# RetrievalConfig knob names keep the engine's spelling because they ARE the
# engine's action space and the evolution loop tunes them by those names:
#   structured_top_k / weight_structured      → metadata view
#   enable_entity_swap / entity_swap_*_k      → party-strip view


def parties_of(row: dict) -> list:
    """Party names, accepting either vocabulary (older memory files)."""
    return list(row.get("parties") or row.get("persons") or [])


_INCIDENTALS: dict = {}

# Ablation switch, not a tuned dimension. The strip list defines what the
# party-strip view *is*, so it stays out of RetrievalConfig: putting it in
# the action space would let one class evolve a different definition of the
# view than another and make the per-class numbers incomparable.
# Default OFF: measured on train (the fold the evolution loop scores on) the
# wider strip list cost -0.0017 / -0.0014 / -0.0002 on LeCaRDv2 / COLIEE /
# LEGAR. Three datasets, all negative. The val numbers that first looked
# positive were small-fold noise -- see METHODOLOGY 16.
STRIP_INCIDENTALS = os.environ.get("STRIP_INCIDENTALS", "0") != "0"


def incidentals_of(row: dict, text: str, dataset: str) -> list:
    """Every case-specific span the party-strip view removes.

    Parties were the original strip list, and they are not the whole of it:
    a judgment also names the company, the bank, the city and the district,
    and those identify this case just as narrowly as a defendant's name does
    while saying nothing about what kind of case it is. The court
    (`location`) joins them here but stays in the metadata view, where the
    same-court signal is useful.

    Cached per query: the evolution loop re-runs every query each round, and
    the spans depend only on the query text.
    """
    if not STRIP_INCIDENTALS:
        return parties_of(row)
    lang = LANG_OF.get(dataset, "zh")
    key = (dataset, str(row.get("query_id", "")) or id(row))
    if key not in _INCIDENTALS:
        spans = [p for p in parties_of(row) if is_party_name(p, lang)]
        spans += LM.extract_incidentals(text, lang)
        loc = str(row.get("location") or "").strip()
        if len(loc) >= 2:
            spans.append(loc)
        seen, out = set(), []
        for sp in spans:
            sp = sp.strip()
            if len(sp) >= 2 and sp not in seen:
                seen.add(sp)
                out.append(sp)
        # Longest first: the spans nest, and removing "北碚区" before
        # "重庆市北碚区" would leave "重庆市" behind as a fragment.
        _INCIDENTALS[key] = sorted(out, key=len, reverse=True)
    return _INCIDENTALS[key]


def legal_refs_of(row: dict) -> list:
    """Statute and charge references, accepting either vocabulary."""
    return list(row.get("legal_refs") or row.get("entities") or [])


def to_engine_keys(cases: list[dict]) -> list[dict]:
    """Repo boundary: expose our fields under the names MultiViewIndex reads.

    Set in place rather than copied — the corpus is 55k dicts and is
    re-indexed every evolution round.
    """
    for m in cases:
        if "persons" not in m:
            m["persons"] = parties_of(m)
        if "entities" not in m:
            m["entities"] = legal_refs_of(m)
    return cases


MR.analyze_query = _legal_analyze_query


@contextmanager
def query_context(row: dict, dataset: str = ""):
    prev, prev_ds = _CURRENT.get("row"), _CURRENT.get("dataset")
    _CURRENT["row"], _CURRENT["dataset"] = row, dataset
    try:
        yield
    finally:
        _CURRENT["row"], _CURRENT["dataset"] = prev, prev_ds


# ── Index ──────────────────────────────────────────────────────────────────
class LegalIndex(MultiViewIndex):
    """MultiViewIndex with language-aware BM25, pool masking and doc ids."""

    CACHE_DIR = "/home/yoosun/legal/evolve_legal/cache"

    def __init__(self, cases: list[dict], dataset: str, embedder=None,
                 metadata_use_parties: bool = False, use_cache: bool = True,
                 embeddings=None, query_embeddings: dict | None = None):
        # Precomputed embeddings are passed in rather than letting the base
        # class encode: the corpus is re-indexed on every evolution round and
        # re-encoding 55k documents each time would dominate the runtime.
        super().__init__(to_engine_keys(cases),
                         None if embeddings is not None else embedder)
        if embeddings is not None:
            self.embeddings = np.asarray(embeddings, dtype=np.float32)
            self.embedder = embedder
        self._qvec = dict(query_embeddings or {})
        self.dataset = dataset
        self.tokenize = get_tokenizer(dataset)
        self.doc_ids = [str(m.get("doc_id") or m.get("memory_id") or i)
                        for i, m in enumerate(cases)]
        self.docid2idx = {d: i for i, d in enumerate(self.doc_ids)}
        self.metadata_use_parties = metadata_use_parties
        self._mask: set[int] | None = None
        self._pool_mask: set[int] | None = None

        # Documents grouped by the penal-code chapter they belong to, for the
        # taxonomy filter. Built once; a query restricts to its own chapter.
        from legal_taxonomy import doc_chapter
        self.doc_chapter = [doc_chapter(m, dataset) for m in cases]
        self.chapter_docs: dict[str, set[int]] = {}
        for i, ch in enumerate(self.doc_chapter):
            self.chapter_docs.setdefault(ch, set()).add(i)

        # Rebuild BM25 over properly tokenised content. Tokenising is by far
        # the slowest step (spaCy over the full COLIEE corpus takes ~14 min),
        # so the token lists are cached against a digest of the inputs.
        self.bm25_corpus = self._tokenised_corpus(use_cache)
        self.bm25 = MBR.FastBM25Okapi(self.bm25_corpus, self.doc_ids)


    def _tokenised_corpus(self, use_cache: bool) -> list[list[str]]:
        if not use_cache:
            return [self.tokenize(c) for c in self.contents]
        h = hashlib.sha1()
        h.update(self.dataset.encode())
        h.update(str(len(self.contents)).encode())
        for c in self.contents:
            h.update(str(len(c)).encode())
        path = os.path.join(self.CACHE_DIR, f"{self.dataset}_{h.hexdigest()[:12]}.pkl")
        # The engine behind the tokenizer is recorded beside the cache rather
        # than mixed into the digest. Mixing it in would make a mismatch look
        # exactly like a cold cache -- the same silent-degradation shape that
        # hid two dead evolution rounds (METHODOLOGY 13) -- and would discard
        # every existing cache for a change that is almost never a mismatch.
        side = path[:-4] + ".engine.json"
        engine = _engine_of(self.tokenize)
        if os.path.exists(path):
            was = None
            if os.path.exists(side):
                try:
                    was = json.load(open(side, encoding="utf-8")).get("engine")
                except (OSError, ValueError):
                    was = None
            if was is not None and was != engine:
                print(f"  [tokenizer] CACHE REJECTED: {os.path.basename(path)} "
                      f"was built with {was}, this run uses {engine}. "
                      f"Re-tokenising.", flush=True)
            else:
                if was is None:          # adopt a cache from before the probe
                    _write_engine_side(side, engine)
                with open(path, "rb") as f:
                    return pickle.load(f)
        corpus = [self.tokenize(c) for c in self.contents]
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(corpus, f, protocol=4)
        _write_engine_side(side, engine)
        return corpus

    # -- pool masking --
    def set_pool(self, doc_ids) -> None:
        """Restrict every view to these documents; None searches the corpus."""
        self._pool_mask = None if doc_ids is None else {
            self.docid2idx[d] for d in map(str, doc_ids) if d in self.docid2idx}
        self._mask = self._pool_mask

    def set_chapter(self, chapter: str | None) -> None:
        """Further restrict to one taxonomy class.

        Falls back to the unrestricted set when the class is unknown or has no
        documents, so a query the taxonomy cannot place is still answered.
        """
        docs = self.chapter_docs.get(chapter) if chapter else None
        if not docs:
            self._mask = self._pool_mask
            return
        self._mask = docs if self._pool_mask is None else (docs & self._pool_mask)
        if not self._mask:
            self._mask = self._pool_mask

    def _emit(self, pairs, top_k):
        if self._mask is not None:
            pairs = [(i, s) for i, s in pairs if i in self._mask]
        return pairs[:top_k]

    # -- views --
    def encode_query(self, query: str):
        v = self._qvec.get(query)
        if v is None:
            v = self.embedder.encode([query], normalize_embeddings=True)[0]
            self._qvec[query] = v
        return np.asarray(v, dtype=np.float32)

    def semantic_search(self, query: str, top_k: int = 20):
        if self.embeddings is None or (self.embedder is None and query not in self._qvec):
            return []
        qe = self.encode_query(query)
        scores = self.embeddings @ qe
        # Over-fetch before masking, or a pool of 100 is usually missed.
        depth = top_k if self._mask is None else max(top_k * 20, 2000)
        order = np.argsort(-scores)[:depth]
        return self._emit([(int(i), float(scores[i])) for i in order
                           if scores[i] > 0.05], top_k)

    def keyword_search(self, keywords: list[str], top_k: int = 8):
        if not keywords:
            return []
        tokens = self.tokenize(" ".join(str(k) for k in keywords))
        if not tokens:
            return []
        if self._mask is not None:
            pool = [self.doc_ids[i] for i in sorted(self._mask)]
            scores, valid = self.bm25.get_scores_subpool(tokens, pool)
            if len(scores) == 0:
                return []
            order = np.argsort(-scores)[:top_k]
            return [(self.docid2idx[valid[j]], float(scores[j])) for j in order
                    if scores[j] > 0]
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(-scores)[:top_k]
        return self._emit([(int(i), float(scores[i])) for i in order
                           if scores[i] > 0], top_k)

    def structured_search(self, persons=None, location=None, entities=None,
                          top_k: int = 5):
        """The metadata view: score by how many legal_refs a document shares.

        Signature keeps the engine's parameter names because the engine calls
        it; `persons`/`entities` here are our parties/legal_refs.

        The base implementation intersects the two and returns a flat 1.0,
        which does not transfer: party names are incidental in case law, so
        an intersection empties the result set, and a constant score gives
        the fusion step nothing to rank on.
        """
        parties, legal_refs = persons, entities
        counts: dict[int, float] = {}
        for e in legal_refs or []:
            for i in self.entity_index.get(str(e).lower(), ()):  # exact
                counts[i] = counts.get(i, 0.0) + 1.0
        if self.metadata_use_parties:
            for p in parties or []:
                for i in self.person_index.get(str(p).lower(), ()):
                    counts[i] = counts.get(i, 0.0) + 0.25
        if location:
            ll = str(location).lower()
            for key, docs in self.location_index.items():
                if ll in key or key in ll:
                    for i in docs:
                        counts[i] = counts.get(i, 0.0) + 0.25
        if not counts:
            return []
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return self._emit(ranked, top_k)

    # -- result mapping --
    def resolve(self, retrieved) -> list[str]:
        """Doc ids of the hits, in rank order."""
        return [r.doc_id for r in retrieved if r.doc_id]


# ── Query rewrite lookup ──────────────────────────────────────────────────
REWRITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "cache", "rewrite")
_REWRITES: dict = {}


def load_rewrites(dataset: str) -> dict:
    """Precomputed LLM rewrites, keyed by query id. Empty when none exist."""
    if dataset not in _REWRITES:
        path = f"{REWRITE_DIR}/{dataset}.json"
        try:
            with open(path, encoding="utf-8") as f:
                _REWRITES[dataset] = json.load(f)
        except (OSError, ValueError):
            _REWRITES[dataset] = {}
    return _REWRITES[dataset]


def expand_query(text: str, query_row: dict, cfg: RetrievalConfig,
                 dataset: str) -> str:
    """Append this query's rewrite when the dimension is on and one exists."""
    if not getattr(cfg, "enable_query_rewrite", False):
        return text
    rw = load_rewrites(dataset).get(str(query_row.get("query_id", "")))
    if not rw:
        return text
    return text + (" " + rw) * max(1, int(getattr(cfg, "query_rewrite_repeat", 1)))


def resolve_config(config: RetrievalConfig, category) -> RetrievalConfig:
    """Apply this query's per-class override and clear the map.

    Clearing the map matters: the engine resolves overrides again from the
    same field, and applying them twice on top of an already-flattened config
    would be a silent no-op at best and wrong when a later recipe edits one
    of the overridden fields.
    """
    ov = (config.per_category_overrides or {}).get(category) \
        or (config.per_category_overrides or {}).get(str(category))
    cfg = replace(config, **{k: v for k, v in ov.items()
                             if hasattr(config, k)}) if ov else config
    return replace(cfg, per_category_overrides={}) if cfg.per_category_overrides else cfg


def search(index: LegalIndex, query_row: dict, config: RetrievalConfig,
           pool=None, category: int | None = None, top_n: int = 100) -> list[str]:
    """Run one query through EvolveMem's retriever and return ranked doc ids.

    Per-class overrides are resolved here, not inside retrieve_multiview,
    because the chapter filter and the query rewrite below need the resolved
    config too. `resolve_config` clears the override map afterwards so the
    engine does not apply it a second time, and the query's real category is
    passed through unchanged.
    """
    cat = query_row.get("category") if category is None else category
    cfg = resolve_config(config, cat)

    index.set_pool(pool)
    index.set_chapter(query_row.get("legal_class") if cfg.enable_chapter_filter else None)
    text = expand_query(query_row["text"], query_row, cfg, index.dataset)
    with query_context(query_row, index.dataset):
        hits = MR.retrieve_multiview(
            index, text, config=cfg, category=cat,
            reference_date=query_row.get("timestamp") or None,
        )
    return index.resolve(hits)[:top_n]


# ── Index construction with the semantic view attached ────────────────────
EMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")


def build_index(dataset: str, cases: list[dict], queries: list[dict] | None = None,
                semantic: bool = True, emb_tag: str = "m3", **kw) -> LegalIndex:
    """LegalIndex with precomputed bge-m3 vectors when they exist on disk.

    Query vectors are loaded too, so evaluation needs no encoder in memory;
    an unseen query would still require one and is handled by encode_query.
    """
    emb = qmap = None
    if semantic:
        mpath = f"{EMB_DIR}/{dataset}_cases_{emb_tag}.npy"
        if os.path.exists(mpath):
            emb = np.load(mpath)
            if len(emb) != len(cases):
                raise ValueError(
                    f"{dataset}: {len(emb)} vectors vs {len(cases)} cases — "
                    "rebuild embeddings after changing the memory file")
        qpath = f"{EMB_DIR}/{dataset}_queries_{emb_tag}.npy"
        if queries and os.path.exists(qpath):
            qv = np.load(qpath)
            qmap = {q["text"]: qv[i] for i, q in enumerate(queries)}
    return LegalIndex(cases, dataset, embeddings=emb, query_embeddings=qmap, **kw)
