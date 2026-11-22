"""Encode the corpus with a multilingual embedder for EvolveMem's semantic view.

EvolveMem defaults to BAAI/bge-base-en-v1.5, which is English-only and useless
on the Chinese and Korean benchmarks here. BAAI/bge-m3 covers all three
languages with one model, so a single configuration applies across datasets and
an evolved setting can transfer between them.

Embeddings are written once to disk; the index loads them instead of re-encoding,
since the evolution loop re-reads the same corpus every round.
"""
from __future__ import annotations
import argparse, datetime as _dt, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legal_meta import LANG_OF

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


def main():
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


if __name__ == "__main__":
    main()
