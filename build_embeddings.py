"""Encode the corpus with a multilingual embedder for EvolveMem's semantic view.

EvolveMem defaults to BAAI/bge-base-en-v1.5, which is English-only and useless
on the Chinese and Korean benchmarks here. BAAI/bge-m3 covers all three
languages with one model, so a single configuration applies across datasets and
an evolved setting can transfer between them.

Embeddings are written once to disk; the index loads them instead of re-encoding,
since the evolution loop re-reads the same corpus every round.
"""
from __future__ import annotations
import argparse, json, os, sys, time

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
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
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
            rows = json.load(open(f"{CASE_DIR}/{ds}_{tag}.json", encoding="utf-8"))
            texts = [to_text(r) for r in rows]
            t0 = time.time()
            vecs = encode(model, texts, tag, a.batch_size, a.model)
            path = f"{EMB}/{ds}_{tag}_{a.tag}.npy"
            np.save(path, vecs.astype(np.float16))
            print(f"[{ds}/{tag}] {len(texts)} texts, {time.time()-t0:.0f}s, "
                  f"dim={vecs.shape[1]} -> {path} "
                  f"({os.path.getsize(path)/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
