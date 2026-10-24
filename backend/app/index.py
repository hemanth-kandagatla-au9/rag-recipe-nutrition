"""Stage B: embeddings + FAISS index + retrieval.

IndexFlatIP over L2-normalised vectors = exact cosine search. It is exact (no approximation), so the same
question always returns the same top-k, and at ~600 chunks it is instant.
"""
import json
from pathlib import Path

import numpy as np

from . import config


def _import_faiss():
    """faiss and torch each ship their own libomp. On macOS, if faiss loads its copy first, the first
    SentenceTransformer/CrossEncoder forward pass segfaults (exit 139). Loading torch first avoids it."""
    try:
        import torch  # noqa: F401
    except ImportError:  # tests run with stand-ins and may not have torch
        pass
    import faiss

    return faiss


class Embedder:
    """Lazy wrapper so importing the app does not load the model until first use."""

    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_index(chunks: list[dict], embedder: Embedder, index_path: Path = config.FAISS_PATH):
    faiss = _import_faiss()

    vecs = embedder.encode([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    return index


class Retriever:
    def __init__(self, embedder: Embedder | None = None, chunks_path: Path = config.CHUNKS_PATH,
                 index_path: Path = config.FAISS_PATH):
        faiss = _import_faiss()

        self.embedder = embedder or Embedder()
        self.chunks = load_chunks(chunks_path)
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index has {self.index.ntotal} vectors but chunks.jsonl has {len(self.chunks)} rows; rebuild with scripts/build_index.py")

    def search(self, query: str, k: int = config.RETRIEVE_K, kind: str | None = None) -> list[dict]:
        """Top-k chunks by cosine similarity. `kind` ('recipe'/'nutrition') softly restricts the source;
        if the restriction leaves fewer than 5 hits we fall back to all documents."""
        q = self.embedder.encode([query])
        fetch = min(len(self.chunks), k * 5 if kind else k)
        scores, ids = self.index.search(q, fetch)
        hits = [{**self.chunks[i], "score": float(s), "retrieval_rank": 0}
                for s, i in zip(scores[0], ids[0]) if i >= 0]
        if kind:
            filtered = [h for h in hits if h["kind"] == kind]
            hits = filtered if len(filtered) >= 5 else hits
        hits = hits[:k]
        for r, h in enumerate(hits, 1):
            h["retrieval_rank"] = r
        return hits
