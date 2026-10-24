"""Cross-encoder re-ranking: reads (question, chunk) together, so it judges relevance far better than the
bi-encoder used for the first-pass search, but is too slow to run over everything - hence retrieve 20, re-rank to 5."""
from . import config


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.model_name, device="cpu")
            # This checkpoint's safetensors header leaves every tensor 2 bytes off a 4-byte boundary. The weights
            # are memory-mapped zero-copy, and Apple Accelerate crashes (SIGBUS) or returns NaN/garbage on
            # misaligned floats. Copying them into freshly allocated (aligned) memory fixes it; 90 MB, done once.
            for p in model.model.parameters():
                p.data = p.data.clone()
            for b in model.model.buffers():
                b.data = b.data.clone()
            self._model = model
        return self._model

    def rerank(self, query: str, candidates: list[dict], top_k: int = config.RERANK_K) -> list[dict]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates], show_progress_bar=False)
        out = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
        out.sort(key=lambda c: (-c["rerank_score"], c["retrieval_rank"]))  # tie-break keeps results deterministic
        for r, c in enumerate(out, 1):
            c["rerank_rank"] = r
        return out[:top_k]
