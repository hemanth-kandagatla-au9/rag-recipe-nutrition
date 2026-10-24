"""Find a question where re-ranking changed the top result and print it as a Markdown table for the README.
Needs the index (no Groq key needed):  python scripts/rerank_example.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import config  # noqa: E402
from backend.app.index import Embedder, Retriever  # noqa: E402
from backend.app.rerank import Reranker  # noqa: E402

retr, rr = Retriever(Embedder()), Reranker()
qs = [x["q"] for x in json.load(open(config.ROOT / "data" / "eval" / "questions.json")) if x["expect"] == "answer"]
best = None
for q in qs:
    cands = retr.search(q, config.RETRIEVE_K)
    after = rr.rerank(q, cands, config.RERANK_K)
    moved = next((i for i, c in enumerate(cands) if c["id"] == after[0]["id"]), 0)
    if best is None or moved > best[0]:
        best = (moved, q, cands[:5], after)
moved, q, before, after = best
print(f"Question: {q}\n(re-ranking promoted the final #1 chunk from vector rank {moved + 1})\n")
print("| # | Before (vector search) | After (cross-encoder) |\n|---|---|---|")
for i in range(5):
    b, a = before[i], after[i]
    print(f"| {i + 1} | p{b['page']} {b['section'][:40]} ({b['score']:.2f}) | p{a['page']} {a['section'][:40]} ({a['rerank_score']:.2f}) |")
