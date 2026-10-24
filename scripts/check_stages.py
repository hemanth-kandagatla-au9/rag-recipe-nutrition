"""Stage-by-stage self check to run on YOUR machine (the build sandbox could not download models or reach Groq).
Prints PASS/FAIL per check and keeps going, then a summary. Run from the project root:
    python scripts/check_stages.py
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import config  # noqa: E402

results = []


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            msg = fn() or ""
            results.append((name, True, msg))
            print(f"PASS  {name}  {msg}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            results.append((name, False, repr(e)))
            print(f"FAIL  {name}: {e!r}")
            traceback.print_exc(limit=3)
        return fn
    return deco


from backend.app import index as idx, ingestion  # noqa: E402

state = {}


@check("A  PDFs present and chunked")
def _():
    n = len(list(config.RAW_DIR.glob("*.pdf")))
    assert n >= 2, f"found {n} PDFs in data/raw/"
    chunks = ingestion.ingest_all()
    assert len(chunks) > 400
    return f"{len(chunks)} chunks"


@check("B1 embeddings + FAISS index build")
def _():
    emb = idx.Embedder()
    state["emb"] = emb
    idx.build_index(idx.load_chunks(), emb)
    state["retr"] = idx.Retriever(emb)
    return f"{state['retr'].index.ntotal} vectors"


@check("B2 retrieval finds the right recipe, deterministically")
def _():
    r = state["retr"]
    a, b = r.search("How much protein is in Mishti Doi?", 20), r.search("How much protein is in Mishti Doi?", 20)
    assert [h["id"] for h in a] == [h["id"] for h in b]
    top = [h["section"] for h in a[:5]]
    assert any("Mishti Doi" in s for s in top), f"top-5 sections: {top}"
    return f"top-5: {top}"


@check("C1 cross-encoder re-ranking + refusal threshold calibration")
def _():
    from backend.app.rerank import Reranker
    rr = Reranker()
    state["rr"] = rr
    inq = ["How much protein is in Mishti Doi?", "Which food groups make up a balanced diet?", "How does frying affect nutrients?"]
    outq = ["What is the capital of France?", "Who won the 2011 cricket world cup?", "How do I fix a flat bicycle tyre?"]
    best = lambda q: rr.rerank(q, state["retr"].search(q, config.RETRIEVE_K), config.RERANK_K)[0]["rerank_score"]
    s_in, s_out = [best(q) for q in inq], [best(q) for q in outq]
    print("     in-scope best scores :", [round(s, 2) for s in s_in])
    print("     out-of-scope best    :", [round(s, 2) for s in s_out])
    print(f"     REFUSE_BELOW is {config.REFUSE_BELOW}: it must sit between max(out) and min(in)")
    assert min(s_in) > max(s_out), "in-scope and out-of-scope scores overlap: send me these numbers"
    return f"gap {min(s_in) - max(s_out):.2f}; suggested REFUSE_BELOW ~ {(min(s_in) + max(s_out)) / 2:.1f}"


@check("C2 Groq key + LLM answer with citations")
def _():
    assert config.GROQ_API_KEY, "GROQ_API_KEY missing in .env"
    from backend.app.chain import RagService
    from backend.app.index import Retriever
    svc = RagService(state["retr"], state["rr"])
    state["svc"] = svc
    r = svc.ask("How much protein is in Mishti Doi?")
    print("     answer:", r["answer"][:200].replace("\n", " "))
    print("     route :", r["route"], "citations:", r["citations"])
    assert not r["refused"] and r["citations"], "expected an answer with citations"
    return ""


@check("C3 out-of-scope question is refused")
def _():
    r = state["svc"].ask("What is the capital of France?")
    assert r["refused"], f"expected refusal, got: {r['answer'][:120]}"
    return f"top score {r['top_rerank_score']}"


@check("C4 multi-part question is decomposed")
def _():
    r = state["svc"].ask("Which has more energy per serving, Madra or Kesar Piyush?")
    print("     route:", r["route"], "\n     answer:", r["answer"][:200].replace("\n", " "))
    assert len(r["route"]["sub_questions"]) >= 1
    assert not r["refused"]


@check("D  RAGAS scores one answer")
def _():
    from backend.app import evaluation
    r = state["svc"].ask("What are the ingredients of Panchamirtham?")
    rec = evaluation.evaluate_and_log(r)
    print("     record:", {k: rec[k] for k in ("faithfulness", "answer_relevancy", "context_precision", "status")})
    assert rec["status"] == "ok" and rec["faithfulness"] is not None, rec["status"]


print("\n=== SUMMARY ===")
for n, ok, m in results:
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"  -> {m}"))
sys.exit(0 if all(o for _, o, _ in results) else 1)
