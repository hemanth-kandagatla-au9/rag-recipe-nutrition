"""Stage C: the RAG chain.

question -> route/decompose -> retrieve (FAISS top-20 per sub-question) -> cross-encoder re-rank (top-5)
         -> refuse if nothing relevant -> prompt + Groq LLM -> answer with citations.

The composition is a LangChain Expression Language (LCEL) chain:
    retrieve_stage | assign(answer = generate)
"""
import json
import re
import time
from pathlib import Path

from . import config
from .index import Embedder, Retriever
from .rerank import Reranker
from .router import route_question

NOT_FOUND = "I could not find this in the provided documents."

SYSTEM_PROMPT = f"""You are a Recipe & Nutrition assistant. Answer ONLY from the numbered context blocks.
Rules:
- Cite every fact with the block number in square brackets, like [1] or [2][3].
- Copy numbers (calories, grams, quantities) exactly as written and say they are per serving when the context says so.
- If the context does not contain the answer, reply exactly: {NOT_FOUND}
- Do not give medical advice, diet plans for health conditions or allergy-safety guarantees; say a doctor or dietitian is the right person for that.
- Do not estimate nutrition for dishes the context does not cover. Keep answers short and plain."""

HUMAN_PROMPT = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def normalize_citations(answer: str) -> str:
    """gpt-oss models write citations as 【1】 or 【1†L3-L5】; turn them into plain [1] so they can be parsed and read."""
    return re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", answer)


def format_context(docs: list[dict]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[{i}] ({d['doc_name']}, PDF page {d['page']}, {d['section']})\n{d['text']}")
    return "\n\n".join(blocks)


def used_citations(answer: str, docs: list[dict]) -> list[dict]:
    """Citations for the blocks the answer actually referenced; if it referenced none, all supplied blocks."""
    idx = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(docs)})
    chosen = [docs[i - 1] for i in idx] or docs
    seen, out = set(), []
    for d in chosen:
        key = (d["doc"], d["page"], d["section"])
        if key not in seen:
            seen.add(key)
            out.append({"document": d["doc_name"], "pdf_page": d["page"], "section": d["section"]})
    return out


def merge_candidates(per_sub: list[list[dict]], limit: int = 6) -> list[dict]:
    """Interleave the re-ranked lists of each sub-question, dropping duplicates, so every part of a
    multi-part question gets evidence."""
    out, seen = [], set()
    for rank in range(max((len(l) for l in per_sub), default=0)):
        for lst in per_sub:
            if rank < len(lst) and lst[rank]["id"] not in seen:
                seen.add(lst[rank]["id"])
                out.append(lst[rank])
    return out[:limit]


def compact(hits: list[dict], score_key: str) -> list[dict]:
    return [{"id": h["id"], "page": h["page"], "section": h["section"], "doc": h["doc"],
             "score": round(h.get(score_key, 0.0), 4)} for h in hits]


class RagService:
    def __init__(self, retriever: Retriever | None = None, reranker: Reranker | None = None, llm=None,
                 router_llm=None, trace_path: Path = config.LOG_DIR / "retrieval_trace.jsonl"):
        self._retriever, self._reranker, self._llm, self._router_llm = retriever, reranker, llm, router_llm
        self.trace_path = trace_path
        self._chain = None

    # lazily created heavy parts
    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = Retriever(Embedder())
        return self._retriever

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            from .llm import make_answer_llm
            self._llm = make_answer_llm()
        return self._llm

    @property
    def router_llm(self):
        if self._router_llm is None:
            from .llm import make_llm
            try:
                self._router_llm = make_llm(config.GROQ_ROUTER_MODEL, max_tokens=600)
            except RuntimeError:
                self._router_llm = False  # no key: router falls back to rules
        return self._router_llm or None

    def reload(self):
        """Call after the index was rebuilt (e.g. after an upload)."""
        self._retriever = None

    # ---- stage 1: route + retrieve + re-rank ----
    def retrieve_stage(self, question: str) -> dict:
        t0 = time.time()
        route = route_question(question, self.router_llm)
        kind = None if route["source"] == "both" else route["source"]
        subs = route["sub_questions"]
        per_sub, first_pass = [], []
        keep = 3 if len(subs) > 1 else config.RERANK_K
        for sq in subs:
            cands = self.retriever.search(sq, config.RETRIEVE_K, kind)
            ranked = self.reranker.rerank(sq, cands, keep)
            first_pass.append(cands[:keep])
            per_sub.append(ranked)
        docs = merge_candidates(per_sub, limit=max(config.RERANK_K, 6 if len(subs) > 1 else config.RERANK_K))
        top = max((d["rerank_score"] for d in docs), default=float("-inf"))
        trace = {"question": question, "route": route,
                 "before_rerank": compact([h for l in first_pass for h in l], "score"),
                 "after_rerank": compact(docs, "rerank_score"),
                 "seconds_retrieval": round(time.time() - t0, 2)}
        return {"question": question, "route": route, "docs": docs, "context": format_context(docs),
                "top_score": top, "refuse": top < config.REFUSE_BELOW, "trace": trace}

    # ---- stage 2: generate (LCEL) ----
    @property
    def chain(self):
        if self._chain is None:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableLambda, RunnablePassthrough

            prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
            llm_chain = prompt | self.llm | StrOutputParser()

            def generate(x: dict) -> str:
                if x["refuse"]:
                    return NOT_FOUND
                return normalize_citations(llm_chain.invoke({"context": x["context"], "question": x["question"]}).strip())

            self._chain = RunnableLambda(self.retrieve_stage) | RunnablePassthrough.assign(answer=RunnableLambda(generate))
        return self._chain

    def ask(self, question: str) -> dict:
        out = self.chain.invoke(question)
        answer = out["answer"]
        refused = out["refuse"] or answer.startswith(NOT_FOUND[:30])
        result = {"question": question, "answer": answer, "refused": refused,
                  "citations": [] if refused else used_citations(answer, out["docs"]),
                  "route": out["route"],
                  "top_rerank_score": round(out["top_score"], 3) if out["top_score"] > float("-inf") else None,
                  "contexts": [d["text"] for d in out["docs"]], "trace": out["trace"]}
        self._log_trace(result)
        return result

    def _log_trace(self, result: dict):
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "refused": result["refused"],
                                    **result["trace"]}, ensure_ascii=False) + "\n")
        except OSError:
            pass
