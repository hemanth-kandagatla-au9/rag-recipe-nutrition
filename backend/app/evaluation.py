"""RAGAS scoring (faithfulness, answer relevancy, context precision) + JSONL log.

Reference-free metrics, so no ground-truth answers are needed per query. The judge is a Groq model via LangChain;
embeddings for answer-relevancy are the same local MiniLM. Scoring never raises: a failure is logged as status 'error'
so a RAGAS hiccup can never break an answer. Groq's free tier is tight on tokens/minute, so one worker + retries.
"""
import functools
import json
import threading
import time

from . import config

_lock = threading.Lock()


def append_log(record: dict, path=None):
    path = path or config.EVAL_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(n: int = 50, path=None) -> list[dict]:
    path = path or config.EVAL_LOG
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows[-n:]


@functools.lru_cache(maxsize=1)
def _judge_and_embeddings():
    """Built once per process: loading the embedding model again for every query would be slow and memory-hungry."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from .llm import make_llm

    return (LangchainLLMWrapper(make_llm(config.GROQ_JUDGE_MODEL, max_tokens=3000)),
            LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)))


def score_with_ragas(question: str, answer: str, contexts: list[str]) -> dict:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
    from ragas.run_config import RunConfig

    judge, emb = _judge_and_embeddings()
    data = EvaluationDataset.from_list([{"user_input": question, "response": answer, "retrieved_contexts": contexts}])
    res = evaluate(
        data,
        metrics=[Faithfulness(), ResponseRelevancy(strictness=1),  # Groq accepts only n=1 completions
                 LLMContextPrecisionWithoutReference()],
        llm=judge, embeddings=emb, run_config=RunConfig(max_workers=1, timeout=180, max_retries=6),
        show_progress=False, raise_exceptions=False)
    row = res.to_pandas().iloc[0].to_dict()

    def pick(prefix):
        for k, v in row.items():
            if k.startswith(prefix) and v == v:  # v == v filters NaN
                return round(float(v), 4)
        return None

    return {"faithfulness": pick("faithfulness"), "answer_relevancy": pick("answer_relevancy"),
            "context_precision": pick("llm_context_precision")}


def evaluate_and_log(result: dict) -> dict:
    """Score one /ask result and append a line to the JSONL log. Refused answers are logged but not scored."""
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": result["question"],
           "answer_preview": result["answer"][:200], "refused": result["refused"],
           "route": result["route"].get("type"), "faithfulness": None, "answer_relevancy": None,
           "context_precision": None, "status": "skipped_refused" if result["refused"] else "ok"}
    if not result["refused"] and not config.RAGAS_LIVE:
        rec["status"] = "skipped_live_scoring_off"
    elif not result["refused"]:
        try:
            rec.update(score_with_ragas(result["question"], result["answer"], result["contexts"]))
        except Exception as e:  # never break the app because of scoring
            rec["status"] = f"error: {type(e).__name__}: {str(e)[:160]}"
    append_log(rec)
    return rec
