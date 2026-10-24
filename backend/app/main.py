"""FastAPI backend. /health is instant (no model loading); models load on the first /upload or /ask."""
import shutil
import threading

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import config, evaluation
from .chain import RagService

app = FastAPI(title="Recipe & Nutrition RAG Assistant")
_service = RagService()
_build_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    evaluate: bool = True  # score with RAGAS in the background


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": config.FAISS_PATH.exists(), "groq_key_set": bool(config.GROQ_API_KEY)}


@app.get("/documents")
def documents():
    return {"pdfs": sorted(p.name for p in config.RAW_DIR.glob("*.pdf")),
            "chunks": sum(1 for _ in open(config.CHUNKS_PATH, encoding="utf-8")) if config.CHUNKS_PATH.exists() else 0}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Save a PDF to data/raw/, then re-ingest everything and rebuild the FAISS index."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.RAW_DIR / file.filename.replace("/", "_").replace("\\", "_")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"saved": dest.name, **rebuild_index()}


def rebuild_index() -> dict:
    from .index import Embedder, build_index
    from .ingestion import ingest_all

    with _build_lock:
        chunks = ingest_all()
        build_index(chunks, _service.retriever.embedder if _service._retriever else Embedder())
        _service.reload()
    return {"chunks": len(chunks)}


@app.post("/ask")
def ask(req: AskRequest, background: BackgroundTasks):
    if not config.FAISS_PATH.exists():
        raise HTTPException(409, "No index yet. Upload the PDFs first (POST /upload) or run scripts/build_index.py.")
    try:
        result = _service.ask(req.question)
    except RuntimeError as e:  # e.g. missing GROQ_API_KEY
        raise HTTPException(503, str(e))
    if req.evaluate and not result["refused"]:
        background.add_task(evaluation.evaluate_and_log, result)
    elif req.evaluate:
        evaluation.evaluate_and_log(result)  # logs the refusal immediately (no scoring needed)
    return {k: result[k] for k in ("answer", "refused", "citations", "route", "top_rerank_score", "trace")} | {
        "eval": "scoring in background; see GET /eval/log" if req.evaluate and not result["refused"] else "not scored"}


@app.get("/eval/log")
def eval_log(n: int = 20):
    return evaluation.read_log(n)
