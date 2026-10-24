"""Talks to the pipeline either over HTTP (BACKEND_URL set) or in-process (default; what the hosted demo uses,
because Streamlit Community Cloud runs only the Streamlit app)."""
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")


def mode() -> str:
    return f"HTTP backend at {BACKEND_URL}" if BACKEND_URL else "in-process (no separate server)"


def _service():
    from backend.app import main  # reuses the same RagService instance the API uses
    return main


def index_ready() -> bool:
    from backend.app import config
    if BACKEND_URL:
        import requests
        try:
            return bool(requests.get(f"{BACKEND_URL}/health", timeout=5).json().get("index_ready"))
        except Exception:
            return False
    return config.FAISS_PATH.exists()


def upload_pdf(name: str, data: bytes) -> dict:
    if BACKEND_URL:
        import requests
        r = requests.post(f"{BACKEND_URL}/upload", files={"file": (name, data, "application/pdf")}, timeout=600)
        r.raise_for_status()
        return r.json()
    from backend.app import config
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_DIR / Path(name).name).write_bytes(data)
    return {"saved": name, **_service().rebuild_index()}


def ask(question: str) -> dict:
    if BACKEND_URL:
        import requests
        r = requests.post(f"{BACKEND_URL}/ask", json={"question": question}, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()
    from backend.app import evaluation
    m = _service()
    result = m._service.ask(question)
    if result["refused"]:
        evaluation.evaluate_and_log(result)
    else:
        threading.Thread(target=evaluation.evaluate_and_log, args=(result,), daemon=True).start()
    return {k: result[k] for k in ("answer", "refused", "citations", "route", "top_rerank_score", "trace")}


def eval_rows(n: int = 10) -> list[dict]:
    if BACKEND_URL:
        import requests
        try:
            return requests.get(f"{BACKEND_URL}/eval/log", params={"n": n}, timeout=10).json()
        except Exception:
            return []
    from backend.app import evaluation
    return evaluation.read_log(n)
