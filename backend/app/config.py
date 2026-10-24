"""Central settings. Everything tunable lives here so decisions are easy to find."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
LOG_DIR = ROOT / "logs"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
FAISS_PATH = INDEX_DIR / "faiss.index"
EVAL_LOG = LOG_DIR / "ragas_log.jsonl"

# Chunking: all-MiniLM-L6-v2 only reads the first 256 word pieces (~1000 characters),
# so chunks are sized to fit that window instead of the 500-800 token range in the guide.
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150  # 15%
MIN_CHUNK_CHARS = 60

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_K = 20   # candidates from FAISS
RERANK_K = 5      # kept after re-ranking
# Cross-encoder logit below which we refuse (ms-marco scores are unbounded). Calibrated 2026-10-24 on 17 in-scope
# questions (lowest best score +2.28) and 8 out-of-scope ones incl. near-domain food questions (highest -3.35).
REFUSE_BELOW = float(os.getenv("REFUSE_BELOW", "-1.0"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# llama-3.3-70b-versatile was retired by Groq on 2026-08-16; gpt-oss-120b is its production replacement.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")        # answers
GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b")  # cheap router calls
# The RAGAS judge makes many calls per query, so it uses the smaller model: Groq's free daily token limit is per model,
# and this keeps scoring from exhausting the model that answers questions (a 429 on 2026-10-24 came from judge traffic).
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")
RAGAS_LIVE = os.getenv("RAGAS_LIVE", "1") != "0"  # set RAGAS_LIVE=0 to stop per-query scoring (e.g. if the hosted quota runs low)

# Friendly document names used in citations, keyed by a phrase of the file name.
DOC_NAMES = {
    "healthy_gut": "Healthy Gut, Healthy You (FSSAI, 2023)",
    "dgi": "Dietary Guidelines for Indians (ICMR-NIN, 2024)",
}
DOC_KIND = {"healthy_gut": "recipe", "dgi": "nutrition"}
