"""Parse + chunk the PDFs in data/raw/ and build the FAISS index. Run from the project root:
    python scripts/build_index.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import config, index, ingestion  # noqa: E402

t0 = time.time()
pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
if not pdfs:
    sys.exit(f"No PDFs in {config.RAW_DIR}. Put the two PDFs there first.")
print("PDFs:", [p.name for p in pdfs])
chunks = ingestion.ingest_all()
print(f"{len(chunks)} chunks -> {config.CHUNKS_PATH}")
index.build_index(chunks, index.Embedder())
print(f"FAISS index -> {config.FAISS_PATH}  ({time.time() - t0:.0f}s)")
