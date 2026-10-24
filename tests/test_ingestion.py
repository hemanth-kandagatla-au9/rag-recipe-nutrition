"""Stage A checks. Run: python -m pytest tests/test_ingestion.py  (needs the two PDFs in data/raw/)."""
import pytest

from backend.app import config, ingestion

pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
pytestmark = pytest.mark.skipif(len(pdfs) < 2, reason="put both PDFs in data/raw/")


@pytest.fixture(scope="module")
def chunks():
    return ingestion.ingest_all()


def test_every_chunk_has_citation_metadata(chunks):
    for c in chunks:
        assert c["doc_name"] and c["source"] and isinstance(c["page"], int) and c["section"] and c["text"].strip()


def test_recipe_nutrition_is_one_clean_chunk(chunks):
    n = [c for c in chunks if c["type"] == "nutrition"]
    assert len(n) >= 24
    panch = [c for c in n if "Panchamirtham" in c["section"]]
    assert panch and "Energy 429 Kcals" in panch[0]["text"] and "Protein 3.18 g" in panch[0]["text"]


def test_chunk_sizes_fit_embedding_window(chunks):
    big = [c for c in chunks if len(c["text"]) > 1300 and c["type"] != "table"]
    assert not big


def test_tables_are_markdown(chunks):
    t = [c for c in chunks if c["type"] == "table"]
    assert t and all("| --- |" in c["text"] for c in t)


def test_ingestion_is_deterministic(chunks):
    again = ingestion.ingest_all()
    assert [c["text"] for c in again] == [c["text"] for c in chunks]
