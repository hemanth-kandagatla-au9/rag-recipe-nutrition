"""Stage B/C plumbing checks with fakes. Run: python -m pytest tests/test_pipeline_logic.py"""
import tempfile
from pathlib import Path

from backend.app import chain, config, index, ingestion
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker, install_fake_faiss, install_fake_langchain_if_missing

install_fake_faiss()
install_fake_langchain_if_missing()


def _service(tmp: Path, llm=None):
    chunks = ingestion.ingest_all() if config.RAW_DIR.glob("*.pdf") else []
    emb = FakeEmbedder()
    index.build_index(chunks, emb, tmp / "f.index")
    # chunks.jsonl already written by ingest_all; Retriever reads both
    retr = index.Retriever(emb, config.CHUNKS_PATH, tmp / "f.index")
    return chain.RagService(retr, FakeReranker(), llm or FakeLLM(), FakeLLM(), trace_path=tmp / "t.jsonl")


def test_index_matches_chunks_and_search_is_deterministic():
    with tempfile.TemporaryDirectory() as d:
        svc = _service(Path(d))
        a = svc.retriever.search("protein in Mishti Doi", 20)
        b = svc.retriever.search("protein in Mishti Doi", 20)
        assert [h["id"] for h in a] == [h["id"] for h in b] and len(a) == 20
        assert any("Mishti Doi" in h["section"] for h in a[:5])


def test_kind_filter_restricts_source_with_fallback():
    with tempfile.TemporaryDirectory() as d:
        svc = _service(Path(d))
        hits = svc.retriever.search("daily salt intake guideline", 20, kind="nutrition")
        assert all(h["kind"] == "nutrition" for h in hits)


def test_answerable_question_cites_page_and_recipe():
    with tempfile.TemporaryDirectory() as d:
        svc = _service(Path(d))
        r = svc.ask("How much protein is in Mishti Doi?")
        assert not r["refused"] and r["citations"]
        assert r["trace"]["before_rerank"] and r["trace"]["after_rerank"]


def test_out_of_scope_question_is_refused_without_calling_llm():
    with tempfile.TemporaryDirectory() as d:
        llm = FakeLLM()
        svc = _service(Path(d), llm)
        r = svc.ask("zzqx blorptastic quantum flux capacitor")
        assert r["refused"] and r["answer"] == chain.NOT_FOUND and r["citations"] == [] and llm.calls == 0


def test_merge_candidates_interleaves_and_dedupes():
    a = [{"id": "a1"}, {"id": "a2"}]
    b = [{"id": "b1"}, {"id": "a1"}]
    assert [x["id"] for x in chain.merge_candidates([a, b])] == ["a1", "b1", "a2"]


def test_used_citations_only_referenced_blocks():
    docs = [{"doc": "x", "doc_name": "X", "page": 1, "section": "s1"}, {"doc": "x", "doc_name": "X", "page": 2, "section": "s2"}]
    assert [c["pdf_page"] for c in chain.used_citations("Yes [2].", docs)] == [2]
    assert len(chain.used_citations("No brackets", docs)) == 2


def test_eval_log_records_refusals_and_survives_scoring_errors(monkeypatch=None):
    from backend.app import evaluation
    with tempfile.TemporaryDirectory() as d:
        old = config.EVAL_LOG
        config.EVAL_LOG = Path(d) / "log.jsonl"
        try:
            base = {"question": "q?", "answer": "a", "route": {"type": "simple"}, "contexts": ["c"]}
            r1 = evaluation.evaluate_and_log({**base, "refused": True})
            assert r1["status"] == "skipped_refused"
            # scoring needs ragas + a Groq key; without them it must log an error, never raise
            r2 = evaluation.evaluate_and_log({**base, "refused": False})
            assert r2["status"] == "ok" or r2["status"].startswith("error")
            rows = evaluation.read_log(10)
            assert len(rows) == 2 and {"timestamp", "question", "faithfulness", "answer_relevancy", "context_precision"} <= set(rows[0])
        finally:
            config.EVAL_LOG = old


def test_fullwidth_citation_markers_are_normalised_and_parsed():
    ans = chain.normalize_citations("22 g per serving【2】 and more【1†L3-L5】.")
    assert ans == "22 g per serving[2] and more[1]."
    docs = [{"doc": "x", "doc_name": "X", "page": 1, "section": "s1"}, {"doc": "x", "doc_name": "X", "page": 2, "section": "s2"},
            {"doc": "x", "doc_name": "X", "page": 3, "section": "s3"}]
    assert [c["pdf_page"] for c in chain.used_citations(ans, docs)] == [1, 2]


def test_rate_limit_detection_and_live_scoring_switch():
    from backend.app import evaluation, llm

    class RateLimitError(Exception):
        pass

    assert llm.is_rate_limit(RateLimitError("x")) and llm.is_rate_limit(Exception("Error code: 429 - tokens per day"))
    assert not llm.is_rate_limit(ValueError("something else"))
    with tempfile.TemporaryDirectory() as d:
        old_log, old_live = config.EVAL_LOG, config.RAGAS_LIVE
        config.EVAL_LOG, config.RAGAS_LIVE = Path(d) / "l.jsonl", False
        try:
            rec = evaluation.evaluate_and_log({"question": "q?", "answer": "a", "route": {"type": "simple"},
                                               "contexts": ["c"], "refused": False})
            assert rec["status"] == "skipped_live_scoring_off"
        finally:
            config.EVAL_LOG, config.RAGAS_LIVE = old_log, old_live
