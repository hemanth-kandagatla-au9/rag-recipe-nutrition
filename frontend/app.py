import os

import streamlit as st

# Hosted deployment: copy Streamlit secrets into the environment BEFORE the backend reads its config.
try:
    for _k in ("GROQ_API_KEY", "GROQ_MODEL", "BACKEND_URL"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

import client  # noqa: E402  (frontend/client.py)

st.set_page_config(page_title="Recipe & Nutrition Assistant", page_icon="🥣")
st.title("Recipe & Nutrition Assistant")
st.caption(
    "Answers from two documents: *Healthy Gut, Healthy You* (FSSAI, 2023) for recipes and per-serving nutrition, and "
    "*Dietary Guidelines for Indians* (ICMR-NIN, 2024) for general nutrition guidance. Every answer shows the document "
    "and PDF page. Not medical advice; if the documents don't contain the answer it says so."
)

with st.sidebar:
    st.write("**Mode:**", client.mode())
    up = st.file_uploader("Add a PDF to the knowledge base", type="pdf")
    if up and st.button("Ingest this PDF"):
        with st.spinner("Parsing, chunking, embedding and indexing..."):
            try:
                st.success(f"Indexed. {client.upload_pdf(up.name, up.getvalue())}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

if not client.index_ready():
    st.warning("No index yet. Upload the two PDFs in the sidebar, or run `python scripts/build_index.py`.")
    st.stop()

examples = ["How much protein is in Mishti Doi?", "What are the ingredients of Panchamirtham?",
            "Which has more energy, Madra or Kesar Piyush?", "How much salt should I eat per day?",
            "What is the capital of France?"]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, key=f"ex{i}"):
        st.session_state["q"] = ex

q = st.text_input("Ask a question", key="q")
if st.button("Ask") and q.strip():
    with st.spinner("Retrieving, re-ranking and answering..."):
        try:
            res = client.ask(q.strip())
        except Exception as e:
            st.error(f"Could not answer: {e}")
            st.stop()
    if res["refused"]:
        st.info(res["answer"])
    else:
        st.markdown(res["answer"])
        st.markdown("**Sources**")
        for c in res["citations"]:
            st.markdown(f"- {c['document']}, PDF page {c['pdf_page']}, {c['section']}")
    with st.expander("How this was found (routing and re-ranking)"):
        st.write("Route:", res["route"])
        st.write("Best re-rank score:", res["top_rerank_score"])
        c1, c2 = st.columns(2)
        c1.write("Before re-ranking (vector search order)")
        c1.dataframe(res["trace"]["before_rerank"], hide_index=True)
        c2.write("After re-ranking (cross-encoder order)")
        c2.dataframe(res["trace"]["after_rerank"], hide_index=True)

with st.expander("Recent RAGAS scores (scored in the background, refresh to see new rows)"):
    rows = client.eval_rows(10)
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.write("No scored queries yet.")
