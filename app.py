# app.py — Minimal Streamlit UI
import streamlit as st
from generation import answer
from retrieval import retrieve_with_scores

st.set_page_config(page_title="Melbourne CBD Restaurant Selector", page_icon="🍜", layout="wide")
st.title("🍜 Melbourne CBD Restaurant Selector ")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K to retrieve", 1, 8, 5)
    min_hybrid = st.slider("Min hybrid score (confidence gate)", 0.0, 1.0, 0.3, 0.05)
    st.caption("If the top score is below this, the bot will answer: “I don’t know.”")

# Main input
query = st.text_input("What are you craving?", placeholder="e.g., Cheap Japanese near Southern Cross")

# Run
if st.button("Search") and query.strip():
    with st.spinner("Thinking…"):
        try:
            response = answer(query.strip(), k=k, min_hybrid=min_hybrid)
            st.markdown(response)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    # (Optional) show the retrieved candidates (to understand sources)
    with st.expander("See retrieved sources"):
        rows = retrieve_with_scores(query.strip(), top_k=k)
        for _, r in rows.iterrows():
            st.markdown(f"- **{r['name']}** — {r['amenity']}  ·  score: {r['hybrid_score']:.2f}")
            st.caption(r["kb_text"])

st.markdown("---")
st.caption("Local RAG: BM25 + embeddings → grounded answer via Ollama. No external APIs.")

