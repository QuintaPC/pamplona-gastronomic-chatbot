"""Run: python -m streamlit run app.py"""
import os
import streamlit as st
from dotenv import load_dotenv
from rag_core import load_documents, make_chunks, make_store, answer_question, EMBEDDING_MODEL

load_dotenv(override=True)
st.set_page_config(page_title="Pamplona Gastronomic Assistant", page_icon="🍽️")
st.title("🍽️ Pamplona Gastronomic Assistant")
st.caption("A document-grounded restaurant guide • AI Technology Final Project")

@st.cache_resource(show_spinner="Reading and indexing restaurant PDFs…")
def prepare():
    pages, pdf_count = load_documents()
    chunks = make_chunks(pages, chunk_size=750, overlap=120)
    return make_store(chunks), pdf_count, len(chunks)

with st.sidebar:
    st.header("How it works")
    st.write("PDFs → 750-character chunks (120 overlap) → multilingual embeddings → Chroma → top-4 retrieval → LLM")
    st.caption("Embedding model: " + EMBEDDING_MODEL)
    st.warning("Information is based on the supplied PDFs, not live booking or price data.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

try:
    store, count, chunks = prepare()
    st.success(f"Loaded {count} restaurant PDFs and {chunks} text chunks")
except Exception as exc:
    st.error(f"Could not initialize the knowledge base: {type(exc).__name__}: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("evidence"):
            with st.expander("Source excerpts"):
                for e in message["evidence"]:
                    st.markdown(f"**[{e['id']}] {e['file']} — page {e['page']}**")
                    st.write(e["text"][:800])

question = st.chat_input("Ask about restaurants, cuisine, atmosphere, or follow-up questions…")
if question:
    history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking the restaurant documents…"):
                response, evidence = answer_question(store, question, history, os.getenv("GOOGLE_API_KEY", ""))
        except Exception as exc:
            response, evidence = (f"I couldn't complete the request ({type(exc).__name__}). Please check the API connection and try again.", [])
        st.markdown(response)
        if evidence:
            with st.expander("Source excerpts"):
                for e in evidence:
                    st.markdown(f"**[{e['id']}] {e['file']} — page {e['page']}**")
                    st.write(e["text"][:800])
    st.session_state.messages.append({"role": "assistant", "content": response, "evidence": evidence})
