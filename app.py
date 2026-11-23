"""Run: python -m streamlit run app.py"""
import os
import streamlit as st
from dotenv import load_dotenv
from rag_core import load_documents, make_chunks, make_store, answer_question, EMBEDDING_MODEL

load_dotenv(override=True)

st.markdown("""
<style>
.stApp {
    background-color: #F5F0E7;
    color: #33252A;
}

/* Main content and header */
[data-testid="stHeader"] {
    background-color: #F5F0E7;
}

[data-testid="stBottom"] {
    background-color: #F5F0E7;
}


/* Navarra Gourmet chat input */
[data-testid="stBottom"] {
    background-color: #F5F0E7 !important;
}

[data-testid="stBottom"] > div {
    background-color: #F5F0E7 !important;
}

[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1px solid #B89560 !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"] > div {
    background-color: #FFFFFF !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #33252A !important;
    -webkit-text-fill-color: #33252A !important;
    caret-color: #651D32 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #82767A !important;
    opacity: 1 !important;
}


/* Main text */
[data-testid="stMain"] {
    color: #33252A;
}

[data-testid="stSidebar"] {
    background-color: #651D32;
    color: #F5F0E7;
}

[data-testid="stSidebar"] * {
    color: #F5F0E7;
}

.gourmet-hero {
    background: linear-gradient(135deg, #651D32, #421321);
    padding: 42px 38px;
    border-radius: 18px;
    margin-bottom: 26px;
    border-bottom: 4px solid #B89560;
}

.gourmet-label {
    color: #DCC5A0;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 600;
}

.gourmet-title {
    color: #FFFFFF;
    font-size: 43px;
    font-weight: 700;
    line-height: 1.15;
    margin: 16px 0;
}

.gourmet-subtitle {
    color: #E9DCD7;
    font-size: 16px;
    line-height: 1.6;
}
</style>

<div class="gourmet-hero">
    <div class="gourmet-label">NAVARRA · GASTRONOMY · DISCOVERY</div>
    <div class="gourmet-title">Discover Pamplona,<br>one restaurant at a time.</div>
    <div class="gourmet-subtitle">
        Your personal guide to the flavors, traditions and
        culinary experiences of Pamplona.
    </div>
</div>
""", unsafe_allow_html=True)

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
    
    st.markdown(
        f"""
        <div style="
            background-color: #FFFDF9;
            border: 1px solid #D9C8AC;
            border-left: 4px solid #B89560;
            border-radius: 12px;
            padding: 18px 22px;
            margin: 16px 0 24px 0;
            color: #651D32;
        ">
            <div style="
                font-size: 12px;
                letter-spacing: 2px;
                font-weight: 700;
                color: #9A7650;
            ">
                YOUR GASTRONOMIC GUIDE IS READY
            </div>
            <div style="
                font-size: 18px;
                font-weight: 600;
                margin-top: 8px;
            ">
                🍷 {count} restaurants to discover
            </div>
            <div style="
                font-size: 13px;
                color: #746B68;
                margin-top: 4px;
            ">
                Explore Pamplona's culinary scene, powered by {chunks} curated text excerpts.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
