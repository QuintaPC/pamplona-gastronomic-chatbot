"""RAG logic adapted from Copia_de_Chatbot_1.ipynb."""
from pathlib import Path
import zipfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import torch
torch.backends.mkldnn.enabled = False

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
GEMINI_MODEL = 'gemini-3.5-flash'  # Same identifier as the original notebook; must be available for your API key.

SYSTEM_PROMPT = """You are a helpful restaurant guide for Pamplona.
Only use the PDF extracts below to answer questions about restaurants.
Do not invent facts or use outside information.
If the documents do not contain the answer, say: "I cannot confirm that from the provided restaurant documents."
If the question is unrelated to the restaurants, politely say that you only answer questions about these restaurants.
Do not claim real-time opening hours, prices or reservation availability.
When possible, mention the source document in your answer.
Use the same language as the user's question.
The chat history can help you understand a follow-up, but it is not a source of restaurant facts.
"""
PROMPT = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    ('human', 'Recent conversation:\n{history}\n\nPDF extracts:\n{context}\n\nQuestion: {question}')
])


def load_documents():
    """Read restaurant PDFs from either folder name or the notebook's ZIP."""
    base = Path(__file__).resolve().parent
    folders = [base / 'Restaurant docs', base / 'restaurant_docs']
    pdf_files = []
    for folder in folders:
        if folder.is_dir():
            pdf_files = sorted(folder.glob('*.pdf'))
            if pdf_files:
                break
    if not pdf_files:
        archive = base / 'restaurant_docs.zip'
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                # Prevent extraction outside the project folder.
                for entry in z.infolist():
                    target = (base / entry.filename).resolve()
                    if not target.is_relative_to(base.resolve()):
                        raise ValueError('Unsafe path in restaurant_docs.zip')
                z.extractall(base)
            for folder in folders:
                pdf_files = sorted(folder.glob('*.pdf')) if folder.is_dir() else []
                if pdf_files:
                    break
    if not pdf_files:
        raise FileNotFoundError('Add the 20 PDFs in "Restaurant docs" / "restaurant_docs", or place restaurant_docs.zip next to app.py.')
    documents = []
    for pdf in pdf_files:
        pages = PyPDFLoader(str(pdf)).load()
        for page in pages:
            page.metadata['restaurant'] = pdf.name
        documents.extend(pages)
    return documents, len(pdf_files)


def make_chunks(pages, chunk_size=750, overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(pages)


def make_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    return Chroma.from_documents(chunks, embedding=embeddings, collection_name='pamplona_guide')


def answer_question(store, question, history, api_key):
    """Return (answer, evidence) using the same retrieval and prompt as the notebook."""
    if not question.strip():
        return 'Please write a question.', []
    previous_question = next((m['content'] for m in reversed(history) if m['role'] == 'user'), '')
    search_question = previous_question + ' ' + question if previous_question else question
    found_docs = store.similarity_search(search_question, k=4)
    evidence = [
        {'id': i, 'file': d.metadata.get('restaurant', 'unknown'),
         'page': d.metadata.get('page', 0) + 1, 'text': d.page_content}
        for i, d in enumerate(found_docs, 1)
    ]
    if not found_docs:
        return 'I cannot confirm that from the provided restaurant documents.', evidence
    if not api_key:
        raise ValueError('GOOGLE_API_KEY is missing. Add it to your .env file.')
    context = '\n\n'.join(f"[{e['id']}] {e['file']}, page {e['page']}: {e['text']}" for e in evidence)
    history_text = '\n'.join(f"{m['role']}: {m['content']}" for m in history[-6:])
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0)
    result = (PROMPT | llm).invoke({'history': history_text, 'context': context, 'question': question})
    answer = result.content
    if isinstance(answer, list):
        answer = '\n'.join(item.get('text', '') for item in answer if isinstance(item, dict) and item.get('type') == 'text')
    return str(answer), evidence
