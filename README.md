# Pamplona Gastronomic Assistant

This Streamlit interface uses the same RAG settings and prompt as `Copia_de_Chatbot_1.ipynb`.

## Setup

1. Install Python 3.10+ (3.11 recommended).
2. In a terminal, navigate to this folder and run:

   ```bash
   python -m pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env`, and replace the placeholder with your real Google Gemini API key. **Do not share or commit `.env`.**
4. Put the 20 PDF files in a folder called `Restaurant docs` or `restaurant_docs` beside `app.py`. Alternatively, place `restaurant_docs.zip` beside `app.py` (with one of those folder names inside the ZIP).
5. Run:

   ```bash
   python -m streamlit run app.py
   ```

The first launch may download the multilingual embedding model and take a few minutes. Streamlit displays a local URL, usually `http://localhost:8501`.

## Notes

- The original notebook uses the model identifier `gemini-3.5-flash`. This project keeps that exact identifier. If the API reports that it is unavailable, change `GEMINI_MODEL` in `rag_core.py` to a model supported by your key. This is not a verified live API test.
- The app uses local session memory, Chroma, four retrieved chunks, and source excerpts.
- No PDF files or credentials are included in this package.
