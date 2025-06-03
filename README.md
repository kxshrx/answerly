# Answerly

A Chrome extension and backend service that lets you ask questions about the content of any webpage. It uses advanced AI models from Hugging Face and Google Gemini to extract, embed, and answer questions based on the live content of the page you are viewing.

---

## Features

- **Ask questions about any webpage** directly from your browser.
- Uses **Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`** for semantic embeddings.
- Uses **Google Gemini (`gemini-2.0-flash-lite`)** via LangChain for answer generation.
- FastAPI backend for robust API serving.
- No premium API keys required (results can be improved with access to premium Gemini models).

---

## Tech Stack

- **LangChain** for orchestration and chaining
- **Google Gemini (`gemini-2.0-flash-lite`)** via `langchain-google-genai`
- **Hugging Face Embeddings** via `langchain-huggingface`
- **ChromaDB** for vector storage
- **FastAPI** for backend API
- **Chrome Extension** (Manifest V3)
- **Python** (see `requirements.txt` for all dependencies)

---

## How to Use

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd web-qa-extension
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up your environment:**
   - Copy `.env` and add your Google API key (for Gemini). The free tier works, but premium Gemini models will give better results.

4. **Start the backend server:**
   ```sh
   uvicorn backend:app --reload
   ```

5. **Load the Chrome extension:**
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" and select the `chrome-extension` folder.

6. **Use the extension:**
   - Open any webpage.
   - Click the extension icon.
   - Enter your question and click "Ask".
   - The answer will appear in the popup, based on the content of the current page.

---

## Main Files

- `main.py`: Core logic for loading, chunking, embedding, and answering questions.
- `backend.py`: FastAPI backend exposing the `/ask` endpoint.
- `requirements.txt`: All Python dependencies.
- `chrome-extension/`: Chrome extension UI and logic.

Other files are for component testing and experimentation.

---

## Notes

- **Premium Gemini models:** This project uses the free tier of Gemini (`gemini-2.0-flash-lite`). If you have access to premium Gemini models, you can update the model name in the code for improved results.
- **No data is stored:** All processing is done locally and temporarily for each session.

---

**Built with Hugging Face, Google Gemini, LangChain, and FastAPI.**