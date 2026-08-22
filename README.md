# Streamlit RAG ChatBot

A multi-model, document-aware chat application built with Streamlit. Upload TXT, PDF, or DOCX files, ask questions about their contents, and choose between a cloud model (Gemini) and locally executed GGUF language models.

The application implements a lightweight Retrieval-Augmented Generation (RAG) pipeline: uploaded documents are split into chunks, embedded locally with Ollama, searched semantically with cosine similarity, and the most relevant chunks are supplied to the selected language model as context.

## Features

- Multi-session chat interface: create, rename, switch, clear, and remove chats.
- Per-chat model selection.
- Document ingestion for `.txt`, `.pdf`, and `.docx` files.
- Sentence-aware chunking with a maximum chunk size of 1,000 characters.
- Local semantic retrieval using Ollama embeddings and cosine similarity.
- Top 5 relevant chunks added to the model context for each question.
- Cloud model support through the Gemini API.
- Local GGUF model support through `llama-cpp-python`.
- Chat history formatted as ChatML for local models.
- Resettable in-memory RAG vault when chats or models change.

## Architecture

```text
                   ┌───────────────────────────────┐
                   │       Streamlit Interface     │
                   │  chats · model selector · UI  │
                   └───────────────┬───────────────┘
                                   │
                 Upload TXT / PDF / DOCX and ask a question
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │      Document Processing      │
                   │ text extraction + chunking    │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │        Local RAG Vault        │
                   │ chunks + Ollama embeddings    │
                   │ mxbai-embed-large             │
                   └───────────────┬───────────────┘
                                   │
                         cosine similarity search
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │      Top-K Relevant Chunks    │
                   │          K = 5                │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
          ┌────────────────────────┴────────────────────────┐
          │                                                 │
          ▼                                                 ▼
┌───────────────────────┐                       ┌────────────────────────┐
│ Gemini 2.5 Flash-Lite │                       │ Local GGUF model       │
│ via Gemini API        │                       │ via llama-cpp-python   │
└───────────────────────┘                       └────────────────────────┘
          │                                                 │
          └────────────────────────┬────────────────────────┘
                                   ▼
                         Context-aware answer
```

## Supported Models

| Model | Execution mode | Purpose |
|---|---|---|
| Gemini 2.5 Flash-Lite | Cloud API | Fast hosted model through the Gemini API |
| Gemma 3 4B Instruct | Local GGUF | Higher-capability local option |
| Gemma 3 1B Instruct | Local GGUF | Lightweight local option |
| EuroLLM 1.7B Instruct | Local GGUF | Compact multilingual local option |
| mxbai-embed-large | Local Ollama | Embedding model used for retrieval |

## Project Structure

```text
src/
├── app.py
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── gemini_bot.py
│   └── local_bot.py
├── GGUF models/
│   ├── gemma-3-4b-it-Q8_0.gguf
│   ├── gemma-3-1b-it-Q8_0.gguf
│   └── EuroLLM-1.7B-Instruct.Q4_K_S.gguf
└── .streamlit/
    └── secrets.toml
```

The `GGUF models/` directory is not included in the repository because model files are large. Download compatible GGUF files separately and update the paths in `app.py` if needed.

## Requirements

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running locally.
- An Ollama embedding model: `mxbai-embed-large`.
- A Gemini API key if you want to use Gemini.
- Local GGUF model files if you want to use the local-model options.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/miRx923/streamlit-rag-chatbot.git
cd streamlit-rag-chatbot
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

Install the dependencies from a `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Depending on your operating system and hardware, `llama-cpp-python` may require a platform-specific installation command for CPU, CUDA, Metal, or Vulkan support. See the library documentation if the default installation fails.

### 4. Install the embedding model

Start Ollama, then download the embedding model:

```bash
ollama pull mxbai-embed-large
```

Verify that it is available:

```bash
ollama list
```

### 5. Configure Gemini (optional)

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

Never commit this file. Add it to `.gitignore`:

```gitignore
.streamlit/secrets.toml
```

If you only use local models, you can omit the Gemini key and avoid selecting Gemini in the interface.

### 6. Add local GGUF models (optional)

Create the directory expected by `app.py`:

```text
GGUF models/
```

Place compatible model files inside it. The current code expects:

```text
GGUF models/gemma-3-4b-it-Q8_0.gguf
GGUF models/gemma-3-1b-it-Q8_0.gguf
GGUF models/EuroLLM-1.7B-Instruct.Q4_K_S.gguf
```

Alternatively, update these paths in `app.py` to match your files.

## Run the Application

With the virtual environment activated and Ollama running:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## How the RAG Pipeline Works

1. A user uploads one or more TXT, PDF, or DOCX documents.
2. The application extracts the textual content from each file.
3. Text is normalized and split into sentence-aware chunks of up to 1,000 characters.
4. The application generates an embedding for each chunk through the local Ollama embedding model.
5. Chunks and their embeddings are stored in an in-memory RAG vault for the active chat.
6. When a question is submitted, the application embeds the question.
7. Cosine similarity ranks all stored chunks against the question embedding.
8. The five highest-scoring chunks are injected into the prompt as context.
9. Gemini or the selected local GGUF model produces the final answer.

The RAG vault is intentionally in-memory. It is reset when chat history is cleared, a chat is changed or removed, or the selected model changes.

## Example Use Case

1. Upload a product price list in PDF format.
2. Ask a question such as:

   ```text
   What is the price of the Liftback 2.0 TDI 110 kW (150 hp) Sportline version?
   ```

3. The application retrieves chunks that are semantically closest to the question and uses them as context for the answer.

## Local Inference Notes

The local models run via `llama-cpp-python` and use GGUF model files. In the current implementation, local inference is configured with:

```python
n_gpu_layers=0
n_ctx=5000
```

This means inference is configured for CPU execution. If you have a compatible GPU build of `llama-cpp-python`, you can experiment with GPU offloading by changing `n_gpu_layers` to `-1` or to a suitable positive number. The optimal setting depends on available VRAM, model size, quantization, and your operating system.

## Limitations

- The RAG vault is held only in memory and is not persisted between application restarts.
- Uploading a new document replaces the existing RAG vault for the active bot rather than appending to it.
- PDF extraction quality depends on whether the document contains selectable text; scanned PDFs may require OCR.
- No source citations, reranking, vector database, authentication, access control, or production deployment layer is included.
- Local-model speed and quality depend strongly on hardware and the selected GGUF quantization.
- This application should not be used with confidential personal, medical, legal, or business data without appropriate security and privacy controls.

## Security

- Keep `.streamlit/secrets.toml` private.
- Do not commit API keys or downloaded GGUF model files.
- Treat uploaded documents as untrusted input.
- Avoid uploading sensitive data to cloud models such as Gemini unless you have assessed the applicable privacy and compliance requirements.
