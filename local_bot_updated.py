"""Local RAG chatbot backed by Ollama.

Supports TXT, PDF, and DOCX uploads. The local Ollama models used are:
- llama3.2 for chat responses
- mxbai-embed-large for document/query embeddings
"""
from __future__ import annotations

import io
import os
import re
from typing import Any

import numpy as np
import ollama
from docx import Document
from PyPDF2 import PdfReader


CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

SYSTEM_PROMPT = """You are SaSW RAG ChatBot, a helpful and accurate AI assistant.
Answer in the same language as the user's question unless they request another language.
Be concise, clear, and factual. Do not invent sources, people, dates, or facts.

Important terminology:
- Generative AI (GenAI) means AI that generates new content, such as text, images,
  audio, video, or code, based on patterns learned from data.
- Artificial General Intelligence (AGI) is a hypothetical AI with broad, human-like
  ability across many different tasks. Do not confuse AGI with Generative AI.
- Do not claim that Generative AI means "General Artificial Intelligence."
- Do not claim that you are constantly learning from this conversation. You are an
  AI assistant powered by a locally running language model.

For questions about uploaded documents, prioritize the relevant document context.
If the context does not contain the answer, say that the uploaded documents do not
provide enough information, then optionally give a clearly labelled general answer.
"""

class LocalRAGBot:
    """A small retrieval-augmented chatbot using local Ollama models."""

    def __init__(self, model_path: str = CHAT_MODEL):
        # Keep model_path for compatibility with the previous app.py interface.
        self.chat_model = model_path if model_path and not model_path.lower().endswith(".gguf") else CHAT_MODEL
        self.embedding_model = EMBED_MODEL
        self.chunks: list[str] = []
        self.embeddings: list[np.ndarray] = []

    def reset_rag_vault(self) -> None:
        """Clear documents from the current RAG vault."""
        self.chunks = []
        self.embeddings = []

    @staticmethod
    def _read_uploaded_file(uploaded_file: Any) -> str:
        name = getattr(uploaded_file, "name", "uploaded_file")
        raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        ext = os.path.splitext(name)[1].lower()

        if ext == ".txt":
            for encoding in ("utf-8", "utf-8-sig", "cp1252"):
                try:
                    return raw.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if ext == ".docx":
            document = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in document.paragraphs)
        raise ValueError(f"Unsupported file type: {ext}. Upload TXT, PDF, or DOCX.")

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                split_at = text.rfind(" ", start + int(chunk_size * 0.65), end)
                if split_at > start:
                    end = split_at
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _embed(self, text: str) -> np.ndarray:
        # Current Ollama Python clients support embed(); older ones support embeddings().
        if hasattr(ollama, "embed"):
            result = ollama.embed(model=self.embedding_model, input=text)
            vectors = result.get("embeddings", [])
            if vectors:
                return np.asarray(vectors[0], dtype=np.float32)
        result = ollama.embeddings(model=self.embedding_model, prompt=text)
        return np.asarray(result["embedding"], dtype=np.float32)

    def _add_file(self, uploaded_file: Any) -> None:
        text = self._read_uploaded_file(uploaded_file)
        new_chunks = self._chunk_text(text)
        if not new_chunks:
            raise ValueError(f"No readable text was found in {getattr(uploaded_file, 'name', 'the file')}.")
        for chunk in new_chunks:
            self.chunks.append(chunk)
            self.embeddings.append(self._embed(chunk))

    def _retrieve(self, query: str, top_k: int = 4) -> list[str]:
        if not query.strip() or not self.chunks or not self.embeddings:
            return []
        query_vector = self._embed(query)
        matrix = np.vstack(self.embeddings)
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
        scores = np.divide(
            matrix @ query_vector,
            denom,
            out=np.zeros(len(self.chunks), dtype=np.float32),
            where=denom != 0,
        )
        indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[int(i)] for i in indices if scores[int(i)] > 0]

    def ask(self, user_text: str, uploaded_files: list[Any] | None = None,
            history: list[dict[str, Any]] | None = None) -> str:
        uploaded_files = uploaded_files or []
        history = history or []

        for uploaded_file in uploaded_files:
            self._add_file(uploaded_file)

        context_chunks = self._retrieve(user_text)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include a short conversation window, excluding UI-only upload notices.
        for item in history[-8:]:
            role = item.get("role")
            content = item.get("content", "")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            if content.startswith("📎 Files added for document search:"):
                continue
            if content.strip():
                messages.append({"role": role, "content": content[:5000]})

        if context_chunks:
            context = "\n\n---\n\n".join(context_chunks)
            user_content = (
                f"Question: {user_text or 'Summarize the uploaded document.'}\n\n"
                f"Relevant uploaded-document context:\n{context}\n\n"
                "Answer the question using this context when relevant. If the context "
                "does not answer it, clearly say so."
            )
        elif uploaded_files and not user_text:
            user_content = "Summarize the uploaded document using the available document context."
        else:
            user_content = user_text

        if not user_content.strip():
            user_content = "Please summarize the uploaded document."

        messages.append({"role": "user", "content": user_content})
        try:
            result = ollama.chat(
                model=self.chat_model,
                messages=messages,
                options={"temperature": 0.3},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not get a response from Ollama model '{self.chat_model}'. "
                "Check that Ollama is running and the model is installed "
                f"(ollama list). Details: {exc}"
            ) from exc
        return result.get("message", {}).get("content", "").strip() or "I couldn't generate a response. Please try again."
