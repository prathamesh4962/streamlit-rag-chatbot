
"""Gemini-powered RAG chatbot for Streamlit Cloud."""

import io
import os
import re
from typing import List

import numpy as np
from google import genai
from PyPDF2 import PdfReader
import docx


MAX_CHUNK_SIZE = 1000
TOP_K_CHUNKS = 5
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash-lite"

SYSTEM_INSTRUCTION = (
    'You are a helpful AI assistant called "RAG ChatBot". '
    "Answer in the same language as the user's question. "
    "When answering questions about uploaded documents, prioritize "
    "their content and do not invent information."
)


def split_text_into_chunks(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= MAX_CHUNK_SIZE:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def extract_and_chunk_file(file) -> List[str]:
    file_name = getattr(file, "name", "uploaded_file")
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    extension = os.path.splitext(file_name)[1].lower()

    try:
        if extension == ".txt":
            text = raw.decode("utf-8", errors="replace")
        elif extension == ".pdf":
            pdf = PdfReader(io.BytesIO(raw))
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
        elif extension == ".docx":
            document = docx.Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in document.paragraphs)
        else:
            return []

        return split_text_into_chunks(text)

    except Exception as exc:
        print(f"Could not read {file_name}: {exc}")
        return []


class GeminiBot:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.vault_chunks = []
        self.vault_embeddings = None

    def _embed(self, text: str) -> np.ndarray:
        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        if not result.embeddings or not result.embeddings[0].values:
            raise ValueError("The embedding API returned no vector.")

        return np.asarray(
            result.embeddings[0].values,
            dtype=np.float32,
        )

    def reset_rag_vault(self):
        self.vault_chunks = []
        self.vault_embeddings = None

    def _get_relevant_context(self, user_input: str) -> List[str]:
        if self.vault_embeddings is None or not self.vault_chunks:
            return []

        try:
            query_vector = self._embed(user_input)
            query_norm = np.linalg.norm(query_vector)

            if query_norm == 0:
                return []

            chunk_norms = np.linalg.norm(
                self.vault_embeddings,
                axis=1,
                keepdims=True,
            )

            normalized_chunks = self.vault_embeddings / np.maximum(
                chunk_norms, 1e-12
            )

            scores = normalized_chunks @ (query_vector / query_norm)
            top_indices = np.argsort(scores)[::-1][:TOP_K_CHUNKS]

            return [self.vault_chunks[i] for i in top_indices]

        except Exception as exc:
            print(f"Embedding/search error: {exc}")
            return []

    def ask(self, prompt_text: str, uploaded_files=None, history=None):
        if not prompt_text and not uploaded_files:
            return "Please enter a question or upload a document."

        if uploaded_files:
            new_chunks = []

            for uploaded_file in uploaded_files:
                new_chunks.extend(extract_and_chunk_file(uploaded_file))

            if not new_chunks:
                return "The uploaded files contained no readable text."

            try:
                vectors = [self._embed(chunk) for chunk in new_chunks]
                self.vault_chunks = new_chunks
                self.vault_embeddings = np.vstack(vectors)
            except Exception as exc:
                return f"Could not index the uploaded documents: {exc}"

        relevant_context = self._get_relevant_context(prompt_text)

        if relevant_context:
            context = "\n---\n".join(relevant_context)
            full_prompt = (
                f"User question:\n{prompt_text}\n\n"
                "Relevant excerpts from uploaded documents:\n"
                f"{context}\n\n"
                "Answer using the excerpts when relevant. If they do not "
                "contain the answer, say so clearly."
            )
        else:
            full_prompt = prompt_text

        try:
            response = self.client.models.generate_content(
                model=CHAT_MODEL,
                contents=full_prompt,
                config={
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                },
            )

            return (
                (response.text or "").strip()
                or "No text response returned."
            )

        except Exception as exc:
            return f"Gemini response error: {exc}"
