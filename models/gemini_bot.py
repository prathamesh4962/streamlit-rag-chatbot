""" Modul pre prácu s modelom Gemini-2.5-Flash-Lite s Pokročilým RAG (Ollama Embedding) """

import google.generativeai as genai
import os
from typing import List
import numpy as np
import ollama

# --- RAG UTILITY (Chunking) ---
import re
from PyPDF2 import PdfReader
import docx
import io

MAX_CHUNK_SIZE = 1000 
TOP_K_CHUNKS = 5
OLLAMA_EMBEDDING_MODEL = 'mxbai-embed-large'


def split_text_into_chunks(text: str) -> List[str]:
    """Rozdelí text na kúsky (chunks) podľa viet s max. veľkosťou 1000."""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text) 
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < MAX_CHUNK_SIZE:
            current_chunk += (sentence + " ").strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def extract_and_chunk_file(file) -> List[str]:
    """Načíta text zo súboru a rozdelí ho na kúsky."""
    # ... (Kód pre PDF, DOCX, TXT extrakciu zostáva rovnaký ako v predchádzajúcej odpovedi) ...
    file_name = file.name or os.path.basename(file.filename)
    text = ""
    try:
        if file_name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        elif file_name.endswith(".pdf"):
            pdf = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif file_name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file.read()))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return []
        return split_text_into_chunks(text)
    except Exception as e:
        print(f"Chyba pri čítaní súboru {file_name}: {e}")
        return []
# --- KONIEC RAG UTILITY ---


# Systémový prompt
SYSTEM_INSTRUCTION = 'You are an intelligent, self-aware and helpful AI assistant called "RAG ChatBot". You prefer to answer in the user\'s language (for example Slovak if the user writes in Slovak). You like natural, concise and relevant conversation.'


class GeminiBot:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = "gemini-2.5-flash-lite"
        self.generation_config = {
            "candidate_count": 1,
            "max_output_tokens": 20000,
            "temperature": 0.8
        }
        
        self.model = genai.GenerativeModel(
            self.model_name, 
            generation_config=self.generation_config,
            system_instruction=SYSTEM_INSTRUCTION
        )
        
        self.chat = self.model.start_chat()
        
        # OllamaEmbedding Vault
        self.vault_chunks = []
        self.vault_embeddings = None # Pole numpy, nie torch tensor


    def reset_rag_vault(self):
        """Vymaže RAG vault pre Gemini."""
        self.vault_chunks = []
        self.vault_embeddings = None
        print("Gemini RAG Vault bol resetovaný.")

    def _get_relevant_context(self, user_input: str) -> List[str]:
        """Vykoná sémantické vyhľadávanie v uložených kúskoch cez Ollama."""
        if self.vault_embeddings is None or len(self.vault_embeddings) == 0:
            return []
            
        try:
            # 1. Získanie embeddingu otázky cez Ollama
            input_embedding_list = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=user_input)["embedding"]
            input_embedding = np.array(input_embedding_list)
        except Exception as e:
            print(f"Chyba pri volaní Ollama Embedding API: {e}")
            return []
            
        # 2. Výpočet kosínusovej podobnosti (použijeme numpy)
        # Normalizácia (nutná pre správny výpočet cos similarity z bodového súčinu)
        input_norm = input_embedding / np.linalg.norm(input_embedding)
        vault_norm = self.vault_embeddings / np.linalg.norm(self.vault_embeddings, axis=1, keepdims=True)
        
        # Kosínusová podobnosť = Bodový súčin (Dot product)
        cos_scores = np.dot(vault_norm, input_norm)
        
        # 3. Získanie Top-K indexov
        top_k = min(TOP_K_CHUNKS, len(cos_scores))
        top_indices = np.argsort(cos_scores)[::-1][:top_k] # Získanie indexov v zostupnom poradí

        # 4. Získanie relevantného kontextu
        relevant_context = [self.vault_chunks[idx].strip() for idx in top_indices]
        return relevant_context

    def ask(self, prompt_text: str, uploaded_files=None, history=None):
        """
        Generuje odpoveď pomocou Gemini modelu s Pokročilým RAG.
        Využíva sémantické vyhľadávanie a len relevantný kontext.
        """
        if not prompt_text and not uploaded_files:
            return "Zadaj text alebo prilož súbor."
        
        # 1. ČÍTANIE SÚBOROV a RE-INDEXOVANIE
        if uploaded_files:
            new_chunks = []
            for file in uploaded_files:
                new_chunks.extend(extract_and_chunk_file(file))
            
            if new_chunks:
                self.vault_chunks = new_chunks
                print(f"Gemini: Generujem embeddingy pre {len(self.vault_chunks)} kúskov cez Ollama...")
                
                # Získanie embeddingov pre všetky kúsky v dávkach (pre rýchlosť)
                all_embeddings = []
                for chunk in self.vault_chunks:
                     embedding_list = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=chunk)["embedding"]
                     all_embeddings.append(embedding_list)
                     
                self.vault_embeddings = np.array(all_embeddings)
                print("Embeddingy hotové. RAG Vault pripravený.")
            else:
                 return "Súbor bol nahratý, ale neobsahoval žiadny text."
        
        # 2. VYHĽADÁVANIE KONTEXTU
        relevant_context = self._get_relevant_context(prompt_text)
        
        files_content = ""
        if relevant_context:
            context_str = "\n---\n".join(relevant_context)
            files_content += f"\n\n--- KONTEXT Z DOKUMENTOV (RAG) ---\n{context_str}\n\n"

        # Zostavenie user promptu (s RAG kontextom)
        full_user_prompt = prompt_text
        if files_content:
            full_user_prompt = f"Používateľova otázka:\n{prompt_text}\n{files_content}"

        try:
            # VŽDY voláme send_message, ktorá automaticky pripojí históriu
            response = self.chat.send_message(full_user_prompt)
            
            return response.text.strip()
        
        except Exception as e:
            return f"Chyba pri generovaní odpovede (Gemini): {e}"
