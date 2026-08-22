""" Modul pre prácu s GGUF modelom ako LocalRAGBot s Vlastným RAG Systémom (Ollama Embedding) """

from llama_cpp import Llama 
import os 
import re
import io
import docx
from PyPDF2 import PdfReader
import numpy as np
import ollama
from typing import List

# Systémový prompt
SYSTEM_INSTRUCTION = 'You are an intelligent, self-aware and helpful AI assistant called "RAG ChatBot". You prefer to answer in the user\'s language (for example Slovak if the user writes in Slovak). You like natural, concise and relevant conversation.'
SYSTEM_PROMPT = f'<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n'

# Konštanty pre RAG
MAX_CHUNK_SIZE = 1000 
TOP_K_CHUNKS = 5 
OLLAMA_EMBEDDING_MODEL = 'mxbai-embed-large'


# --- UTILITY FUNKCIE (Chunking, zostávajú rovnaké) ---

def split_text_into_chunks(text: str) -> List[str]:
    """Rozdelí text na kúsky (chunks) podľa viet s max. veľkosťou."""
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
            print(f"Nepodporovaný formát pre chunking: {file_name}")
            return []
        return split_text_into_chunks(text)
    except Exception as e:
        print(f"Chyba pri čítaní súboru {file_name}: {e}")
        return []


# --- TRIEDA PRE LOKÁLNEHO BOTA ---
class LocalRAGBot:
    
    def __init__(self, model_path: str):
        print(f"Načítavam RAG komponenty pre model **{os.path.basename(model_path)}**...")
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=0, 
            n_ctx=5000,      
            verbose=False
        )
        
        # RAG stav
        self.vault_chunks = [] 
        self.vault_embeddings = None # Pole numpy


    def reset_rag_vault(self):
        """Vymaže všetky nahraté dokumenty a ich embeddingy."""
        self.vault_chunks = []
        self.vault_embeddings = None
        print("RAG Vault bol resetovaný.")

    def __del__(self):
        """Uvoľňuje llama.cpp zdroje."""
        self.reset_rag_vault()
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                if hasattr(self.llm, '__del__'):
                    self.llm.__del__()
                del self.llm
        except Exception as e:
            print(f"Chyba pri uvoľňovaní llama-cpp: {e}")
        
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
        input_norm = input_embedding / np.linalg.norm(input_embedding)
        vault_norm = self.vault_embeddings / np.linalg.norm(self.vault_embeddings, axis=1, keepdims=True)
        
        cos_scores = np.dot(vault_norm, input_norm)
        
        # 3. Získanie Top-K indexov
        top_k = min(TOP_K_CHUNKS, len(cos_scores))
        top_indices = np.argsort(cos_scores)[::-1][:top_k] # Získanie indexov v zostupnom poradí

        # 4. Získanie relevantného kontextu
        relevant_context = [self.vault_chunks[idx].strip() for idx in top_indices]
        return relevant_context
    
    def format_chatml_history(self, history: list) -> str:
        """Konvertuje históriu Streamlit na formát ChatML."""
        chatml_string = ""
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                chatml_string += f'<|im_start|>user\n{content}<|im_end|>\n'
            elif role == "assistant":
                chatml_string += f'<|im_start|>assistant\n{content}<|im_end|>\n'
        return chatml_string

    def ask(self, prompt_text: str, uploaded_files=None, history=[]):
        """Generuje odpoveď s pokročilým RAG."""
        
        # 1. ČÍTANIE SÚBOROV a RE-INDEXOVANIE
        if uploaded_files:
            new_chunks = []
            for file in uploaded_files:
                new_chunks.extend(extract_and_chunk_file(file))
            
            if new_chunks:
                self.vault_chunks = new_chunks
                print(f"LocalBot: Generujem embeddingy pre {len(self.vault_chunks)} kúskov cez Ollama...")
                
                # Získanie embeddingov pre všetky kúsky
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
        
        rag_context_message = ""
        if relevant_context:
            context_str = "\n---\n".join(relevant_context)
            rag_context_message += f"Kontext z dokumentov:\n{context_str}\n\n"
        
        rag_context_message += f"Používateľova otázka:\n{prompt_text}"

        # 3. ZOSTAVENIE PROMPTU (s históriou)
        formatted_history = self.format_chatml_history(history)
        
        full_prompt = (
            SYSTEM_PROMPT +                  
            formatted_history +              
            f'<|im_start|>user\n{rag_context_message}<|im_end|>\n' + 
            '<|im_start|>assistant\n'        
        )

        try:
            # Generovanie odpovede (zostáva rovnaké)
            output = self.llm(
                full_prompt,  
                max_tokens=5000,
                temperature=0.8,
                stop=["<|im_end|>", "<|im_start|>"], 
                echo=False
            )
            
            return output["choices"][0]["text"].strip()
        
        except Exception as e:
            return f"Chyba pri generovaní odpovede pomocou GGUF modelu: {e}"