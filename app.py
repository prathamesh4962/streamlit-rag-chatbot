import streamlit as st
import uuid
import os
from models.gemini_bot import GeminiBot
from models.local_bot import LocalRAGBot


# --- KONŠTANTY A IKONKY ---
PAGE_ICON = "💬"
GEMINI_CODENAME = "Gemini-2.5-Flash-Lite"
GEMMA_3_4B_CODENAME = "Gemma-3-4B-It"
GEMMA_3_1B_CODENAME = "Gemma-3-1B-It"
EUROLLM_CODENAME = "EuroLLM-1.7B-Instruct"

# Dynamické určenie cesty k adresáru 'GGUF models'
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
LOCAL_MODEL_PATH_1 = os.path.join(BASE_DIR, "GGUF models", "gemma-3-4b-it-Q8_0.gguf")
LOCAL_MODEL_PATH_2 = os.path.join(BASE_DIR, "GGUF models", "gemma-3-1b-it-Q8_0.gguf")
LOCAL_MODEL_PATH_3 = os.path.join(BASE_DIR, "GGUF models", "EuroLLM-1.7B-Instruct.Q4_K_S.gguf")


st.set_page_config(page_title="SaSW RAG ChatBot", page_icon=PAGE_ICON, layout="wide")

st.title("💬 SaSW RAG ChatBot")


# === FUNKCIE PRE VIACERO CHATOV (Callbacky) ===

def start_new_chat():
    """Inicializuje nový chat. Spustí sa v on_click, spoliehame sa na automatický re-run."""
    new_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_chat_id] = []  
    st.session_state.current_chat_id = new_chat_id
    st.session_state.chat_names[new_chat_id] = f"Nový Chat {len(st.session_state.chat_sessions)}"
    
    if "bot" in st.session_state:
        # Voláme reset_rag_vault priamo, ak je to LocalRAGBot
        if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)): # Reset pre oba RAG boty
            st.session_state.bot.reset_rag_vault()
        
        del st.session_state.bot
        del st.session_state.model_choice
    # st.rerun() je odstránené, aby sa zabránilo 'no-op' chybe


def remove_chat():
    """Odstráni aktuálny chat. Spoliehame sa na automatický re-run."""
    if st.session_state.current_chat_id in st.session_state.chat_sessions:
        current_id = st.session_state.current_chat_id
        del st.session_state.chat_sessions[current_id]
        del st.session_state.chat_names[current_id]
        if current_id in st.session_state.model_per_chat:
             del st.session_state.model_per_chat[current_id]
        
        st.session_state.current_chat_id = None

        if st.session_state.chat_sessions:
            st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
        
        if "bot" in st.session_state:
             if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)): # Reset pre oba RAG boty
                st.session_state.bot.reset_rag_vault()
             del st.session_state.bot
             del st.session_state.model_choice


def clear_chat_history():
    """Vymaže históriu aktuálneho chatu. Spoliehame sa na automatický re-run."""
    if st.session_state.current_chat_id in st.session_state.chat_sessions:
         st.session_state.chat_sessions[st.session_state.current_chat_id] = []
         
         if "bot" in st.session_state:
            # Ak je to RAG bot, resetujeme jeho RAG vault
            if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)): # Reset pre oba RAG boty
                st.session_state.bot.reset_rag_vault()
            del st.session_state.bot
            del st.session_state.model_choice


# --- STAVOVÉ PREMENNÉ A PRVÁ INICIALIZÁCIA ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_chat_id = None
    st.session_state.chat_names = {}
    st.session_state.model_per_chat = {}
    
    start_new_chat()
    st.rerun() # Prvý rerun pre zobrazenie počiatočného stavu


# Zabezpečenie, že máme aktuálny chat
current_chat_id = st.session_state.current_chat_id
current_chat_messages = st.session_state.chat_sessions[current_chat_id]
model_options = [GEMINI_CODENAME, GEMMA_3_4B_CODENAME, GEMMA_3_1B_CODENAME, EUROLLM_CODENAME]


# --- SIDEBAR (Prepínanie modelov a chatov) ---
with st.sidebar:
    st.title("⚙️ Nastavenia")

    # === Prepínanie modelov ===
    current_model_choice = st.session_state.model_per_chat.get(current_chat_id, GEMINI_CODENAME)
    
    st.header("🕸️ Model")
    selected_model_choice = st.selectbox(
        label="🕸️ Model - skryte ale povinny parameter", 
        label_visibility="collapsed",
        options=model_options, 
        index=model_options.index(current_model_choice)
    )

    # Ak sa zmenil model, uložíme ho do stavu a vynútime re-inicializáciu
    if selected_model_choice != current_model_choice:
        st.session_state.model_per_chat[current_chat_id] = selected_model_choice
        if "bot" in st.session_state:
            # Ak je starý model RAG, resetujeme vault pred re-inicializáciou
            if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)): # Reset pre oba RAG boty
                 st.session_state.bot.reset_rag_vault()
            del st.session_state.bot
            del st.session_state.model_choice
        st.rerun() 

    model_choice = selected_model_choice

    st.header("📝 Správa chatov")
    
    # !!! Nový Chat Tlačidlo (Logika priamo) !!!
    if st.button('Nový Chat', use_container_width=True):
         start_new_chat()


    # === Opravený blok pre ZMENU MENA CHATU ===
    st.header("✍️ Meno chatu")
    st.text_input(
        label="✍️ Meno chatu - skryte ale povinny parameter",
        label_visibility="collapsed", 
        value=st.session_state.chat_names.get(current_chat_id, "Nový Chat"), 
        key="chat_name_input"
    )
    
    if "chat_name_input" in st.session_state:
        new_name = st.session_state["chat_name_input"]
        current_name = st.session_state.chat_names.get(current_chat_id)
        
        if current_name != new_name and new_name.strip():
            st.session_state.chat_names[current_chat_id] = new_name
            st.rerun() 
        elif not new_name.strip() and current_name != f"Chat {current_chat_id[:4]}":
             st.session_state.chat_names[current_chat_id] = f"Chat {current_chat_id[:4]}"
             st.rerun()


    st.button('Vymazať históriu aktuálneho chatu', on_click=clear_chat_history, use_container_width=True)
    if len(st.session_state.chat_sessions) > 1:
        st.button('Odstrániť chat', on_click=remove_chat, use_container_width=True)
    
    # === Prepínanie chatov ===
    st.header("👁️ Vyber chat")
    if st.session_state.chat_sessions:
        selected_chat = st.radio(
            label="Vyber chat - skryte, ale povinny parameter",
            label_visibility="collapsed",
            options=list(st.session_state.chat_sessions.keys()),
            format_func=lambda x: st.session_state.chat_names.get(x, f"Chat {x[:8]}"),
            index=list(st.session_state.chat_sessions.keys()).index(current_chat_id)
        )
        
        if selected_chat != current_chat_id:
            st.session_state.current_chat_id = selected_chat
            if "bot" in st.session_state:
                if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)): # Reset pre oba RAG boty
                    st.session_state.bot.reset_rag_vault()
                del st.session_state.bot
                del st.session_state.model_choice
            st.rerun()


# --- Inicializácia modelu ---
if "bot" not in st.session_state or st.session_state.get("model_choice") != model_choice:
    with st.spinner(f"Načítavam model **{model_choice}**..."):
        if model_choice == GEMINI_CODENAME:
            st.session_state.bot = GeminiBot(api_key=st.secrets["GEMINI_API_KEY"]) 
        elif model_choice == EUROLLM_CODENAME:
            st.session_state.bot = LocalRAGBot(model_path=LOCAL_MODEL_PATH_3)
        elif model_choice == GEMMA_3_4B_CODENAME:
            st.session_state.bot = LocalRAGBot(model_path=LOCAL_MODEL_PATH_1)
        elif model_choice == GEMMA_3_1B_CODENAME:
            st.session_state.bot = LocalRAGBot(model_path=LOCAL_MODEL_PATH_2)
    st.session_state.model_choice = model_choice


# --- Zobrazenie histórie aktuálneho chatu ---
for msg in current_chat_messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
             st.markdown(msg["content"])
        else:
            st.markdown(str(msg["content"])) 

# Chat input
prompt = st.chat_input(
    "Napíš niečo a/alebo prilož súbor...",
    accept_file=True,
    file_type=["txt", "pdf", "docx"] 
)

# --- Spracovanie vstupu ---
if prompt:
    user_text = prompt.text.strip() if hasattr(prompt, "text") and prompt.text else ""
    uploaded_files = prompt.files if hasattr(prompt, "files") and prompt.files else []

    if not user_text and not uploaded_files:
        st.error("Zadaj text alebo prilož aspoň jeden súbor.")
        st.stop()

    display_content = user_text
    if uploaded_files:
        file_list = ", ".join(f.name for f in uploaded_files)
        if display_content:
            display_content += "\n\n"
        display_content += f"📎 Súbory sa načítavajú do RAG Vaultu: **{file_list}**"


    # Zobrazenie používateľovej správy
    with st.chat_message("user"):
        st.markdown(display_content)
    
    # Uloženie správy do aktuálnej chatovej histórie!
    current_chat_messages.append({"role": "user", "content": display_content})

    # Generovanie odpovede
    with st.spinner("Premýšľam, generujem embeddingy a vyhľadávam relevantné informácie (RAG)..."):
        try:
            history_for_model = current_chat_messages[:-1] 
            
            response = st.session_state.bot.ask(
                user_text, 
                uploaded_files, 
                history=history_for_model
            )
             
        except Exception as e:
            response = f"Chyba pri generovaní odpovede: {e}"

    with st.chat_message("assistant"):
        st.markdown(f"<b>RAG ChatBot:</b><br>{response}", unsafe_allow_html=True)

    # Uloženie odpovede do aktuálnej chatovej histórie!
    current_chat_messages.append({"role": "assistant", "content": response})


# Posielam ti cenník áut Škoda Octavia. Za akú cenu si môžem kúpiť Liftback 2.0 TDI 110kW (150k) Sportline?