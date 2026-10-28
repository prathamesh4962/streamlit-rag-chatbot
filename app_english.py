import streamlit as st
import uuid
from models.gemini_bot import GeminiBot
from models.local_bot import LocalRAGBot

# Model options
PAGE_ICON = "💬"
GEMINI_CODENAME = "Gemini 2.5 Flash Lite"
OLLAMA_CODENAME = "Llama 3.2 (Ollama)"

st.set_page_config(page_title="SaSW RAG ChatBot", page_icon=PAGE_ICON, layout="wide")
st.title("💬 SaSW RAG ChatBot")


def start_new_chat():
    """Create a new chat session."""
    new_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_chat_id] = []
    st.session_state.current_chat_id = new_chat_id
    st.session_state.chat_names[new_chat_id] = f"New Chat {len(st.session_state.chat_sessions)}"

    if "bot" in st.session_state:
        if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)):
            st.session_state.bot.reset_rag_vault()
        del st.session_state.bot
    st.session_state.pop("model_choice", None)


def remove_chat():
    """Delete the current chat session."""
    current_id = st.session_state.current_chat_id
    if current_id in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[current_id]
        st.session_state.chat_names.pop(current_id, None)
        st.session_state.model_per_chat.pop(current_id, None)
        st.session_state.current_chat_id = (
            next(iter(st.session_state.chat_sessions), None)
        )

        if "bot" in st.session_state:
            if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)):
                st.session_state.bot.reset_rag_vault()
            del st.session_state.bot
        st.session_state.pop("model_choice", None)

        if st.session_state.current_chat_id is None:
            start_new_chat()


def clear_chat_history():
    """Clear messages in the current chat."""
    chat_id = st.session_state.current_chat_id
    if chat_id in st.session_state.chat_sessions:
        st.session_state.chat_sessions[chat_id] = []

        if "bot" in st.session_state:
            if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)):
                st.session_state.bot.reset_rag_vault()
            del st.session_state.bot
        st.session_state.pop("model_choice", None)


# Initialize chat state
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_chat_id = None
    st.session_state.chat_names = {}
    st.session_state.model_per_chat = {}
    start_new_chat()

if st.session_state.current_chat_id not in st.session_state.chat_sessions:
    start_new_chat()

current_chat_id = st.session_state.current_chat_id
current_chat_messages = st.session_state.chat_sessions[current_chat_id]
model_options = [GEMINI_CODENAME, OLLAMA_CODENAME]

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    current_model_choice = st.session_state.model_per_chat.get(
        current_chat_id, OLLAMA_CODENAME
    )
    if current_model_choice not in model_options:
        current_model_choice = OLLAMA_CODENAME
        st.session_state.model_per_chat[current_chat_id] = current_model_choice

    st.header("🤖 Model")
    selected_model_choice = st.selectbox(
        "Select a model",
        options=model_options,
        index=model_options.index(current_model_choice),
    )

    if selected_model_choice != current_model_choice:
        st.session_state.model_per_chat[current_chat_id] = selected_model_choice
        if "bot" in st.session_state:
            if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)):
                st.session_state.bot.reset_rag_vault()
            del st.session_state.bot
        st.session_state.pop("model_choice", None)
        st.rerun()

    model_choice = selected_model_choice

    st.header("📝 Chat Management")
    if st.button("＋ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.header("✍️ Chat Name")
    st.text_input(
        "Chat name",
        value=st.session_state.chat_names.get(current_chat_id, "New Chat"),
        key=f"chat_name_input_{current_chat_id}",
    )
    new_name = st.session_state.get(f"chat_name_input_{current_chat_id}", "")
    if new_name.strip() and st.session_state.chat_names.get(current_chat_id) != new_name:
        st.session_state.chat_names[current_chat_id] = new_name

    st.button(
        "Clear Current Chat",
        on_click=clear_chat_history,
        use_container_width=True,
    )
    if len(st.session_state.chat_sessions) > 1:
        st.button("Delete Current Chat", on_click=remove_chat, use_container_width=True)

    st.header("👁️ Select a Chat")
    if st.session_state.chat_sessions:
        chat_ids = list(st.session_state.chat_sessions.keys())
        selected_chat = st.radio(
            "Chat sessions",
            options=chat_ids,
            format_func=lambda chat_id: st.session_state.chat_names.get(
                chat_id, f"Chat {chat_id[:8]}"
            ),
            index=chat_ids.index(current_chat_id),
        )

        if selected_chat != current_chat_id:
            if "bot" in st.session_state:
                if isinstance(st.session_state.bot, (LocalRAGBot, GeminiBot)):
                    st.session_state.bot.reset_rag_vault()
                del st.session_state.bot
            st.session_state.pop("model_choice", None)
            st.session_state.current_chat_id = selected_chat
            st.rerun()

# Initialize selected model
if (
    "bot" not in st.session_state
    or st.session_state.get("model_choice") != model_choice
):
    with st.spinner(f"Loading {model_choice}..."):
        if model_choice == GEMINI_CODENAME:
            st.session_state.bot = GeminiBot(
                api_key=st.secrets["GEMINI_API_KEY"]
            )
        else:
            # LocalRAGBot now uses the installed Ollama llama3.2 model.
            st.session_state.bot = LocalRAGBot(model_path="llama3.2")
    st.session_state.model_choice = model_choice

# Display conversation history
for msg in current_chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(str(msg["content"]))

# Chat input
prompt = st.chat_input(
    "Type a message or attach a file...",
    accept_file=True,
    file_type=["txt", "pdf", "docx"],
)

# Process message
if prompt:
    user_text = prompt.text.strip() if hasattr(prompt, "text") and prompt.text else ""
    uploaded_files = prompt.files if hasattr(prompt, "files") and prompt.files else []

    if not user_text and not uploaded_files:
        st.error("Please enter a message or attach at least one file.")
        st.stop()

    display_content = user_text
    if uploaded_files:
        file_list = ", ".join(file.name for file in uploaded_files)
        if display_content:
            display_content += "\n\n"
        display_content += f"📎 Files added for document search: **{file_list}**"

    with st.chat_message("user"):
        st.markdown(display_content)

    current_chat_messages.append({"role": "user", "content": display_content})

    with st.spinner("Thinking, creating embeddings, and searching document context (RAG)..."):
        try:
            history_for_model = current_chat_messages[:-1]
            response = st.session_state.bot.ask(
                user_text,
                uploaded_files,
                history=history_for_model,
            )
        except Exception as e:
            response = f"Error generating a response: {e}"

    with st.chat_message("assistant"):
        st.markdown(f"**RAG ChatBot:**\n\n{response}")

    current_chat_messages.append({"role": "assistant", "content": response})
