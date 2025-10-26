# Import the libraries
import streamlit as st
import requests
import json
import speech_recognition as sr
from gtts import gTTS
import io
import os
import glob
import time
import pyperclip # For copy functionality
from datetime import datetime # For date/time check

# --- RAG Specific Imports ---
import chromadb # Vector Store
from chromadb.utils import embedding_functions # Easy way to handle embeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Document Loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter # CORRECT import
import tempfile # To handle uploaded files temporarily

# --- Constants ---
HISTORY_DIR = "chat_history"
VECTOR_DB_PATH = "vector_db" # Directory to store the vector database
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Recommended embedding model
CHROMA_COLLECTION_NAME = "chat_documents" # Name for the collection in ChromaDB

# --- Initialize Vector DB and Embedding Function ---
# Use SentenceTransformerEmbeddingFunction for ease with ChromaDB
# Downloads model on first run if not cached
try:
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    # Initialize ChromaDB client (persistent storage)
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    # Get or create the collection with the specified embedding function
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"} # Use cosine distance
    )
    print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' loaded/created at '{VECTOR_DB_PATH}'.")
    print(f"Current document count: {collection.count()}")
except Exception as e:
    st.error(f"🚨 Failed to initialize Vector DB or Embedding Model: {e}")
    # Optionally stop the app if DB is critical, or just disable RAG features
    st.stop()


# --- Page Setup ---
st.set_page_config(
    page_title="Code Red AI Assistant (RAG)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define History Directory ---
os.makedirs(HISTORY_DIR, exist_ok=True) # Create folder if it doesn't exist


# --- Centered Title ---
col1_title, col2_title, col3_title = st.columns([1, 4, 1])
with col2_title:
    st.markdown("<h1 style='text-align: center;'>Code Red AI Assistant 🤖</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Powered by Ollama, RAG, Streamlit & gTTS</p>", unsafe_allow_html=True) # Updated caption


# --- Chat History Functions ---
def get_chat_files():
    """Gets all chat .json files, sorted newest first."""
    files = glob.glob(os.path.join(HISTORY_DIR, "chat_*.json")); files.sort(key=os.path.getmtime, reverse=True); return files

def load_chat_history(file_path):
    """Loads chat history from a JSON file. Returns empty list on error/empty."""
    try:
        with open(file_path, "r", encoding="utf-8") as f: content = f.read(); return json.loads(content) if content.strip() else []
    except json.JSONDecodeError: st.error(f"⚠️ Error decoding JSON: {os.path.basename(file_path)}"); return []
    except FileNotFoundError: st.error(f"⚠️ Chat file not found: {os.path.basename(file_path)}"); return []
    except Exception as e: st.error(f"⚠️ Error loading chat {os.path.basename(file_path)}: {e}"); return []

def save_chat_history(file_path, messages):
    """Saves chat messages list to a JSON file."""
    try:
        if file_path and isinstance(messages, list):
             with open(file_path, "w", encoding="utf-8") as f: json.dump(messages, f, indent=2, ensure_ascii=False)
    except Exception as e: st.error(f"⚠️ Error saving chat: {e}")

# --- STT Function ---
def listen_from_mic(lang_code, lang_name):
    """Listens via microphone, shows status in sidebar, returns text."""
    status_placeholder = st.sidebar.empty()
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        status_placeholder.info(f"Listening ({lang_name})...");
        try: recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except Exception as e: print(f"Noise adjust failed: {e}") # Non-critical
        recognizer.pause_threshold = 1.0
        try: audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError: status_placeholder.warning("No speech detected."); time.sleep(2); status_placeholder.empty(); return None
    try:
        status_placeholder.info("Recognizing..."); prompt_text = recognizer.recognize_google(audio, language=lang_code)
        status_placeholder.success(f"Heard: {prompt_text[:50]}..."); time.sleep(1); status_placeholder.empty(); return prompt_text
    except sr.UnknownValueError: status_placeholder.error("Could not understand.")
    except sr.RequestError as e: status_placeholder.error(f"API Error: {e}")
    except Exception as e: status_placeholder.error(f"Recognition Error: {e}")
    time.sleep(2); status_placeholder.empty(); return None

# --- Document Processing Function ---
@st.cache_data # Cache results to avoid reprocessing same file unless content changes
def process_uploaded_file(uploaded_file):
    """Loads, splits, and prepares documents from an uploaded file."""
    docs = []
    temp_dir = tempfile.mkdtemp()
    try:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            st.warning(f"Unsupported file type: {file_extension}. Skipping.")
            return [], uploaded_file.name # Return empty list and filename

        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Add filename to metadata for each chunk
        for i, doc in enumerate(docs):
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["chunk_index"] = i # Add chunk index

    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {e}")
        return [], uploaded_file.name # Return empty list on error
    finally:
        # Clean up temporary file
        if os.path.exists(temp_dir):
            try: # Add try-except for cleanup robustness
                for f_name in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f_name))
                os.rmdir(temp_dir)
            except Exception as cleanup_e:
                 print(f"Error during temp dir cleanup: {cleanup_e}")
    return docs, uploaded_file.name

def add_docs_to_vector_db(docs, collection):
    """Adds Langchain Document objects to ChromaDB collection."""
    if not docs: return 0

    doc_contents = [doc.page_content for doc in docs]
    doc_metadatas = [doc.metadata for doc in docs]
    # Create unique IDs based on source and chunk index
    doc_ids = [f"{meta.get('source', 'unknown')}_{meta.get('chunk_index', idx)}" for idx, meta in enumerate(doc_metadatas)]

    try:
        # Use upsert to handle potential re-uploads gracefully
        collection.upsert(
            documents=doc_contents,
            metadatas=doc_metadatas,
            ids=doc_ids
        )
        print(f"Upserted {len(docs)} document chunks.")
        return len(docs)
    except Exception as e:
        st.error(f"Error adding/updating documents in ChromaDB: {e}")
        return 0


# --- Chat Logic Function with RAG ---
def process_and_display_chat(prompt, lang_code, lang_name, selected_model, system_prompt, enable_tts):
    """Includes RAG query before calling Ollama via /api/chat."""
    global collection # Access the global ChromaDB collection

    if not prompt: st.warning("Please enter or say something first."); return

    with st.chat_message("user"): st.markdown(prompt)
    if 'messages' not in st.session_state: st.session_state.messages = []

    current_file = st.session_state.get("current_chat_file")
    is_first_message_for_file = False
    if not current_file:
        is_first_message_for_file = True; timestamp = int(time.time())
        safe_prompt_name = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()[:20] if prompt else "NewChat"
        current_file = os.path.join(HISTORY_DIR, f"chat_{timestamp}_{safe_prompt_name}.json")
        st.session_state.current_chat_file = current_file; st.session_state.messages = []
        print(f"Set new chat file: {os.path.basename(current_file)}")

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Direct Date/Time Handling
    prompt_lower = prompt.lower(); datetime_response = None
    if any(k in prompt_lower for k in ["date", "today", "what day", "day is it"]) or \
       any(k in prompt_lower for k in ["time", "what time", "time is it"]):
        now = datetime.now(); day_name = now.strftime("%A"); date_str = now.strftime("%B %d, %Y"); time_str = now.strftime("%I:%M %p")
        if any(k in prompt_lower for k in ["date", "today", "what day", "day is it"]): datetime_response = f"Today is {day_name}, {date_str}."
        if any(k in prompt_lower for k in ["time", "what time", "time is it"]):
             if datetime_response: datetime_response += f" The current time is {time_str}."
             else: datetime_response = f"The current time is {time_str}."

    # Process Response
    with st.chat_message("assistant"):
        response_text_to_speak = None; full_response = ""
        if datetime_response: # If handled directly
            message_placeholder = st.empty(); message_placeholder.markdown(datetime_response)
            st.session_state.messages.append({"role": "assistant", "content": datetime_response})
            save_chat_history(current_file, st.session_state.messages)
            response_text_to_speak = datetime_response; full_response = datetime_response
        else: # Ask Ollama (potentially with RAG context)
            message_placeholder = st.empty()
            error_message = None
            retrieved_context_str = "" # To store context from DB

            try:
                # --- RAG Query ---
                if collection.count() > 0:
                    with st.spinner("📚 Searching documents..."):
                        try:
                            results = collection.query(query_texts=[prompt], n_results=3) # Retrieve top 3 chunks
                            if results and results.get('documents') and results['documents'][0]:
                                context_parts = []
                                for doc, meta in zip(results['documents'][0], results.get('metadatas', [{}])[0]):
                                    source = meta.get('source', 'Unknown')
                                    context_parts.append(f"Source: {source}\nContent: {doc}")
                                retrieved_context_str = "\n\n---\n\n".join(context_parts)
                                print(f"Retrieved Context:\n{retrieved_context_str}\n---") # Log context
                            else: print("No relevant documents found.")
                        except Exception as query_e: st.warning(f"⚠️ Failed RAG query: {query_e}")

                # --- Prepare messages for /api/chat ---
                api_messages = []
                rag_system_prompt = system_prompt or "You are a helpful AI assistant."
                if retrieved_context_str:
                    rag_system_prompt += "\n\nUse the following relevant document excerpts to answer the user's question:\n---BEGIN CONTEXT---\n"
                    rag_system_prompt += retrieved_context_str
                    rag_system_prompt += "\n---END CONTEXT---"
                api_messages.append({"role": "system", "content": rag_system_prompt})

                # Add history
                max_history_turns_ollama = 4 # Keep history reasonable with RAG
                relevant_history_ollama = st.session_state.messages[-(2*max_history_turns_ollama):-1]
                for msg in relevant_history_ollama:
                    content = msg.get("content");
                    if content is not None: api_messages.append({"role": msg["role"], "content": content})

                # Add final user prompt
                final_user_prompt = f"IMPORTANT: Respond ONLY in {lang_name}. Considering provided context and history, answer: '{prompt}'"
                api_messages.append({"role": "user", "content": final_user_prompt})

                # --- Streaming API Call using /api/chat ---
                ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/chat")
                data_payload = {"model": selected_model, "messages": api_messages, "stream": True}

                def stream_ollama_chat_response():
                    nonlocal full_response, error_message
                    try:
                        with requests.post(ollama_url, json=data_payload, stream=True, timeout=120) as response:
                            response.raise_for_status(); buffer = ""
                            for chunk_bytes in response.iter_content(chunk_size=None):
                                buffer += chunk_bytes.decode('utf-8')
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    if line.strip():
                                        try:
                                            chunk = json.loads(line)
                                            if "message" in chunk and "content" in chunk["message"]: token = chunk["message"]["content"]; full_response += token; yield token
                                            if chunk.get("done"):
                                                if chunk.get("error"): error_message = f"Ollama error: {chunk['error']}"; print(error_message); yield f"\n\n🚨 {error_message}"
                                                return
                                        except json.JSONDecodeError: print(f"Failed JSON decode: {line}")
                        if buffer.strip() and not error_message: print(f"Warn: Stream end buffer: {buffer}")
                    except requests.exceptions.Timeout: error_message = "🚨 Error: Ollama timeout."; yield f"\n\n{error_message}"
                    except requests.exceptions.RequestException as e: error_message = f"🚨 Error connecting: {e}."; yield f"\n\n{error_message}"
                    except Exception as e: error_message = f"🚨 Unexpected stream error: {e}"; yield f"\n\n{error_message}"

                with st.spinner("🧠 Thinking..."): # Spinner wraps stream display
                    message_placeholder.write_stream(stream_ollama_chat_response)
                time.sleep(0.1)

                if error_message: print(f"Final Error: {error_message}")
                elif not full_response.strip(): error_message = "⚠️ Error: Ollama stream empty."; message_placeholder.warning(error_message)

            except Exception as e: error_message = f"🚨 Unexpected error before streaming: {e}"

            # --- Save history and prepare TTS ---
            if error_message:
                if message_placeholder.empty(): message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"{error_message}"})
                save_chat_history(current_file, st.session_state.messages)
                response_text_to_speak = "An error occurred."
            elif full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
                save_chat_history(current_file, st.session_state.messages)
                response_text_to_speak = full_response.strip()

        # --- Text-to-Speech ---
        if response_text_to_speak and enable_tts:
            try:
                tts_lang = lang_code.split('-')[0]
                tts = gTTS(text=response_text_to_speak, lang=tts_lang)
                audio_bytes = io.BytesIO(); tts.write_to_fp(audio_bytes); audio_bytes.seek(0)
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            except Exception as tts_error: st.toast(f"⚠️ Audio Error: {tts_error}", icon="🔇")

    if is_first_message_for_file and current_file: time.sleep(0.5); st.rerun()


# --- Function to format chat for clipboard ---
def format_chat_for_clipboard(messages):
    formatted_string = "";
    for msg in messages: role = "👤 You" if msg.get("role") == "user" else "🤖 Assistant"; content = msg.get('content', 'N/A'); formatted_string += f"**{role}:**\n{content}\n\n---\n\n"
    return formatted_string.strip()

# -----------------------------------------------------------------
# --- SIDEBAR ---
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Code Red AI")
    # Placeholder Navigation
    st.button("🔍 Search", use_container_width=True, disabled=True); st.button("💬 Chat", use_container_width=True, type="primary")
    st.button("🖼 Imagine", use_container_width=True, disabled=True); st.button("📁 Projects", use_container_width=True, disabled=True)
    st.button("➕ New Workspace", use_container_width=True, disabled=True); st.divider()

    # Chat Controls
    st.header("⚙️ Controls")
    if st.button("➕ New Chat", use_container_width=True, help="Start new conversation"): st.session_state.messages = []; st.session_state.current_chat_file = None; st.rerun()

    # System Prompt Input
    if 'system_prompt' not in st.session_state: st.session_state.system_prompt = "You are a helpful AI assistant."
    system_prompt = st.text_area("System Prompt:", value=st.session_state.system_prompt, key="system_prompt_input", height=100)
    st.session_state.system_prompt = system_prompt

    # Model Selection
    AVAILABLE_MODELS = ["llama3:8b", "phi3:latest", "mistral:instruct", "qwen2:7b-instruct"] # Add yours
    if 'selected_model' not in st.session_state or st.session_state.selected_model not in AVAILABLE_MODELS: st.session_state.selected_model = AVAILABLE_MODELS[0]
    selected_model = st.selectbox("Select Model:", options=AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(st.session_state.selected_model), key="model_select_sidebar")
    st.session_state.selected_model = selected_model

    # Language Selection
    LANGUAGES = { "English": "en-US", "Spanish": "es-ES", "French": "fr-FR", "German": "de-DE", "Hindi": "hi-IN", "Japanese": "ja-JP" }
    selected_lang_name = st.selectbox("Select Language:", options=list(LANGUAGES.keys()), key="language_select_sidebar")
    selected_lang_code = LANGUAGES[selected_lang_name]

    # TTS Toggle
    if 'enable_tts' not in st.session_state: st.session_state.enable_tts = True
    enable_tts = st.toggle("🔊 Voice Output", value=st.session_state.enable_tts, key="tts_toggle")
    st.session_state.enable_tts = enable_tts

    # Speak Button
    if st.button("🎙 Speak", use_container_width=True, help=f"Speak in {selected_lang_name}"):
        voice_prompt = listen_from_mic(selected_lang_code, selected_lang_name)
        if voice_prompt:
             if 'messages' not in st.session_state: st.session_state.messages = []
             if "current_chat_file" not in st.session_state: st.session_state.current_chat_file = None
             process_and_display_chat(voice_prompt, selected_lang_code, selected_lang_name, selected_model, system_prompt, enable_tts) # Pass TTS state

    st.divider()

    # --- Document Upload Section ---
    st.header("📄 Document Q&A (RAG)")
    uploaded_files = st.file_uploader(
        "Upload Documents (TXT, PDF)", type=["txt", "pdf"],
        accept_multiple_files=True, help="Upload files to chat about their content."
    )

    if uploaded_files:
        processed_files_info = {} # Track processed files in this session
        files_to_process = []
        for f in uploaded_files:
             # Basic check to avoid reprocessing if filename seen (adjust if needed)
             if f.name not in processed_files_info:
                  files_to_process.append(f)
                  processed_files_info[f.name] = "processing" # Mark as seen

        if files_to_process:
            total_added_chunks = 0
            with st.spinner(f"Processing {len(files_to_process)} new document(s)..."):
                for uploaded_file in files_to_process:
                    docs, filename = process_uploaded_file(uploaded_file)
                    if docs:
                        added_count = add_docs_to_vector_db(docs, collection)
                        if added_count > 0:
                            st.toast(f"✅ Added {added_count} chunks from {filename}", icon="📄")
                            total_added_chunks += added_count
                        else: st.toast(f"❌ Failed add: {filename}", icon="⚠️")
                    else: st.warning(f"⚠️ Cannot process: {filename}")
            if total_added_chunks > 0:
                st.success(f"Added {total_added_chunks} total chunks to DB.")
                # Optionally clear uploader via rerun, or manage state differently
                # st.rerun()

    st.caption(f"Vector DB contains {collection.count()} document chunks.")
    if st.button("Clear Document DB", help="⚠️ Deletes all ingested documents!"):
        with st.spinner("Clearing DB..."):
            try:
                collection.delete(where={"chunk_index": {"$gte": 0}}) # Simple way to target all
                st.success(f"Cleared all documents. Count: {collection.count()}")
                time.sleep(1); st.rerun()
            except Exception as e: st.error(f"Failed to clear DB: {e}")
    st.divider()


    # Chat History Section
    st.header("📜 History")
    chat_files = get_chat_files();
    if not chat_files: st.caption("No previous chats.")
    for i, file_path in enumerate(chat_files):
        file_name = os.path.basename(file_path)
        try: parts = file_name.split("_"); label = " ".join(parts[2:]).replace(".json", "") or f"Chat {parts[1]}"
        except Exception: label = file_name
        col1_hist, col2_hist, col3_hist = st.columns([0.7, 0.15, 0.15])
        with col1_hist: # Load (ensure indentation is correct)
            is_current = st.session_state.get("current_chat_file") == file_path; button_type = "primary" if is_current else "secondary"
            if st.button(label, key=f"load_{i}_{file_path}", use_container_width=True, type=button_type, help=f"Load: {label}"):
                if not is_current: st.session_state.messages = load_chat_history(file_path); st.session_state.current_chat_file = file_path; st.rerun()
        with col2_hist: # Copy (ensure indentation is correct)
            if st.button("📋", key=f"copy_{i}_{file_path}", help="Copy chat", use_container_width=True):
                try: # TRY starts here
                    chat_to_copy = load_chat_history(file_path)
                    if chat_to_copy:
                        pyperclip.copy(format_chat_for_clipboard(chat_to_copy))
                        st.toast("Copied!", icon="✅")
                    else:
                        st.warning("Empty chat.")
                except Exception as e: # EXCEPT is correctly placed here
                    st.error(f"Copy failed: {e}")
        with col3_hist: # Delete (ensure indentation is correct)
            if st.button("🗑️", key=f"del_{i}_{file_path}", help="Delete chat", use_container_width=True):
                try: # TRY starts here
                    os.remove(file_path); st.toast(f"Deleted '{label}'", icon="🗑️")
                    if st.session_state.get("current_chat_file") == file_path: st.session_state.messages = []; st.session_state.current_chat_file = None
                    time.sleep(0.3); st.rerun()
                except FileNotFoundError: # EXCEPT is correctly placed here
                    st.error("Already deleted."); time.sleep(0.3); st.rerun()
                except Exception as e: # Another EXCEPT is correctly placed here
                    st.error(f"Delete failed: {e}")


# -----------------------------------------------------------------
# --- MAIN CHAT AREA ---
# -----------------------------------------------------------------

# Chat Message Display Area
chat_container = st.container()
with chat_container:
    current_messages = st.session_state.get('messages', [])
    if not current_messages and st.session_state.get("current_chat_file"): current_messages = load_chat_history(st.session_state.current_chat_file); st.session_state.messages = current_messages
    if current_messages:
        for msg_index, message in enumerate(current_messages):
            if isinstance(message, dict) and "role" in message and "content" in message:
                with st.chat_message(message["role"]):
                    col1_msg, col2_msg = st.columns([0.95, 0.05])
                    with col1_msg: st.markdown(message["content"])
                    if message["role"] == "assistant": # Check role for copy button
                        # Ensure content exists before creating button key
                        content_slice = message.get("content", "")[:10]
                        button_key = f"copy_msg_{msg_index}_{content_slice}"
                        with col2_msg:
                            if st.button("📋", key=button_key, help="Copy", type="secondary"):
                                pyperclip.copy(message.get("content", "")) # Use get for safety
                                st.toast("Message copied!", icon="✅")
            else: print(f"Skipping malformed message: {message}")
    else: st.caption("Start a new chat, load history, or upload documents.")


# Chat Input Box
if prompt := st.chat_input("Ask about your documents or anything else..."): # Updated placeholder
    if 'messages' not in st.session_state: st.session_state.messages = []
    if "current_chat_file" not in st.session_state: st.session_state.current_chat_file = None
    # Pass all relevant states
    process_and_display_chat(prompt, selected_lang_code, selected_lang_name, st.session_state.selected_model, st.session_state.system_prompt, st.session_state.enable_tts)