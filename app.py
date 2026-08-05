import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.patch_torchvision  # noqa: F401

import streamlit as st
import config
import ingest
from rag import answer_query

# Page Configuration
st.set_page_config(
    page_title="Multimodal Self-Healing RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Main App Gradient Background & Typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background: radial-gradient(circle at 10% 10%, #1e1b4b 0%, #0f172a 50%, #030712 100%) !important;
        background-attachment: fixed !important;
        color: #f8fafc;
    }

    /* Glassmorphism Main Content Container */
    [data-testid="stMainBlockContainer"] {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem 2.5rem !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* Gradient Headers */
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }

    /* Citation Card Styling */
    .citation-box {
        background: rgba(30, 41, 59, 0.85);
        color: #f1f5f9;
        border-left: 4px solid #6366f1;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        border-radius: 8px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    .card-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 1.3rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    .stAlert {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = config.OPENROUTER_API_KEY or ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = config.LLM_MODEL_NAME

if "selected_embedding" not in st.session_state:
    st.session_state.selected_embedding = config.EMBEDDING_MODEL_NAME

if "enable_reranker" not in st.session_state:
    st.session_state.enable_reranker = config.ENABLE_RERANKER


# Helper function to list documents in data folder
def get_data_files():
    data_files = set(config.DATA_DIR.glob("*.*")) | set(config.DATA_DIR.glob("**/*.*"))
    data_files = [f for f in data_files if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS]
    return sorted(data_files, key=lambda x: x.name.lower())


# Sidebar Navigation & Settings
with st.sidebar:
    st.title("📚 RAG System Portal")

    # Main Navigation Selector
    page = st.radio(
        "Navigation",
        options=["💬 Main RAG Chat", "📁 Document Manager", "🧠 Self-Healing Architecture", "❓ Help & User Guide"],
        index=0,
    )

    st.markdown("---")

    # API Key Setup
    st.subheader("1. API Key Setup")
    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=st.session_state.openrouter_api_key,
        type="password",
        help="Enter your OpenRouter API Key to call LLMs.",
    )
    if api_key_input:
        st.session_state.openrouter_api_key = api_key_input

    st.markdown("---")

    # LLM Selection
    st.subheader("2. LLM Selection")
    selected_llm = st.selectbox(
        "Model Provider & Architecture",
        options=config.AVAILABLE_LLMS,
        index=0 if config.LLM_MODEL_NAME not in config.AVAILABLE_LLMS else config.AVAILABLE_LLMS.index(config.LLM_MODEL_NAME),
        help="Select the LLM for answer generation via OpenRouter.",
    )
    st.session_state.selected_model = selected_llm

    custom_model = st.text_input("Custom Model ID (Optional)", placeholder="e.g. openai/gpt-4o-mini")
    if custom_model.strip():
        st.session_state.selected_model = custom_model.strip()

    st.markdown("---")

    # Embedding Model Selection
    st.subheader("3. Embedding Model")
    embedding_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
    ]
    selected_emb = st.selectbox(
        "Embedding Model",
        options=embedding_models,
        index=0,
        help="Model used to embed chunks and queries for FAISS vector search.",
    )
    st.session_state.selected_embedding = selected_emb

    try:
        from utils.vector_store import load_embedding_model_name
        saved_db_model = load_embedding_model_name()
        if saved_db_model:
            st.caption(f"Saved Vector DB Model: `:green[{saved_db_model}]`")
            if saved_db_model != st.session_state.selected_embedding:
                st.warning(f"⚠️ Vector DB was created with `{saved_db_model}`. Re-ingest documents if switching embedding models.")
    except Exception:
        pass

    st.markdown("---")

    # Reranker Controls
    st.subheader("4. Reranker Controls")
    st.session_state.enable_reranker = st.toggle(
        "Enable BGE Cross-Encoder Reranker",
        value=st.session_state.enable_reranker,
        help="Reranks candidate chunks using BAAI/bge-reranker-base for higher precision.",
    )

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==============================================================================
# PAGE 1: MAIN RAG CHAT INTERFACE
# ==============================================================================
if page == "💬 Main RAG Chat":
    st.markdown("<h1 class='main-header'>💬 Multimodal Self-Healing RAG Chat</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Ask questions about your uploaded documents (PDF, DOCX, TXT, MD, Images). "
        "Answers are strictly grounded using Hybrid Search (FAISS + BM25) and BGE Reranking.</p>",
        unsafe_allow_html=True,
    )

    # Display Active Documents Info Banner
    current_files = get_data_files()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.info(f"📁 **Active Document Database**: **{len(current_files)}** document(s) in `data/` folder.")
    with col_b:
        if st.button("Process / Re-Ingest DB", type="primary", use_container_width=True):
            if not current_files:
                st.warning("No documents found in data directory. Upload files in Document Manager.")
            else:
                progress_bar = st.progress(0.0)
                status_box = st.empty()
                log_box = st.empty()
                logs = []

                def on_progress(event: dict):
                    msg = event.get("message", "")
                    current = event.get("current", 0)
                    total = max(1, event.get("total", 1))
                    ratio = float(min(1.0, max(0.0, current / total)))

                    progress_bar.progress(ratio)
                    status_box.markdown(f"⏳ **Status**: {msg}")

                    if msg and (not logs or logs[-1] != msg):
                        logs.append(f"• {msg}")
                        if len(logs) > 6:
                            logs.pop(0)
                        log_box.code("\n".join(logs), language="text")

                try:
                    metadata = ingest.ingest_all_documents(
                        data_dir=config.DATA_DIR,
                        embedding_model_name=st.session_state.selected_embedding,
                        progress_callback=on_progress,
                    )
                    progress_bar.progress(1.0)
                    status_box.empty()
                    log_box.empty()
                    st.success(f"🎉 Successfully ingested **{len(metadata)}** chunks across **{len(current_files)}** document(s)!")
                except Exception as err:
                    st.error(f"Ingestion failed: {err}")

    st.markdown("---")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "confidence" in message:
                st.caption(f"🎯 **Answer Confidence**: {message['confidence']}%")
            if "sources" in message and message["sources"]:
                with st.expander(f"View {len(message['sources'])} Retrieved Source Citations"):
                    for src in message["sources"]:
                        doc_name = src.get("document_name") or src.get("source") or "Document"
                        page_num = src.get("page_number", 1)
                        heading = src.get("heading")
                        heading_str = f" | Heading: {heading}" if heading else ""
                        st.markdown(
                            f"<div class='citation-box'>"
                            f"<b>Source:</b> {doc_name}, Page {page_num}{heading_str}<br>"
                            f"<b>Snippet:</b> {src.get('text', '')[:350]}..."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # Chat Input Handler
    if user_query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating grounded response..."):
                try:
                    result = answer_query(
                        query=user_query,
                        llm_model=st.session_state.selected_model,
                        api_key=st.session_state.openrouter_api_key,
                        enable_reranker=st.session_state.enable_reranker,
                    )

                    answer_text = result["answer"]
                    confidence = result["confidence_score"]
                    sources = result["sources"]

                    st.markdown(answer_text)
                    st.caption(f"🎯 **Answer Confidence**: {confidence}%")

                    if sources:
                        with st.expander(f"View {len(sources)} Retrieved Source Citations"):
                            for src in sources:
                                doc_name = src.get("document_name") or src.get("source") or "Document"
                                page_num = src.get("page_number", 1)
                                heading = src.get("heading")
                                heading_str = f" | Heading: {heading}" if heading else ""
                                st.markdown(
                                    f"<div class='citation-box'>"
                                    f"<b>Source:</b> {doc_name}, Page {page_num}{heading_str}<br>"
                                    f"<b>Snippet:</b> {src.get('text', '')[:350]}..."
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "confidence": confidence,
                            "sources": sources,
                        }
                    )

                except Exception as err:
                    st.error(f"⚠️ Error: {err}")


# ==============================================================================
# PAGE 2: DOCUMENT MANAGER (UPLOAD, DELETE, PREVIEW)
# ==============================================================================
elif page == "📁 Document Manager":
    st.markdown("<h1 class='main-header'>📁 Document Management Portal</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Upload, preview, manage, or delete documents in your project `data/` directory.</p>",
        unsafe_allow_html=True,
    )

    # 1. Upload Section
    st.subheader("1. Add New Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, MD, PNG, JPG, JPEG files",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        save_dir = config.DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        new_count = 0
        for uploaded_file in uploaded_files:
            file_path = save_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_count += 1
        st.success(f"Successfully added **{new_count}** document(s) to `data/` folder!")
        st.rerun()

    st.markdown("---")

    # 2. Manage Active Documents
    st.subheader("2. Manage & Delete Active Documents")
    current_files = get_data_files()

    if not current_files:
        st.info("No documents found in `data/` folder. Upload files above.")
    else:
        st.markdown(f"Total Active Documents: **{len(current_files)}**")

        for idx, file_path in enumerate(current_files, 1):
            col_1, col_2, col_3, col_4 = st.columns([3, 1.5, 1.5, 1.5])
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            with col_1:
                st.markdown(f"📄 **{idx}. {file_path.name}**")
            with col_2:
                st.caption(f"Size: {file_size_mb:.2f} MB")
            with col_3:
                st.caption(f"Type: `{file_path.suffix.upper()}`")
            with col_4:
                if st.button(f"🗑️ Delete", key=f"del_{file_path.name}_{idx}"):
                    try:
                        os.remove(file_path)
                        st.toast(f"Deleted {file_path.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")

        st.markdown("---")
        col_rebuild, col_clear = st.columns([2, 2])
        with col_rebuild:
            if st.button("⚡ Process / Re-Ingest Vector DB Index", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_box = st.empty()
                log_box = st.empty()
                logs = []

                def on_progress(event: dict):
                    msg = event.get("message", "")
                    current = event.get("current", 0)
                    total = max(1, event.get("total", 1))
                    ratio = float(min(1.0, max(0.0, current / total)))

                    progress_bar.progress(ratio)
                    status_box.markdown(f"⏳ **Status**: {msg}")

                    if msg and (not logs or logs[-1] != msg):
                        logs.append(f"• {msg}")
                        if len(logs) > 6:
                            logs.pop(0)
                        log_box.code("\n".join(logs), language="text")

                try:
                    metadata = ingest.ingest_all_documents(
                        data_dir=config.DATA_DIR,
                        embedding_model_name=st.session_state.selected_embedding,
                        progress_callback=on_progress,
                    )
                    progress_bar.progress(1.0)
                    status_box.empty()
                    log_box.empty()
                    st.success(f"🎉 Successfully ingested **{len(metadata)}** text chunks!")
                except Exception as err:
                    st.error(f"Ingestion failed: {err}")

        with col_clear:
            if st.button("⚠️ Delete All Documents", use_container_width=True):
                for f in current_files:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                st.success("All documents deleted from `data/` directory.")
                st.rerun()


# ==============================================================================
# PAGE 3: SELF-HEALING RAG ARCHITECTURE GUIDE
# ==============================================================================
elif page == "🧠 Self-Healing Architecture":
    st.markdown("<h1 class='main-header'>🧠 Self-Healing RAG Architecture Guide</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Detailed technical architecture of your Multimodal Self-Healing Corrective RAG System.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 📌 The 6 Steps of Self-Healing RAG (CRAG)

        ```
        [ User Query ]
               │
               ▼
        [ Step 1 & 2: LLM Query Refiner & Expansion ]
               │ ──► Fixes OCR noise, typos, & expands synonyms while preserving IDs/numbers
               ▼
        [ Step 1: Hybrid Search (FAISS + BM25) ]
               │ ──► Dense Vector Similarity + Sparse Keyword Precision
               ▼
        [ Step 3 & 4: Relevance Evaluation & Self-Healing Loop ]
               │ ──► Evaluates top context score against threshold (0.35)
               │ ──► If low score, triggers self-healing query rewrite (Max 2 retries)
               ▼
        [ Step 3: BGE Cross-Encoder Reranker ]
               │ ──► Scores candidates with BAAI/bge-reranker-base
               ▼
        [ Step 5 & 6: Grounded Generation & Citation Metrics ]
               │ ──► Generates strictly grounded answer with 95%+ Confidence Badges
        ```
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            #### 🖼️ Multi-Engine Neural OCR
            - **PyMuPDF (`fitz`)**: Native digital text extraction & spatial bounding box `(y0, x0)` layout sorting.
            - **RapidOCR**: Fast ONNX neural OCR engine for scanned documents & tables.
            - **Docling (v2.117.0)**: Advanced document parser for complex layouts.
            - **EasyOCR & Pytesseract**: Deep learning fallbacks for rotated photos.
            """
        )
    with col2:
        st.markdown(
            """
            #### ⚡ Hybrid Retrieval & Reranking
            - **FAISS**: Vector index for dense semantic search (`all-MiniLM-L6-v2`).
            - **BM25**: Keyword search for exact numbers, account IDs, & legal rules.
            - **BGE Reranker**: Cross-Encoder (`bge-reranker-base`) for precision candidate scoring.
            - **Zero Hallucination Guard**: Strict grounding prompt prevents false claims.
            """
        )


# ==============================================================================
# PAGE 4: HELP & USER GUIDE
# ==============================================================================
elif page == "❓ Help & User Guide":
    st.markdown("<h1 class='main-header'>❓ Help & Frequently Asked Questions</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Instructions on running, configuring, and troubleshooting your RAG system.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 🛠️ Quick Start Guide

        1. **API Key Setup**:
           - Obtain an OpenRouter API key from [OpenRouter.ai](https://openrouter.ai).
           - Enter the key in sidebar field **1. API Key Setup**.

        2. **Document Management**:
           - Go to page **📁 Document Manager** in the sidebar.
           - Upload your PDF, DOCX, TXT, MD, or image files.
           - Click **Process / Re-Ingest Vector DB Index** to index your documents.

        3. **Ask Questions**:
           - Go to page **💬 Main RAG Chat**.
           - Type your question in the chat input at the bottom.
           - View 95%+ confidence grounded answers and source citations!

        ---

        ### ❓ Frequently Asked Questions

        **Q: Why did I get "I could not find this information"?**
        * **A:** Ensure you clicked **Process / Re-Ingest Vector DB Index** after uploading new files.

        **Q: What LLM models are recommended?**
        * **A:** `google/gemini-2.5-flash` or `openai/gpt-4o-mini` provide fast execution and accurate grounded answers.

        **Q: Where are files stored locally?**
        * **A:** Documents are stored in `D:\\projects\\RAG\\data`, and vector stores are saved in `D:\\projects\\RAG\\vectordb`.
        """
    )
