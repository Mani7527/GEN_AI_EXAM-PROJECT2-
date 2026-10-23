"""
app.py - Streamlit Frontend for Designs Law Assistant

A clean, professional web application for answering questions on India's
Manual of Designs Practice & Procedure using LangChain, ChromaDB, and Gemini.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import DesignsLawRAG, DEFAULT_CHROMA_DIR, DEFAULT_RETRIEVAL_K, DEFAULT_RETRIEVAL_THRESHOLD

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Designs Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Navy Blue, Light Blue, White Theme)
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header typography */
    .app-header {
        background: linear-gradient(135deg, #0a2540 0%, #1a365d 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(10, 37, 64, 0.15);
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-header p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Cards */
    .answer-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .answer-card.not-found {
        border-left: 5px solid #e53e3e;
        background-color: #fffaf0;
    }
    .answer-card.success {
        border-left: 5px solid #2b6cb0;
    }

    /* Badges */
    .badge-page {
        display: inline-block;
        background-color: #ebf8ff;
        color: #2b6cb0;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #bee3f8;
    }
    .badge-chapter {
        display: inline-block;
        background-color: #edf2f7;
        color: #2d3748;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-not-found {
        display: inline-block;
        background-color: #fed7d7;
        color: #9b2c2c;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
    }

    /* Disclaimer */
    .disclaimer-box {
        background-color: #f7fafc;
        border-top: 1px solid #e2e8f0;
        padding: 16px 20px;
        font-size: 0.85rem;
        color: #718096;
        border-radius: 8px;
        margin-top: 30px;
    }

    /* Sample Questions */
    .sample-q-btn {
        margin-bottom: 8px;
        text-align: left;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Default PDF file location
DEFAULT_PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "Manual_of_Designs_Practice_and_Procedure.pdf")


@st.cache_resource(show_spinner=False)
def get_rag_pipeline(api_key: str):
    """Instantiate and cache the RAG pipeline."""
    return DesignsLawRAG(
        chroma_dir=DEFAULT_CHROMA_DIR,
        api_key=api_key
    )


# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚖️ System Configuration")

    # API Key Input
    env_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        help="Reads GOOGLE_API_KEY from .env by default. You can override it here."
    )

    if api_key:
        st.success("API Key configured", icon="✅")
    else:
        st.warning("API Key missing! Add to .env or enter above.", icon="⚠️")

    st.markdown("---")
    st.markdown("### 📄 Document & Index Status")

    # Initialize RAG
    rag = get_rag_pipeline(api_key)
    if api_key:
        rag.set_api_key(api_key)

    index_status = rag.get_index_status()

    # Auto-index default PDF if not already indexed
    if not index_status["is_indexed"] and os.path.exists(DEFAULT_PDF_PATH):
        with st.spinner("Initial indexing of 'Manual of Designs Practice & Procedure'..."):
            try:
                index_result = rag.index_document(DEFAULT_PDF_PATH)
                st.toast("Manual indexed successfully!", icon="📚")
                index_status = rag.get_index_status()
            except Exception as e:
                st.error(f"Error during initial indexing: {e}")

    if index_status["is_indexed"]:
        meta = index_status.get("metadata", {})
        pages = meta.get("total_pages", 87)
        chunks = index_status.get("total_chunks", 0)
        st.metric(label="Document Pages", value=pages)
        st.metric(label="ChromaDB Chunks", value=chunks)
        st.info(f"Source: {meta.get('filename', 'Manual_of_Designs_Practice_and_Procedure.pdf')}")
    else:
        st.warning("No document currently indexed. Please upload or index below.", icon="📁")

    # File uploader
    st.markdown("#### Upload Source PDF")
    uploaded_file = st.file_uploader("Upload Manual PDF", type=["pdf"])

    if uploaded_file is not None:
        save_path = os.path.join(os.path.dirname(__file__), "data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")
        if st.button("Index Uploaded PDF", type="primary"):
            with st.spinner("Extracting pages, generating embeddings, and storing in ChromaDB..."):
                try:
                    res = rag.index_document(save_path, force_reindex=True)
                    st.success(f"Indexed {res['total_pages']} pages into {res['total_chunks']} chunks!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    if index_status["is_indexed"]:
        if st.button("Force Re-index"):
            target_path = DEFAULT_PDF_PATH if os.path.exists(DEFAULT_PDF_PATH) else None
            if target_path:
                with st.spinner("Re-indexing document..."):
                    rag.index_document(target_path, force_reindex=True)
                    st.success("Re-indexing complete!")
                    st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Parameters")
    retrieval_k = st.slider("Top-K Excerpts", min_value=1, max_value=8, value=DEFAULT_RETRIEVAL_K, step=1)
    retrieval_threshold = st.slider(
        "Distance Threshold",
        min_value=0.40,
        max_value=1.00,
        value=DEFAULT_RETRIEVAL_THRESHOLD,
        step=0.05,
        help="Cosine distance cutoff. Lower is stricter. If best chunk distance exceeds this, 'Not found' is returned directly."
    )

    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# Main Application Area
st.markdown("""
<div class="app-header">
    <h1>⚖️ Designs Law Assistant</h1>
    <p>AI-powered answers grounded in India's <strong>Manual of Designs Practice & Procedure</strong></p>
</div>
""", unsafe_allow_html=True)

# Assignment Sample Questions
st.markdown("#### 🎯 Clickable Assignment Questions")
st.caption("Click any of the five required assignment questions to test the grounded RAG pipeline immediately:")

ASSIGNMENT_QUESTIONS = [
    "How long does a proprietor have copyright in a registered design, and can it be extended?",
    "Which classification system is used to register designs?",
    "What is the government fee to register a trademark?",
    "Who is the current Controller General of Patents?",
    "What is the fee to register a design?"
]

col1, col2 = st.columns([1, 1])

clicked_sample = None
for idx, q_text in enumerate(ASSIGNMENT_QUESTIONS):
    target_col = col1 if idx < 3 else col2
    with target_col:
        if st.button(f"Q{idx+1}: {q_text}", key=f"sample_q_{idx}", use_container_width=True):
            clicked_sample = q_text

# Text input for questions
user_question = st.text_input(
    "Ask a question about Indian Industrial Designs Law:",
    placeholder="e.g., How long is a registered design protected in India?",
    key="user_query_input"
)

ask_btn = st.button("🔍 Ask Assistant", type="primary")

query_to_run = clicked_sample or (user_question.strip() if ask_btn else None)

# Execute question
if query_to_run:
    if not api_key:
        st.error("Please provide a valid Google Gemini API key in the sidebar before asking questions.")
    elif not index_status["is_indexed"]:
        st.error("Document is not yet indexed in ChromaDB. Please ensure the PDF is indexed via the sidebar.")
    else:
        with st.spinner("Searching Manual and generating verified answer..."):
            result = rag.answer_question(
                question=query_to_run,
                k=retrieval_k,
                threshold=retrieval_threshold
            )
            st.session_state.chat_history.insert(0, result)

# Display latest answer and history
if st.session_state.chat_history:
    st.markdown("### 💬 Answers & Verified Sources")

    for idx, item in enumerate(st.session_state.chat_history):
        q = item.get("question", "")
        ans = item.get("answer", "")
        is_nf = item.get("is_not_found", False)
        is_err = item.get("is_error", False)
        citations = item.get("citations", [])
        excerpts = item.get("source_excerpts", [])
        reason = item.get("reason", "")

        card_class = "not-found" if is_nf else "success"

        st.markdown(f"""
        <div class="answer-card {card_class}">
            <h4 style="margin-top: 0; color: #1a365d;">📌 Question: {q}</h4>
            <hr style="border: none; border-top: 1px solid #edf2f7; margin: 12px 0;">
        """, unsafe_allow_html=True)

        if is_nf:
            st.markdown("""
            <div style="margin-bottom: 12px;">
                <span class="badge-not-found">Not found</span>
            </div>
            """, unsafe_allow_html=True)
            if reason:
                st.info(f"**Verification Reason:** {reason}")
            st.write(ans)
        elif is_err:
            st.error(ans)
        else:
            st.markdown(ans)

            # Citations
            if citations:
                st.markdown("##### 📖 Source Citations:")
                citation_html = ""
                for c in citations:
                    page_badge = f"<span class='badge-page'>PDF Page {c.get('page', 'N/A')}</span>"
                    ch_badge = f"<span class='badge-chapter'>{c.get('chapter', 'Chapter N/A')}</span>"
                    sec = c.get('section')
                    sec_badge = f"<span class='badge-chapter'>Section: {sec}</span>" if sec and sec != "N/A" else ""
                    citation_html += f"<div>{page_badge} {ch_badge} {sec_badge}</div>"
                st.markdown(citation_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Supporting Source Excerpts Expander
        if excerpts:
            with st.expander(f"📑 View Supporting Excerpts from Manual ({len(excerpts)} retrieved chunks)", expanded=False):
                for e_idx, ex in enumerate(excerpts, 1):
                    meta = ex.get("metadata", {})
                    sim = ex.get("similarity", 0)
                    dist = ex.get("distance", 0)
                    st.markdown(f"**Excerpt {e_idx} — Page {meta.get('page')} | {meta.get('chapter')}**")
                    if meta.get("section") and meta.get("section") != "N/A":
                        st.caption(f"Section: `{meta.get('section')}` | Similarity Score: `{sim}` (Distance: `{dist}`)")
                    else:
                        st.caption(f"Similarity Score: `{sim}` (Distance: `{dist}`)")
                    st.text(ex.get("text", ""))
                    st.markdown("---")

# Legal Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Disclaimer:</strong> This assistant answers questions using the uploaded document. Verify important information against the original source. It is not a substitute for professional legal advice.
</div>
""", unsafe_allow_html=True)
