"""
app.py - High-End Streamlit Frontend for Designs Law Assistant

A polished, state-of-the-art web application for answering questions on India's
Manual of Designs Practice & Procedure using LangChain, ChromaDB, and Gemini.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    DesignsLawRAG,
    DEFAULT_CHROMA_DIR,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_RETRIEVAL_THRESHOLD,
    DEFAULT_GEMINI_MODEL
)

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Designs Law Assistant | RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base CSS (Fonts & Core Components)
BASE_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Fonts & Smoothing */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Hero Banner */
    .hero-banner {
        border-radius: 18px;
        padding: 30px 34px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid #38bdf8;
        color: #38bdf8;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 10px #38bdf8;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }
    .hero-title {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 2.7rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.01em !important;
        margin: 6px 0 12px 0 !important;
        color: #38bdf8 !important;
        text-shadow: 0 2px 14px rgba(56, 189, 248, 0.45);
        display: block !important;
        line-height: 1.2 !important;
    }

    /* Section Label */
    .section-label {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 20px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Cards */
    .rag-card {
        border-radius: 16px;
        padding: 24px 28px;
        margin: 18px 0;
        transition: all 0.2s ease;
    }
    .rag-q-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .rag-ans-body {
        font-size: 1.0rem;
        line-height: 1.65;
        margin-bottom: 16px;
    }

    /* Badges */
    .badge-not-found {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #f87171;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 5px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .badge-page {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.28);
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 3px 10px;
        border-radius: 6px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-chapter {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-weight: 500;
        font-size: 0.82rem;
        padding: 3px 10px;
        border-radius: 6px;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Sidebar Glass Card */
    .sidebar-stat-card {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
    }

    /* Disclaimer */
    .footer-disclaimer {
        border-radius: 12px;
        padding: 14px 20px;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 36px;
        text-align: center;
    }
</style>
"""

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

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "🌙 Dark Mode"

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🎨 Display Theme")
    theme_choice = st.radio(
        "Theme Mode",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state.theme_mode == "🌙 Dark Mode" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state.theme_mode = theme_choice
    is_light = (theme_choice == "☀️ Light Mode")

    st.markdown("---")
    st.markdown("### ⚖️ Assistant Controls")

    # API Key Input
    env_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        help="API Key is loaded from .env by default. You can also override it here."
    )

    if api_key:
        st.success("API Key Active", icon="⚡")
    else:
        st.warning("API Key Missing! Set in .env or paste above.", icon="🔑")

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")

    rag = get_rag_pipeline(api_key)
    if api_key:
        rag.set_api_key(api_key)

    index_status = rag.get_index_status()

    # Auto-index default PDF if not already indexed
    if not index_status["is_indexed"] and os.path.exists(DEFAULT_PDF_PATH):
        with st.spinner("Indexing official 87-page Manual..."):
            try:
                rag.index_document(DEFAULT_PDF_PATH)
                index_status = rag.get_index_status()
                st.toast("Manual indexed successfully!", icon="✅")
            except Exception as e:
                st.error(f"Error during initial indexing: {e}")

    if index_status["is_indexed"]:
        meta = index_status.get("metadata", {})
        pages = meta.get("total_pages", 87)
        chunks = index_status.get("total_chunks", 0)

        st.markdown(f"""
        <div class="sidebar-stat-card">
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; color:#38bdf8; font-weight:700;">Vector Store Active</div>
            <div style="display:flex; justify-content:space-between; margin-top:8px;">
                <div><span style="font-size:1.4rem; font-weight:800; color:#fff;">{pages}</span><br><span style="font-size:0.78rem; color:#94a3b8;">Total Pages</span></div>
                <div><span style="font-size:1.4rem; font-weight:800; color:#38bdf8;">{chunks}</span><br><span style="font-size:0.78rem; color:#94a3b8;">Text Chunks</span></div>
            </div>
            <div style="margin-top:10px; font-size:0.75rem; color:#64748b; word-break:break-all;">Manual_of_Designs_Practice_and_Procedure.pdf</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No document currently indexed. Please upload or index below.", icon="📁")

    # File uploader
    with st.expander("📤 Upload / Change PDF"):
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        if uploaded_file is not None:
            save_path = os.path.join(os.path.dirname(__file__), "data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")
            if st.button("Index This PDF", type="primary"):
                with st.spinner("Extracting and vectorizing document..."):
                    try:
                        res = rag.index_document(save_path, force_reindex=True)
                        st.success(f"Indexed {res['total_pages']} pages into {res['total_chunks']} chunks!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

    if index_status["is_indexed"]:
        if st.button("🔄 Force Re-index"):
            target_path = DEFAULT_PDF_PATH if os.path.exists(DEFAULT_PDF_PATH) else None
            if target_path:
                with st.spinner("Re-indexing document..."):
                    rag.index_document(target_path, force_reindex=True)
                    st.success("Re-indexing complete!")
                    st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Precision Parameters")
    retrieval_k = st.slider("Top Excerpts (K)", min_value=1, max_value=8, value=DEFAULT_RETRIEVAL_K, step=1)
    retrieval_threshold = st.slider(
        "Relevance Cutoff (Distance)",
        min_value=0.40,
        max_value=1.00,
        value=DEFAULT_RETRIEVAL_THRESHOLD,
        step=0.05,
        help="Cosine distance threshold. If best chunk distance exceeds this, 'Not found' is strictly returned."
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Dynamic Theme Injection
if is_light:
    THEME_CSS = """<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"], .main, .block-container {
    background-color: #f8fafc !important;
    background: #f8fafc !important;
    background-image: none !important;
    color: #0f172a !important;
}
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #cbd5e1 !important;
}
section[data-testid="stSidebar"] * {
    color: #0f172a !important;
}
.rag-card {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
}
.rag-card.verified {
    border-left: 5px solid #0284c7 !important;
}
.rag-card.not-found-card {
    border-left: 5px solid #ef4444 !important;
    background-color: #fef2f2 !important;
}
.rag-q-title {
    color: #0f172a !important;
}
.rag-ans-body {
    color: #1e293b !important;
}
.section-label {
    color: #0f172a !important;
}
.sidebar-stat-card {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}
.footer-disclaimer {
    background-color: #e2e8f0 !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
}
.badge-chapter {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}
.stTextInput input, .stTextArea textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}
</style>"""
else:
    THEME_CSS = """<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
    background-color: #0b1120 !important;
    background: #0b1120 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(30, 58, 138, 0.25) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(14, 116, 144, 0.18) 0px, transparent 50%) !important;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
}
.rag-card {
    background: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
}
.rag-card.verified {
    border-left: 4px solid #38bdf8 !important;
}
.rag-card.not-found-card {
    border-left: 4px solid #ef4444 !important;
    background: rgba(28, 16, 24, 0.7) !important;
}
.rag-q-title {
    color: #f8fafc !important;
}
.rag-ans-body {
    color: #cbd5e1 !important;
}
.section-label {
    color: #e2e8f0 !important;
}
.sidebar-stat-card {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
}
.footer-disclaimer {
    background: rgba(15, 23, 42, 0.6) !important;
    color: #64748b !important;
}
.badge-chapter {
    background: rgba(148, 163, 184, 0.12) !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    color: #cbd5e1 !important;
}
</style>"""

st.markdown(BASE_CSS.strip() + THEME_CSS.strip(), unsafe_allow_html=True)

# Main Hero Header
st.markdown("""
<div class="hero-banner" style="border: 2px solid #2563eb !important; background: linear-gradient(135deg, #091a38 0%, #0e2752 50%, #173b75 100%) !important; padding: 28px 32px !important; border-radius: 16px !important; box-shadow: 0 12px 30px -8px rgba(37, 99, 235, 0.35) !important;">
    <div class="hero-tag" style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; font-weight: 700;">
        <span class="pulse-dot"></span>
        Grounded Legal RAG &bull; India IP Office (87-Page Manual)
    </div>
    <h1 style="color: #38bdf8 !important; font-family: 'Outfit', sans-serif !important; font-size: 2.6rem !important; font-weight: 900 !important; margin: 8px 0 10px 0 !important; letter-spacing: -0.01em !important; text-shadow: 0 2px 16px rgba(56, 189, 248, 0.4) !important;">
        ⚖️ DESIGNS LAW ASSISTANT
    </h1>
    <p style="color: #e2e8f0 !important; font-size: 1.05rem !important; margin: 0 !important; line-height: 1.5 !important;">
        AI-Powered Legal Assistant Grounded in India's <strong>Manual of Designs Practice & Procedure</strong>.
        Verified Citations &bull; Zero Hallucinations &bull; Exact PDF Page & Section References.
    </p>
</div>
""", unsafe_allow_html=True)

# Clickable Assignment Questions
st.markdown('<div class="section-label">🎯 Clickable Assignment Questions</div>', unsafe_allow_html=True)
st.caption("Click any required test question below to run the actual retrieval and grounding pipeline instantly:")

ASSIGNMENT_QUESTIONS = [
    ("Q1 • Copyright Duration", "How long does a proprietor have copyright in a registered design, and can it be extended?"),
    ("Q2 • Classification System", "Which classification system is used to register designs?"),
    ("Q3 • Negative Test: Trademark", "What is the government fee to register a trademark?"),
    ("Q4 • Negative Test: Controller", "Who is the current Controller General of Patents?"),
    ("Q5 • Design Registration Fee", "What is the fee to register a design?")
]

col1, col2 = st.columns([1, 1])

clicked_sample = None
for idx, (label, q_text) in enumerate(ASSIGNMENT_QUESTIONS):
    target_col = col1 if idx < 3 else col2
    with target_col:
        if st.button(f"**{label}**\n\n_{q_text}_", key=f"sample_q_{idx}", use_container_width=True):
            clicked_sample = q_text

# Text input for custom questions
st.markdown('<div class="section-label">💬 Custom Question Input</div>', unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])

with col_input:
    user_question = st.text_input(
        "Ask a custom question:",
        placeholder="Type any question about Indian Industrial Designs Law...",
        label_visibility="collapsed",
        key="user_query_input"
    )

with col_btn:
    ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

url_q = st.query_params.get("q", None)
query_to_run = url_q or clicked_sample or (user_question.strip() if ask_btn else None)

# Execute query through RAG pipeline
if query_to_run:
    if not api_key:
        st.error("Please configure your Google Gemini API key in the sidebar before asking questions.")
    elif not index_status["is_indexed"]:
        st.error("Document is not yet indexed in ChromaDB. Please check the sidebar to index the PDF.")
    else:
        with st.spinner("Retrieving verified excerpts & generating grounded answer..."):
            result = rag.answer_question(
                question=query_to_run,
                k=retrieval_k,
                threshold=retrieval_threshold
            )
            # Avoid duplicate consecutive insertions
            if not st.session_state.chat_history or st.session_state.chat_history[0].get("question") != query_to_run:
                st.session_state.chat_history.insert(0, result)

# Display Responses and History
if st.session_state.chat_history:
    st.markdown('<div class="section-label">📋 Answers & Verified Sources</div>', unsafe_allow_html=True)

    for idx, item in enumerate(st.session_state.chat_history):
        q = item.get("question", "")
        ans = item.get("answer", "")
        is_nf = item.get("is_not_found", False)
        is_err = item.get("is_error", False)
        citations = item.get("citations", [])
        excerpts = item.get("source_excerpts", [])
        reason = item.get("reason", "")

        card_class = "not-found-card" if is_nf else "verified"

        st.markdown(f"""
        <div class="rag-card {card_class}">
            <div class="rag-q-title">📌 {q}</div>
        """, unsafe_allow_html=True)

        if is_nf:
            st.markdown("""
            <div class="badge-not-found">
                🔒 Not found
            </div>
            """, unsafe_allow_html=True)
            if reason:
                st.info(f"**Verification Grounding:** {reason}")
            st.markdown(f"<div class='rag-ans-body'><strong>Result:</strong> {ans}</div>", unsafe_allow_html=True)
        elif is_err:
            st.error(ans)
        else:
            st.markdown(f"<div class='rag-ans-body'>{ans}</div>", unsafe_allow_html=True)

            # Source Badges
            if citations:
                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#94a3b8; margin-top:14px; margin-bottom:6px;'>📖 VERIFIED CITATIONS:</div>", unsafe_allow_html=True)
                citation_html = "<div style='display:flex; flex-wrap:wrap; gap:6px;'>"
                for c in citations:
                    page_badge = f"<span class='badge-page'>📄 PDF Page {c.get('page', 'N/A')}</span>"
                    ch_badge = f"<span class='badge-chapter'>📚 {c.get('chapter', 'Chapter N/A')}</span>"
                    sec = c.get('section')
                    sec_badge = f"<span class='badge-chapter'>⚖️ Section {sec}</span>" if sec and sec != "N/A" else ""
                    citation_html += f"<div>{page_badge} {ch_badge} {sec_badge}</div>"
                citation_html += "</div>"
                st.markdown(citation_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Supporting Excerpts Expander
        if excerpts:
            with st.expander(f"📑 Supporting Excerpts from Manual ({len(excerpts)} retrieved chunks)", expanded=(idx == 0)):
                for e_idx, ex in enumerate(excerpts, 1):
                    meta = ex.get("metadata", {})
                    sim = ex.get("similarity", 0)
                    dist = ex.get("distance", 0)
                    st.markdown(f"**Excerpt {e_idx} &bull; PDF Page {meta.get('page')} &bull; {meta.get('chapter')}**")
                    if meta.get("section") and meta.get("section") != "N/A":
                        st.caption(f"Section: `{meta.get('section')}` | Similarity Match: `{round(sim * 100, 1)}%` (Distance: `{dist}`)")
                    else:
                        st.caption(f"Similarity Match: `{round(sim * 100, 1)}%` (Distance: `{dist}`)")
                    st.code(ex.get("text", ""), language="text")

# Legal Disclaimer Footer
st.markdown("""
<div class="footer-disclaimer">
    <strong>⚠️ Disclaimer:</strong> This assistant answers questions exclusively using excerpts from India's 
    <em>Manual of Designs Practice & Procedure</em>. Verify critical matters against the official gazette. It is not a substitute for legal counsel.
</div>
""", unsafe_allow_html=True)
