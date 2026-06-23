"""
SAP SD Intelligent Agent — Streamlit UI
=========================================
Production web interface for the SAP ERP RAG Pipeline.

Run locally:
  streamlit run app.py

Run on Colab (via ngrok):
  !pip install streamlit pyngrok
  from pyngrok import ngrok
  ngrok.set_auth_token("YOUR_NGROK_TOKEN")
  public_url = ngrok.connect(8501)
  print(f"App URL: {public_url}")
  !streamlit run app.py --server.port 8501 &

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import streamlit as st
import time
import re
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAP SD Intelligent Agent",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main { padding: 1rem 2rem; }

    /* Header */
    .sap-header {
        background: linear-gradient(135deg, #0070C0 0%, #00439C 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .sap-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .sap-header p  { color: #B0D4F1; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Chat messages */
    .user-msg {
        background: #E3F2FD;
        border-left: 4px solid #0070C0;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .assistant-msg {
        background: #F8F9FA;
        border-left: 4px solid #28A745;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .error-msg {
        background: #FFF3CD;
        border-left: 4px solid #FFC107;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #0070C0; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }

    /* Query badge */
    .intent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .intent-aggregate { background: #D4EDDA; color: #155724; }
    .intent-semantic  { background: #D1ECF1; color: #0C5460; }
    .intent-error     { background: #F8D7DA; color: #721C24; }

    /* Sidebar */
    .sidebar-section {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Hide streamlit branding */
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD PIPELINE ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SAP pipeline...")
def load_pipeline():
    """Load pipeline once and cache. Returns (query_sap_fn, error_msg)."""
    try:
        from pipeline_v5 import query_sap, DB_NAME, MODEL_2, USE_CLAUDE
        import os
        model1 = f"Claude ({os.getenv('CLAUDE_MODEL','claude-sonnet-4-6')})" if USE_CLAUDE else os.getenv("LLM_MODEL_1","llama3.1:8b")
        return query_sap, None, {
            "db":     DB_NAME,
            "model1": model1,
            "model2": MODEL_2,
        }
    except Exception as e:
        return None, str(e), {}

query_sap, load_error, pipeline_info = load_pipeline()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 SAP SD Agent")
    st.markdown("---")

    # Pipeline status
    st.markdown("### Pipeline Status")
    if load_error:
        st.error(f"❌ Pipeline failed to load:\n{load_error}")
    else:
        st.success("✅ Pipeline ready")
        st.markdown(f"""
        <div class='sidebar-section'>
            <b>Model 1</b> (Query Gen)<br>
            <small>{pipeline_info.get('model1','—')}</small><br><br>
            <b>Model 2</b> (Format)<br>
            <small>{pipeline_info.get('model2','—')}</small><br><br>
            <b>Database</b><br>
            <small>MongoDB Atlas — {pipeline_info.get('db','—')}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Example questions
    st.markdown("### 💡 Example Questions")
    examples = [
        "Top 5 customers by total invoiced value",
        "Top selling product by quantity",
        "3 products with the least margins",
        "Total billing value for Sales Organization 1000",
        "Which billing type appears most frequently?",
        "Average net value per billing document",
        "Which material group has worst average margin?",
        "Top 5 most profitable materials",
        "Revenue per distribution channel",
        "Which sales office generates most revenue?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state.pending_question = ex

    st.markdown("---")

    # Session stats
    if "history" in st.session_state and st.session_state.history:
        st.markdown("### 📊 Session Stats")
        total    = len(st.session_state.history)
        success  = sum(1 for h in st.session_state.history if h.get("status") == "success")
        avg_time = sum(h.get("time", 0) for h in st.session_state.history) / total if total else 0
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", total)
        with col2:
            st.metric("Success", f"{success}/{total}")
        st.metric("Avg Time", f"{avg_time:.1f}s")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class='sap-header'>
    <h1>🏭 SAP SD Intelligent Agent</h1>
    <p>Ask business questions in plain English — powered by MongoDB Atlas + Claude + LangGraph</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# ─── CHAT DISPLAY ─────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info("👋 Ask any SAP SD business question — or pick an example from the sidebar.")
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='user-msg'>
                    <b>You</b><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Parse answer and metadata
                content    = msg["content"]
                intent     = msg.get("intent", "")
                elapsed    = msg.get("time", 0)
                status     = msg.get("status", "")

                # Badge
                if intent == "aggregate":
                    badge = "<span class='intent-badge intent-aggregate'>📊 Aggregate</span>"
                elif intent == "semantic":
                    badge = "<span class='intent-badge intent-semantic'>🔍 Semantic</span>"
                else:
                    badge = "<span class='intent-badge intent-error'>⚠️ Error</span>"

                # Split answer from queries
                parts      = content.split("---")
                answer     = parts[0].strip()
                queries    = "---".join(parts[1:]) if len(parts) > 1 else ""

                st.markdown(f"""
                <div class='assistant-msg'>
                    <b>Assistant</b> {badge}
                    <small style='color:#888; margin-left:8px'>⏱ {elapsed:.1f}s</small>
                    <br><br>{answer.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

                # Show queries in expander
                if queries.strip():
                    with st.expander("📋 View MongoDB + ABAP Queries"):
                        st.markdown(queries)

# ─── INPUT ────────────────────────────────────────────────────────────────────
st.markdown("---")

# Handle example button clicks
default_q = ""
if "pending_question" in st.session_state:
    default_q = st.session_state.pop("pending_question")

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Ask a SAP business question:",
        value=default_q,
        placeholder="e.g. Top 5 customers by total invoiced value",
        label_visibility="collapsed",
        key="question_input"
    )
with col2:
    ask_btn = st.button("Ask →", type="primary", use_container_width=True)

# ─── PROCESS QUESTION ─────────────────────────────────────────────────────────
if (ask_btn or default_q) and question.strip() and query_sap:
    q = question.strip()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": q})

    # Run pipeline
    with st.spinner("Querying SAP data..."):
        start = time.time()
        try:
            answer  = query_sap(q, verbose=False)
            elapsed = time.time() - start
            status  = "success"

            # Detect intent from answer
            if "⚠️" in answer:
                intent = "error"
            elif "No matching records" in answer:
                intent = "empty"
            else:
                # Check if MongoDB query was generated
                intent = "aggregate" if "```python" in answer else "semantic"

        except Exception as e:
            answer  = f"⚠️ Pipeline error: {e}"
            elapsed = time.time() - start
            intent  = "error"
            status  = "error"

    # Add assistant message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "intent":  intent,
        "time":    elapsed,
        "status":  status,
    })

    # Update history stats
    st.session_state.history.append({
        "question": q,
        "time":     elapsed,
        "status":   status,
        "intent":   intent,
    })

    st.rerun()

elif ask_btn and not question.strip():
    st.warning("Please enter a question.")

elif ask_btn and not query_sap:
    st.error("Pipeline not loaded. Check the error in the sidebar.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#999'>SAP SD Intelligent Agent v5 · "
    "MongoDB Atlas · Claude + LangGraph · Keva Fragrances Internship · "
    f"Rohit Kumar · {datetime.now().strftime('%Y')}</small>",
    unsafe_allow_html=True
)