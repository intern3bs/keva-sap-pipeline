"""
SAP SD Intelligent Agent — Streamlit UI
=========================================
Works with both pipeline_v5.py and pipeline_v6.py.
Set PIPELINE_VERSION=v5 or v6 in .env to switch.

Run locally : streamlit run app.py
Run on Colab: see notebook cell comments

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import streamlit as st
import time
import os
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
    .main { padding: 1rem 2rem; }
    .sap-header {
        background: linear-gradient(135deg, #0070C0 0%, #00439C 100%);
        color: white; padding: 1.5rem 2rem;
        border-radius: 10px; margin-bottom: 1.5rem;
    }
    .sap-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .sap-header p  { color: #B0D4F1; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    .user-msg {
        background: #E3F2FD; border-left: 4px solid #0070C0;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .assistant-msg {
        background: #F8F9FA; border-left: 4px solid #28A745;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.78rem; font-weight: 600; margin-left: 8px;
    }
    .badge-aggregate { background: #D4EDDA; color: #155724; }
    .badge-semantic  { background: #D1ECF1; color: #0C5460; }
    .badge-error     { background: #F8D7DA; color: #721C24; }
    .sidebar-info {
        background: #F8F9FA; border-radius: 8px;
        padding: 0.8rem; margin-bottom: 1rem; font-size: 0.85rem;
    }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD PIPELINE ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SAP pipeline...")
def load_pipeline():
    version = os.getenv("PIPELINE_VERSION", "v6").lower()
    try:
        if version == "v5":
            from pipeline_v5 import query_sap, DB_NAME, MODEL_2, USE_CLAUDE
            model1 = f"Claude ({os.getenv('CLAUDE_MODEL','claude-sonnet-4-6')})" if USE_CLAUDE else os.getenv("LLM_MODEL_1","llama3.1:8b")
        else:
            from pipeline_v6 import query_sap, DB_NAME, MODEL_2, USE_CLAUDE
            model1 = f"Claude ({os.getenv('CLAUDE_MODEL','claude-sonnet-4-6')}) via MCP"
        return query_sap, None, {
            "version": version.upper(),
            "db":      DB_NAME,
            "model1":  model1,
            "model2":  MODEL_2,
        }
    except Exception as e:
        return None, str(e), {}

query_sap, load_error, info = load_pipeline()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 SAP SD Agent")
    st.markdown("---")

    # Pipeline info
    st.markdown("### Pipeline Status")
    if load_error:
        st.error(f"❌ Failed:\n{load_error}")
    else:
        st.success(f"✅ Pipeline {info['version']} ready")
        st.markdown(f"""
        <div class='sidebar-info'>
            <b>Architecture</b>: Pipeline {info['version']}<br>
            <b>Model 1</b>: {info['model1']}<br>
            <b>Model 2</b>: {info['model2']} (local)<br>
            <b>Database</b>: MongoDB Atlas — {info['db']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Example questions
    st.markdown("### 💡 Examples")
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
        "For each customer how many distinct materials ordered?",
        "Average net value per sales document type",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:25]}", use_container_width=True):
            st.session_state.pending_question = ex

    st.markdown("---")

    # Session stats
    if st.session_state.get("history"):
        h = st.session_state.history
        st.markdown("### 📊 Session")
        col1, col2 = st.columns(2)
        col1.metric("Asked", len(h))
        col2.metric("Success", sum(1 for x in h if x["ok"]))
        avg = sum(x["t"] for x in h) / len(h)
        st.metric("Avg time", f"{avg:.1f}s")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='sap-header'>
    <h1>🏭 SAP SD Intelligent Agent</h1>
    <p>Pipeline {info.get('version','V6')} · MongoDB Atlas · Claude + Llama + LangGraph · Keva Fragrances</p>
</div>
""", unsafe_allow_html=True)

# Init state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# ─── CHAT DISPLAY ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.info("👋 Ask any SAP SD business question or pick an example from the sidebar.")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'><b>You</b><br>{msg['content']}</div>",
                    unsafe_allow_html=True)
    else:
        content = msg["content"]
        elapsed = msg.get("t", 0)
        intent  = msg.get("intent", "aggregate")
        badge_cls = {"aggregate": "badge-aggregate",
                     "semantic":  "badge-semantic"}.get(intent, "badge-error")
        badge_lbl = {"aggregate": "📊 Aggregate",
                     "semantic":  "🔍 Semantic"}.get(intent, "⚠️ Error")

        parts  = content.split("---")
        answer = parts[0].strip()
        queries = "---".join(parts[1:]) if len(parts) > 1 else ""

        st.markdown(
            f"<div class='assistant-msg'>"
            f"<b>Assistant</b> "
            f"<span class='badge {badge_cls}'>{badge_lbl}</span>"
            f"<small style='color:#888;margin-left:8px'>⏱ {elapsed:.1f}s</small>"
            f"<br><br>{answer.replace(chr(10),'<br>')}"
            f"</div>",
            unsafe_allow_html=True
        )
        if queries.strip():
            with st.expander("📋 MongoDB + ABAP Queries"):
                st.markdown(queries)

# ─── INPUT ────────────────────────────────────────────────────────────────────
st.markdown("---")

default_q = st.session_state.pop("pending_question", "")

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Question", value=default_q,
        placeholder="e.g. Top 5 customers by total invoiced value",
        label_visibility="collapsed"
    )
with col2:
    ask = st.button("Ask →", type="primary", use_container_width=True)

# ─── PROCESS ──────────────────────────────────────────────────────────────────
if (ask or default_q) and question.strip() and query_sap:
    q = question.strip()
    st.session_state.messages.append({"role": "user", "content": q})

    with st.spinner("Querying SAP data..."):
        t0 = time.time()
        try:
            answer  = query_sap(q, verbose=False)
            elapsed = time.time() - t0
            ok      = True
            intent  = "aggregate" if "```python" in answer else "semantic"
        except Exception as e:
            answer  = f"⚠️ Error: {e}"
            elapsed = time.time() - t0
            ok      = False
            intent  = "error"

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "t": elapsed, "intent": intent
    })
    st.session_state.history.append({"t": elapsed, "ok": ok})
    st.rerun()

elif ask and not question.strip():
    st.warning("Please enter a question.")
elif ask and not query_sap:
    st.error("Pipeline not loaded. Check sidebar for error.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<small style='color:#999'>SAP SD Intelligent Agent · "
    f"Pipeline {info.get('version','V6')} · MongoDB Atlas · "
    f"Keva Fragrances Internship · Rohit Kumar · {datetime.now().year}</small>",
    unsafe_allow_html=True
)