"""
SAP SD Intelligent Agent Pipeline v5 — Production
====================================================
Text-to-MQL architecture based on MongoDB + LangChain best practices.
Reference: https://www.mongodb.com/blog/post/technical/natural-language-agents-mongodb-text-mql-langchain

Two-model architecture:
  Model 1: question + schema → MongoDB query + ABAP query
           (sees NO data — only schema — can be Claude API or local Ollama)
  Model 2: executes query + formats answer
           (sees data — always local Ollama — data never leaves machine)

Related files:
  prompts.py     — all LLM prompts (edit prompts there, not here)
  mcp_server.py  — MongoDB MCP server for Claude Desktop integration

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import os
import re
from typing import Annotated, TypedDict, Literal
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ─── PROMPTS — all prompts live in prompts.py ─────────────────────────────────
from prompts import (
    QUERY_GEN_PROMPT,
    ABAP_PROMPT,
    RETRY_PROMPT,
    SEMANTIC_FORMAT_PROMPT,
    AGGREGATE_FORMAT_PROMPT,
)

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGO_URI")
DB_NAME     = os.getenv("DB_NAME")
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_1     = os.getenv("LLM_MODEL_1",   "llama3.1:8b")
MODEL_2     = os.getenv("LLM_MODEL_2",   "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL",   "mxbai-embed-large")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ─── CONNECTIONS ──────────────────────────────────────────────────────────────
mongo_client = MongoClient(MONGO_URI)
db           = mongo_client[DB_NAME]

USE_CLAUDE = os.getenv("USE_CLAUDE_MODEL1", "false").lower() == "true"
if USE_CLAUDE:
    llm_1       = ChatAnthropic(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        temperature=TEMPERATURE,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    model1_name = f"Claude API ({os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-6')})"
else:
    llm_1       = ChatOllama(model=MODEL_1, base_url=OLLAMA_URL, temperature=TEMPERATURE)
    model1_name = MODEL_1

llm_2      = ChatOllama(model=MODEL_2, base_url=OLLAMA_URL, temperature=TEMPERATURE)
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectordb   = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="sap_sd"
)

print(f"  Model 1 (query gen) : {model1_name}")
print(f"  Model 2 (execute)   : {MODEL_2}")
print(f"  Embed               : {EMBED_MODEL}")

# ─── SCHEMA — loaded once at startup ──────────────────────────────────────────
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}')

def build_full_schema() -> dict:
    """Load complete schema + date ranges at startup."""
    schema_doc = db["_schema"].find_one({"type": "schema_registry"}, {"_id": 0})
    tables     = schema_doc.get("tables", {}) if schema_doc else {}
    if not tables:
        tables = {
            c: list((db[c].find_one({}, {"_id": 0}) or {}).keys())
            for c in db.list_collection_names()
            if not c.startswith("_")
        }
    full = {}
    for table, columns in tables.items():
        sample    = db[table].find_one({}, {"_id": 0}) or {}
        non_empty = {k: v for k, v in sample.items() if v not in [None, "", 0, "0"]}

        date_ranges = {}
        for k, v in non_empty.items():
            if isinstance(v, str) and DATE_RE.match(v):
                q  = {k: {"$nin": [None, ""]}}
                mn = db[table].find_one(q, {k: 1, "_id": 0}, sort=[(k,  1)])
                mx = db[table].find_one(q, {k: 1, "_id": 0}, sort=[(k, -1)])
                if mn and mx and mn.get(k) and mx.get(k):
                    date_ranges[k] = (mn[k], mx[k])

        full[table] = {
            "columns":     columns,
            "sample":      non_empty,
            "date_ranges": date_ranges,
        }
    return full

FULL_SCHEMA = build_full_schema()

def get_relevant_schema(question: str) -> str:
    """Build schema context — all collections, exact field names + types + date ranges."""
    lines = ["SAP MongoDB Database Schema:\n"]
    for table, info in FULL_SCHEMA.items():
        sample = info["sample"]
        if not sample:
            continue
        sample_lines = []
        for k, v in sample.items():
            if isinstance(v, (int, float)):
                sample_lines.append(f'    "{k}": {v} (number)')
            elif isinstance(v, str) and len(v) < 40:
                sample_lines.append(f'    "{k}": {repr(v)} (string)')
        lines.append(f"Collection: {table}")
        lines.append(f"  Fields: {', '.join(chr(34)+c+chr(34) for c in info['columns'][:35])}")
        lines.append("  Sample:")
        lines.extend(sample_lines)
        for k, (lo, hi) in info.get("date_ranges", {}).items():
            lines.append(f'  Date range: "{k}" spans {lo} → {hi}')
        lines.append("")
    return "\n".join(lines)

# ─── STATE ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:     str
    mongo_code:   str
    abap_query:   str
    tool_result:  str
    exec_status:  str   # 'data' | 'empty' | 'error' | 'blocked'
    intent:       str
    final_answer: str
    messages:     Annotated[list, add_messages]

# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — QUERY GENERATOR (Model 1)
# Sees: question + schema only — never sees actual SAP data
# ══════════════════════════════════════════════════════════════════════════════
def node_generate_query(state: AgentState) -> AgentState:
    # Generate MongoDB query
    prompt = PromptTemplate(
        template=QUERY_GEN_PROMPT,
        input_variables=["schema", "question"]
    )
    raw = (prompt | llm_1 | StrOutputParser()).invoke({
        "schema":   get_relevant_schema(state["question"]),
        "question": state["question"]
    })
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    intent_m   = re.search(r'INTENT:\s*(aggregate|semantic)', raw, re.I)
    intent     = intent_m.group(1).lower() if intent_m else "semantic"
    code_m     = re.search(r'MONGO_CODE:\s*```python\s*(.*?)```', raw, re.DOTALL)
    mongo_code = code_m.group(1).strip() if code_m else ""

    # Generate ABAP query (documentation only — never executed)
    abap_prompt = PromptTemplate(
        template=ABAP_PROMPT,
        input_variables=["question"]
    )
    abap_raw   = (abap_prompt | llm_1 | StrOutputParser()).invoke({"question": state["question"]})
    abap_query = re.sub(r'<think>.*?</think>', '', abap_raw, flags=re.DOTALL).strip()

    return {**state,
            "intent":     intent,
            "mongo_code": mongo_code,
            "abap_query": abap_query,
            "messages":   [HumanMessage(content=state["question"])]}

# ══════════════════════════════════════════════════════════════════════════════
# NODE 2A — EXECUTE (always local — data never leaves machine)
# ══════════════════════════════════════════════════════════════════════════════
def safe_exec(code: str) -> tuple[str, str]:
    """Execute generated MongoDB code safely.
    Returns (result_text, status) where status ∈ {'data','empty','error','blocked'}
    """
    if not code:
        return "", "error"

    # Security: block dangerous operations
    for forbidden in ["import ", "open(", "exec(", "eval(", "__", "os.", "sys.", "subprocess"]:
        if forbidden in code:
            return f"Blocked: contains '{forbidden}'", "blocked"

    if "result" not in code:
        return "Code must assign to 'result'", "error"

    try:
        local = {
            "db": db, "list": list, "dict": dict, "sorted": sorted,
            "len": len, "round": round, "abs": abs, "sum": sum,
            "min": min, "max": max, "zip": zip, "range": range, "result": None
        }
        exec(code, {"__builtins__": {}}, local)
        result = local["result"]

        if result is None:
            return "", "empty"

        if isinstance(result, list):
            if not result:
                return "", "empty"

            # Null-group guard: catch grouping on non-existent field
            gf = re.search(r'["\']_id["\']\s*:\s*["\']\$([^"\']+)["\']', code)
            if gf and all(isinstance(r, dict) and r.get("_id") is None for r in result):
                return (f"Field '${gf.group(1)}' not found in this collection "
                        f"— all group keys are null. Check collection routing."), "error"

            lines = []
            for i, row in enumerate(result[:100], 1):
                if isinstance(row, dict):
                    txt = "  |  ".join(
                        f"{k}: {v}" for k, v in row.items()
                        if v not in [None, ""]
                    )
                    lines.append(f"{i}. {txt}")
                else:
                    lines.append(f"{i}. {row}")
            return "\n".join(lines), "data"

        return str(result), "data"

    except Exception as e:
        return f"Execution error: {e}", "error"

def node_execute(state: AgentState) -> AgentState:
    code = state.get("mongo_code", "")
    text, status = safe_exec(code)
    attempts = 0

    # Retry up to 2x on error using RETRY_PROMPT
    while status in ("error", "blocked") and code and attempts < 2:
        retry_prompt = PromptTemplate(
            template=RETRY_PROMPT,
            input_variables=["error", "code", "schema"]
        )
        fixed = (retry_prompt | llm_1 | StrOutputParser()).invoke({
            "error":  text,
            "code":   code,
            "schema": get_relevant_schema(state["question"])
        })
        fixed = re.sub(r'<think>.*?</think>', '', fixed, flags=re.DOTALL).strip()
        m = re.search(r'```python\s*(.*?)```', fixed, re.DOTALL)
        if not m:
            break
        code = m.group(1).strip()
        text, status = safe_exec(code)
        attempts += 1

    return {**state, "mongo_code": code, "tool_result": text, "exec_status": status}

# ══════════════════════════════════════════════════════════════════════════════
# NODE 2B — RAG SEARCH (semantic fallback only)
# ══════════════════════════════════════════════════════════════════════════════
def rrf(lists: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion for hybrid vector + text search."""
    scores, texts = {}, {}
    for results in lists:
        for rank, doc in enumerate(results, start=1):
            did = doc["id"]
            scores[did] = scores.get(did, 0) + 1.0 / (k + rank)
            texts[did]  = doc["text"]
    return sorted(
        [{"id": d, "text": texts[d]} for d in scores],
        key=lambda x: scores[x["id"]], reverse=True
    )

def node_rag_search(state: AgentState) -> AgentState:
    """Hybrid vector + BM25 search across all SAP collections."""
    question    = state["question"]
    collections = [c for c in db.list_collection_names() if not c.startswith("_")]
    vec, txt    = [], []

    # Vector search
    for t in collections:
        for doc in vectordb.similarity_search(question, k=3, filter={"table": t}):
            vec.append({"id": f"{t}_{hash(doc.page_content)}", "text": doc.page_content})

    # BM25 text search
    for t in collections:
        try:
            for doc in db[t].find(
                {"$text": {"$search": question}},
                {"score": {"$meta": "textScore"}, "_id": 0}
            ).sort([("score", {"$meta": "textScore"})]).limit(3):
                text = " | ".join(f"{k}: {v}" for k, v in doc.items() if k != "score" and v)
                txt.append({"id": f"{t}_t_{hash(text)}", "text": text[:400]})
        except Exception:
            pass

    fused   = rrf([vec, txt])[:10]
    context = "\n\n---\n\n".join(r["text"] for r in fused)
    return {**state, "tool_result": context, "exec_status": "data", "intent": "semantic"}

# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — FORMAT (Model 2 — always local)
# ══════════════════════════════════════════════════════════════════════════════
def node_format(state: AgentState) -> AgentState:
    status     = state.get("exec_status", "")
    mongo_code = state.get("mongo_code", "")
    abap       = state.get("abap_query", "")

    # Error/blocked — honest failure, never say "no data"
    if status in ("error", "blocked"):
        final = (
            f"⚠️ Could not execute this query (this does NOT mean there is no data):\n\n"
            f"{state['tool_result']}\n\n"
            f"---\nMongoDB Query:\n```python\n{mongo_code}\n```\n\nABAP Query:\n{abap}"
        )
        return {**state, "final_answer": final}

    # Empty result — honest no-data message with context
    if status == "empty":
        final = (
            f"No matching records were found. This may be because the requested "
            f"time period is outside the available data range, or no records match "
            f"the filter criteria.\n\n"
            f"---\nMongoDB Query:\n```python\n{mongo_code}\n```\n\nABAP Query:\n{abap}"
        )
        return {**state, "final_answer": final}

    # Semantic path — LLM answers from RAG context
    if state["intent"] == "semantic":
        prompt = PromptTemplate(
            template=SEMANTIC_FORMAT_PROMPT,
            input_variables=["context", "question"]
        )
        answer = (prompt | llm_2 | StrOutputParser()).invoke({
            "context":  state["tool_result"],
            "question": state["question"]
        })
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        # Fallback if model outputs only thinking tags
        if not answer:
            answer = state["tool_result"]

    # Aggregate path — LLM formats query results with fabrication guard
    else:
        prompt = PromptTemplate(
            template=AGGREGATE_FORMAT_PROMPT,
            input_variables=["data", "question"]
        )
        answer = (prompt | llm_2 | StrOutputParser()).invoke({
            "data":     state["tool_result"],
            "question": state["question"]
        })
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

        # Fabrication guard — normalize to 2dp to handle model rounding differences
        def extract_nums(s):
            nums = set()
            for x in re.findall(r'-?\d[\d,]*\.?\d*', s):
                x = x.replace(",", "")
                try:
                    nums.add(round(float(x), 2))
                except ValueError:
                    nums.add(x)
            return nums

        invented = extract_nums(answer) - extract_nums(state["tool_result"])
        if not answer.strip():
            answer = "Results:\n" + state["tool_result"]
        elif len(invented) > 2:
            # LLM invented numbers not in source — use raw results
            answer = "Results:\n" + state["tool_result"]

    final = (
        f"{answer}\n\n"
        f"---\n"
        f"MongoDB Query:\n```python\n{mongo_code}\n```\n\n"
        f"ABAP Query:\n{abap}"
    )
    return {**state, "final_answer": final}

# ─── ROUTER ───────────────────────────────────────────────────────────────────
def route(state: AgentState) -> Literal["node_execute", "node_rag_search"]:
    if state.get("intent") == "semantic" or not state.get("mongo_code"):
        return "node_rag_search"
    return "node_execute"

# ─── BUILD GRAPH ──────────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("node_generate_query", node_generate_query)
    g.add_node("node_execute",        node_execute)
    g.add_node("node_rag_search",     node_rag_search)
    g.add_node("node_format",         node_format)
    g.set_entry_point("node_generate_query")
    g.add_conditional_edges("node_generate_query", route, {
        "node_execute":    "node_execute",
        "node_rag_search": "node_rag_search",
    })
    g.add_edge("node_execute",    "node_format")
    g.add_edge("node_rag_search", "node_format")
    g.add_edge("node_format",     END)
    return g.compile()

agent = build_agent()

# ─── PUBLIC API ───────────────────────────────────────────────────────────────
def query_sap(question: str, verbose: bool = False) -> str:
    """Main entry point. Ask any SAP SD business question in plain English."""
    result = agent.invoke({
        "question":     question,
        "mongo_code":   "",
        "abap_query":   "",
        "tool_result":  "",
        "exec_status":  "",
        "intent":       "",
        "final_answer": "",
        "messages":     []
    })
    if verbose:
        print(f"  [Intent: {result['intent']} | Status: {result['exec_status']}]")
        print(f"  [MongoDB:\n{result['mongo_code']}]")
    return result["final_answer"]

# ─── TERMINAL ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  SAP SD Intelligent Agent  —  v5")
    print(f"  Model 1 : {model1_name}")
    print(f"  Model 2 : {MODEL_2}")
    print(f"  Database: {DB_NAME}")
    print("  Text-to-MQL | LangGraph | MongoDB Atlas")
    print("  MCP Server  : python mcp_server.py")
    print("="*65)
    print("Type 'verbose' to toggle debug | 'quit' to exit\n")

    verbose = False
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not q:
            continue
        if q.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if q.lower() == "verbose":
            verbose = not verbose
            print(f"  [Verbose: {'ON' if verbose else 'OFF'}]\n")
            continue
        print(f"\nAssistant: {query_sap(q, verbose)}\n")
        print("-" * 65 + "\n")