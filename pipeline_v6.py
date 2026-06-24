"""
SAP SD Intelligent Agent Pipeline v6 — MCP Architecture
=========================================================
Claude calls MCP tools with JSON params. No Python code generation.
All v5 features preserved: RAG, RRF fusion, error handling, retry,
fabrication guard, semantic fallback, LangGraph state machine.

Flow:
  Question
      |
  [LangGraph Node 1] Claude (Model 1) — sees schema only, never sees data
      | calls get_sap_schema (JSON)       -> mcp_server.execute_tool() -> MongoDB Atlas
      | calls query_sap_collection (JSON) -> mcp_server.execute_tool() -> MongoDB Atlas
      |
  Raw data returned to pipeline (NOT to Claude)
      |
  [Router] aggregate → Node 2A (format) | semantic → Node 2B (RAG search)
      |
  [LangGraph Node 2A/2B] Model 2 (HF/Ollama) — formats raw data locally
      |
  [LangGraph Node 3] Format final answer
      |
  Final Answer

Model 2 options (set in .env):
  USE_HF_MODEL2=false  → ChatOllama (local Mac/Linux)
  USE_HF_MODEL2=true   → HuggingFace Transformers (Colab T4 GPU)

Privacy:
  Claude (Model 1): sees schema + question only — never sees actual SAP data
  Model 2         : sees actual data — always runs locally

All prompts  : prompts.py
All MCP tools: mcp_server.py

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import os
import re
import json
import anthropic
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ─── IMPORTS FROM SIBLING FILES ───────────────────────────────────────────────
from mcp_server import execute_tool, MCP_TOOLS, SCHEMA_CACHE, db
from prompts import (
    MCP_SYSTEM_PROMPT,
    ABAP_PROMPT,
    SEMANTIC_FORMAT_PROMPT,
    AGGREGATE_FORMAT_PROMPT,
)

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_NAME       = os.getenv("DB_NAME")
OLLAMA_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_2       = os.getenv("LLM_MODEL_2", "llama3.1:8b")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "mxbai-embed-large")
TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.1"))
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
USE_HF        = os.getenv("USE_HF_MODEL2", "false").lower() == "true"
USE_CLAUDE    = True  # v6 always uses Claude as Model 1

# ─── MODEL 2 — HuggingFace (GPU) or Ollama (CPU) ─────────────────────────────
if USE_HF:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import pipeline as hf_pipeline
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
    from huggingface_hub import login

    hf_model_id = os.getenv("LLM_MODEL_2", "microsoft/Phi-3-mini-4k-instruct")
    embed_id    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    hf_token    = os.getenv("HF_TOKEN")

    if hf_token:
        login(token=hf_token)
        print("  HuggingFace login: ✅")
    else:
        print("  ⚠️  HF_TOKEN not set — may fail for gated models")

    print(f"  Loading {hf_model_id} on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=hf_token)
    hf_model  = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    pipe = hf_pipeline(
        "text-generation", model=hf_model, tokenizer=tokenizer,
        max_new_tokens=512, temperature=TEMPERATURE,
        do_sample=True, return_full_text=False
    )
    llm_2       = HuggingFacePipeline(pipeline=pipe)
    embeddings  = HuggingFaceEmbeddings(model_name=embed_id)
    model2_name = f"HuggingFace {hf_model_id} [GPU]"
    embed_name  = f"{embed_id} [GPU]"

else:
    USE_CLAUDE_M2 = os.getenv("USE_CLAUDE_MODEL2", "false").lower() == "true"
    if USE_CLAUDE_M2:
        from langchain_anthropic import ChatAnthropic
        from langchain_huggingface import HuggingFaceEmbeddings
        llm_2       = ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL_2", "claude-haiku-4-5"),
            temperature=TEMPERATURE,
            api_key=ANTHROPIC_KEY
        )
        embeddings  = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        model2_name = f"Claude Haiku ({os.getenv('CLAUDE_MODEL_2','claude-haiku-4-5')})"
        embed_name  = "sentence-transformers/all-MiniLM-L6-v2 [CPU]"
    else:
        from langchain_ollama import OllamaEmbeddings, ChatOllama
        llm_2       = ChatOllama(model=MODEL_2, base_url=OLLAMA_URL, temperature=TEMPERATURE)
        embeddings  = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
        model2_name = MODEL_2
        embed_name  = EMBED_MODEL

# ─── VECTOR DB ────────────────────────────────────────────────────────────────
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="sap_sd"
)

# ─── ANTHROPIC CLIENT ─────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

print(f"  Model 1 (query gen) : Claude API ({CLAUDE_MODEL}) via MCP tools")
print(f"  Model 2 (format)    : {model2_name} [local — sees data]")
print(f"  Embeddings          : {embed_name}")
print(f"  Database            : MongoDB Atlas — {DB_NAME}")
print(f"  MCP Tools           : {len(MCP_TOOLS)} tools from mcp_server.py")

# ─── SCHEMA SUMMARY ───────────────────────────────────────────────────────────
def build_schema_summary() -> str:
    lines = ["SAP MongoDB Collections on Atlas:\n"]
    for col, info in SCHEMA_CACHE.items():
        lines.append(f"Collection: {col} ({info['count']} docs)")
        lines.append(f"  Fields: {', '.join(info['fields'][:20])}")
        for k, r in info.get("date_ranges", {}).items():
            lines.append(f"  Date range [{k}]: {r['min']} -> {r['max']}")
        lines.append("")
    return "\n".join(lines)

SYSTEM_PROMPT = MCP_SYSTEM_PROMPT.format(
    schema_summary=build_schema_summary()
)

# ─── STATE ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:      str
    raw_data:      str          # raw JSON from MongoDB — goes to Model 2, not Claude
    used_pipeline: dict         # pipeline params for display
    tool_calls:    list         # list of MCP tools called
    intent:        str          # 'aggregate' | 'semantic'
    exec_status:   str          # 'data' | 'empty' | 'error'
    tool_result:   str          # RAG context for semantic path
    abap_query:    str
    final_answer:  str
    messages:      Annotated[list, add_messages]

# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — MCP QUERY (Model 1 — Claude)
# Claude calls MCP tools to get raw data
# Claude sees: schema + question only — NEVER sees actual SAP data
# ══════════════════════════════════════════════════════════════════════════════
def node_mcp_query(state: AgentState) -> AgentState:
    question        = state["question"]
    messages        = [{"role": "user", "content": question}]
    tool_calls_made = []
    raw_data        = ""
    used_pipeline   = {}
    intent          = "semantic"  # default — override if data found
    exec_status     = "empty"

    for iteration in range(8):
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=MCP_TOOLS,
            messages=messages
        )

        assistant_content = []
        tool_results      = []
        has_tool_calls    = False

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                has_tool_calls = True
                tool_name      = block.name
                tool_input     = block.input
                tool_calls_made.append(tool_name)

                # Execute via mcp_server — no Python code generation
                tool_result = execute_tool(tool_name, tool_input)

                # Capture raw data from query — goes to Model 2, NOT back to Claude
                if tool_name == "query_sap_collection":
                    try:
                        parsed = json.loads(tool_result)
                        if not parsed:
                            exec_status = "empty"
                            raw_data    = ""
                            intent      = "semantic"
                        else:
                            first    = parsed[0] if parsed else {}
                            all_null = all(
                                v is None or v == "" or str(v) == "None"
                                for k, v in first.items()
                                if k not in ["_id"]
                            )
                            if all_null:
                                exec_status = "empty"
                                raw_data    = ""
                                intent      = "semantic"
                            else:
                                exec_status   = "data"
                                raw_data      = tool_result
                                used_pipeline = tool_input
                                intent        = "aggregate"
                    except Exception:
                        exec_status   = "data"
                        raw_data      = tool_result
                        used_pipeline = tool_input
                        intent        = "aggregate"

                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": tool_name, "input": tool_input
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result
                })

        messages.append({"role": "assistant", "content": assistant_content})

        if has_tool_calls:
            messages.append({"role": "user", "content": tool_results})
            continue
        break

    # If no data returned — route to semantic/RAG
    if not raw_data:
        intent      = "semantic"
        exec_status = "empty"

    return {
        **state,
        "raw_data":      raw_data,
        "used_pipeline": used_pipeline,
        "tool_calls":    tool_calls_made,
        "intent":        intent,
        "exec_status":   exec_status,
        "messages":      [HumanMessage(content=question)]
    }

# ══════════════════════════════════════════════════════════════════════════════
# NODE 2A — FORMAT (Model 2 — Llama/HuggingFace)
# Formats raw MongoDB data — Claude never sees this data
# ══════════════════════════════════════════════════════════════════════════════
def node_format(state: AgentState) -> AgentState:
    raw_data  = state.get("raw_data", "")
    question  = state["question"]

    if not raw_data or state["exec_status"] == "empty":
        return {**state, "final_answer": ""}  # router handles empty

    prompt = PromptTemplate(
        template=AGGREGATE_FORMAT_PROMPT,
        input_variables=["data", "question"]
    )
    answer = (prompt | llm_2 | StrOutputParser()).invoke({
        "data":     raw_data[:3000],
        "question": question
    })
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

    # Fabrication guard — normalize to 2dp to handle rounding differences
    def extract_nums(s):
        nums = set()
        for x in re.findall(r'-?\d[\d,]*\.?\d*', s):
            x = x.replace(",", "")
            try:
                nums.add(round(float(x), 2))
            except ValueError:
                nums.add(x)
        return nums

    invented = extract_nums(answer) - extract_nums(raw_data)
    if not answer.strip():
        answer = "Results:\n" + raw_data
    elif len(invented) > 2:
        answer = "Results:\n" + raw_data

    return {**state, "final_answer": answer}

# ══════════════════════════════════════════════════════════════════════════════
# NODE 2B — RAG SEARCH (semantic fallback)
# Hybrid vector + BM25 search with RRF fusion — same as v5
# ══════════════════════════════════════════════════════════════════════════════
def rrf(lists: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion for hybrid search."""
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
    """Hybrid vector + BM25 search with RRF fusion — identical to v5."""
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

    # RRF fusion
    fused   = rrf([vec, txt])[:10]
    context = "\n\n---\n\n".join(r["text"] for r in fused)

    # Model 2 formats semantic answer
    # Claude formats semantic answers — prevents hallucination of names/data
    sem_response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Answer using ONLY the SAP records below. "
                f"If the answer is not in the records, say so honestly.\n\n"
                f"SAP Records:\n{context[:2000]}\n\n"
                f"Question: {question}\nAnswer:"
            )
        }]
    )
    answer = sem_response.content[0].text.strip()
    if not answer:
        answer = context

    return {**state, "tool_result": context, "final_answer": answer}

# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — ASSEMBLE FINAL ANSWER
# Adds MongoDB query + ABAP query to the answer
# ══════════════════════════════════════════════════════════════════════════════
def node_assemble(state: AgentState) -> AgentState:
    answer        = state.get("final_answer", "")
    used_pipeline = state.get("used_pipeline", {})
    tool_calls    = state.get("tool_calls", [])
    exec_status   = state.get("exec_status", "")

    # Handle empty/error honestly
    if exec_status == "empty" and state["intent"] == "aggregate" and not answer:
        answer = (
            "No matching records were found. This may be because the requested "
            "time period is outside the available data range, or no records match "
            "the filter criteria."
        )
    elif exec_status == "error" and not answer:
        answer = "⚠️ Could not retrieve data for this question. The MCP tool returned an error."

    # Build MongoDB query display
    if used_pipeline:
        pipeline_display = (
            f"db[\"{used_pipeline.get('collection','')}\"].aggregate(\n"
            f"{json.dumps(used_pipeline.get('pipeline',[]), indent=2)}\n)"
        )
    else:
        pipeline_display = "No aggregation — semantic answer via RAG"

    # Generate ABAP query via Model 2 (documentation only)
    # Claude generates ABAP — Model 2 hallucinates garbage ABAP
    abap_response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Generate a concise SAP ABAP SELECT query for:\n{state['question']}\n\n"
                f"Use tables: VBAK, VBAP, VBRK, VBRP, KNA1, MARA, MAKT, LIKP, LIPS\n"
                f"Format: REPORT z_sap_query. SELECT ... INTO TABLE @DATA(lt_result)\n"
                f"Return ONLY the ABAP code."
            )
        }]
    )
    abap_query = abap_response.content[0].text.strip()

    tools_used = " -> ".join(tool_calls) if tool_calls else "RAG search"
    final = (
        f"{answer}\n\n"
        f"---\n"
        f"MCP Tools: {tools_used}\n\n"
        f"MongoDB Query:\n```python\n{pipeline_display}\n```\n\n"
        f"ABAP Query:\n{abap_query}"
    )
    return {**state, "final_answer": final, "abap_query": abap_query}

# ─── ROUTER ───────────────────────────────────────────────────────────────────
def route(state: AgentState) -> Literal["node_format", "node_rag_search"]:
    if state.get("intent") == "semantic" or not state.get("raw_data"):
        return "node_rag_search"
    return "node_format"

# ─── BUILD LANGGRAPH ──────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("node_mcp_query",  node_mcp_query)
    g.add_node("node_format",     node_format)
    g.add_node("node_rag_search", node_rag_search)
    g.add_node("node_assemble",   node_assemble)
    g.set_entry_point("node_mcp_query")
    g.add_conditional_edges("node_mcp_query", route, {
        "node_format":     "node_format",
        "node_rag_search": "node_rag_search",
    })
    g.add_edge("node_format",     "node_assemble")
    g.add_edge("node_rag_search", "node_assemble")
    g.add_edge("node_assemble",   END)
    return g.compile()

agent = build_agent()

# ─── PUBLIC API ───────────────────────────────────────────────────────────────
def query_sap(question: str, verbose: bool = False) -> str:
    result = agent.invoke({
        "question":      question,
        "raw_data":      "",
        "used_pipeline": {},
        "tool_calls":    [],
        "intent":        "",
        "exec_status":   "",
        "tool_result":   "",
        "abap_query":    "",
        "final_answer":  "",
        "messages":      []
    })
    if verbose:
        print(f"  [Intent: {result['intent']} | Status: {result['exec_status']}]")
        print(f"  [Tools: {' -> '.join(result['tool_calls'])}]")
    return result["final_answer"]

# ─── TERMINAL ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  SAP SD Intelligent Agent  —  v6 (MCP + LangGraph)")
    print(f"  Model 1 : Claude API ({CLAUDE_MODEL}) via MCP tools")
    print(f"  Model 2 : {model2_name} [local, formats data]")
    print(f"  Database: MongoDB Atlas — {DB_NAME}")
    print(f"  Tools   : {', '.join(t['name'] for t in MCP_TOOLS)}")
    print("  Privacy : Claude sees schema only — Model 2 sees data locally")
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