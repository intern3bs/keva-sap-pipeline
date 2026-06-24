"""
SAP SD Intelligent Agent Pipeline v6 — MCP Architecture
=========================================================
Claude calls MCP tools with JSON params. No Python code generation.
Llama (Model 2) formats results — Claude never sees actual SAP data.

Flow:
  Question
      |
  Claude (Model 1) — sees schema only, never sees data
      | calls get_sap_schema (JSON)    -> mcp_server.execute_tool() -> MongoDB Atlas
      | calls query_sap_collection (JSON) -> mcp_server.execute_tool() -> MongoDB Atlas
      |
  Raw data returned to pipeline (NOT to Claude)
      |
  Llama (Model 2) — formats raw data for user
      |
  Final Answer

vs v5:
  v5: Claude generates Python string -> safe_exec -> MongoDB -> Llama formats
  v6: Claude calls JSON tool -> execute_tool() -> MongoDB -> Llama formats

Privacy:
  Claude (Model 1): sees schema + question only — never sees actual SAP data
  Llama (Model 2): sees actual data — runs locally, data never leaves machine

All prompts  : prompts.py
All MCP tools: mcp_server.py

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import os
import re
import json
import anthropic
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
USE_CLAUDE    = True  # v6 always uses Claude as Model 1

# ─── CONNECTIONS ──────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# Model 2 — always local, always formats data, data never leaves machine
USE_CLAUDE_M2 = os.getenv("USE_CLAUDE_MODEL2", "false").lower() == "true"
if USE_CLAUDE_M2:
    llm_2       = ChatAnthropic(
        model=os.getenv("CLAUDE_MODEL_2", "claude-haiku-4-5"),
        temperature=TEMPERATURE,
        api_key=ANTHROPIC_KEY
    )
    model2_name = f"Claude Haiku ({os.getenv('CLAUDE_MODEL_2','claude-haiku-4-5')})"
else:
    llm_2       = ChatOllama(model=MODEL_2, base_url=OLLAMA_URL, temperature=TEMPERATURE)
    model2_name = MODEL_2

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectordb   = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="sap_sd"
)

print(f"  Model 1 (query gen) : Claude API ({CLAUDE_MODEL}) via MCP tools")
print(f"  Model 2 (format)    : {model2_name} [local — sees data]")
print(f"  Database            : MongoDB Atlas — {DB_NAME}")
print(f"  MCP Tools           : {len(MCP_TOOLS)} tools from mcp_server.py")

# ─── SCHEMA SUMMARY — built from mcp_server.SCHEMA_CACHE ──────────────────────
def build_schema_summary() -> str:
    """Concise schema for Claude's system prompt — sourced from mcp_server."""
    lines = ["SAP MongoDB Collections on Atlas:\n"]
    for col, info in SCHEMA_CACHE.items():
        lines.append(f"Collection: {col} ({info['count']} docs)")
        lines.append(f"  Fields: {', '.join(info['fields'][:20])}")
        for k, r in info.get("date_ranges", {}).items():
            lines.append(f"  Date range [{k}]: {r['min']} -> {r['max']}")
        lines.append("")
    return "\n".join(lines)

# Build system prompt once at startup — schema injected from mcp_server
SYSTEM_PROMPT = MCP_SYSTEM_PROMPT.format(
    schema_summary=build_schema_summary()
)

# ─── MAIN QUERY FUNCTION ──────────────────────────────────────────────────────
def query_sap_v6(question: str, verbose: bool = False) -> str:
    """
    v6 entry point.
    Claude calls MCP tools to get raw data.
    Llama (Model 2) formats the raw data — Claude never sees actual data.
    """
    messages         = [{"role": "user", "content": question}]
    tool_calls_made  = []
    raw_data         = ""       # raw JSON from MongoDB — goes to Llama, not Claude
    used_pipeline    = None

    # ── Step 1: Claude calls MCP tools to get raw data ────────────────────────
    for iteration in range(8):

        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,        # lower — Claude only generates tool calls not answers
            system=SYSTEM_PROMPT,
            tools=MCP_TOOLS,        # tool specs from mcp_server.py
            messages=messages
        )

        if verbose:
            print(f"  [v6 iter {iteration+1}] stop_reason: {response.stop_reason}")

        assistant_content = []
        tool_results      = []
        has_tool_calls    = False

        for block in response.content:

            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                if verbose and block.text.strip():
                    print(f"  [Claude] {block.text[:150]}")

            elif block.type == "tool_use":
                has_tool_calls = True
                tool_name      = block.name
                tool_input     = block.input
                tool_calls_made.append(tool_name)

                if verbose:
                    print(f"  [Tool] {tool_name}")
                    print(f"  [JSON] {json.dumps(tool_input)[:200]}")

                # Execute via mcp_server.execute_tool() — no Python code generation
                tool_result = execute_tool(tool_name, tool_input)

                if verbose:
                    print(f"  [Data] {tool_result[:200]}...")

                # Capture raw data from query — this goes to Llama, NOT back to Claude
                if tool_name == "query_sap_collection":
                    raw_data      = tool_result
                    used_pipeline = tool_input

                assistant_content.append({
                    "type":  "tool_use",
                    "id":    block.id,
                    "name":  tool_name,
                    "input": tool_input
                })
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     tool_result  # schema/confirmation only sent back to Claude
                })

        messages.append({"role": "assistant", "content": assistant_content})

        if has_tool_calls:
            messages.append({"role": "user", "content": tool_results})
            continue

        # No more tool calls — Claude is done querying
        break

    # ── Step 2: Llama (Model 2) formats the raw data ──────────────────────────
    # Claude never formats — Llama always formats
    # This means Claude never sees the actual SAP data values

    if raw_data:
        # Aggregate path — Llama formats MongoDB results
        intent = "aggregate"
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

    else:
        # No data returned — RAG fallback via Llama
        intent = "semantic"
        answer = rag_fallback(question, verbose)

    # ── Step 3: Generate ABAP query via Llama (documentation only) ────────────
    abap_prompt = PromptTemplate(
        template=ABAP_PROMPT,
        input_variables=["question"]
    )
    abap_query = (abap_prompt | llm_2 | StrOutputParser()).invoke({"question": question})
    abap_query = re.sub(r'<think>.*?</think>', '', abap_query, flags=re.DOTALL).strip()

    # ── Step 4: Build MongoDB pipeline display ─────────────────────────────────
    if used_pipeline:
        pipeline_display = (
            f"db[\"{used_pipeline['collection']}\"].aggregate(\n"
            f"{json.dumps(used_pipeline['pipeline'], indent=2)}\n)"
        )
    else:
        pipeline_display = "No aggregation — semantic answer via RAG"

    # ── Final answer ───────────────────────────────────────────────────────────
    tools_used = " -> ".join(tool_calls_made) if tool_calls_made else "RAG search"
    return (
        f"{answer}\n\n"
        f"---\n"
        f"MCP Tools: {tools_used}\n\n"
        f"MongoDB Query:\n```python\n{pipeline_display}\n```\n\n"
        f"ABAP Query:\n{abap_query}"
    )


def rag_fallback(question: str, verbose: bool = False) -> str:
    """RAG search for semantic questions where no MQL query applies."""
    if verbose:
        print("  [Fallback] Using RAG search")
    collections = [c for c in db.list_collection_names() if not c.startswith("_")]
    vec = []
    for t in collections:
        for doc in vectordb.similarity_search(question, k=3, filter={"table": t}):
            vec.append(doc.page_content)
    context = "\n\n---\n\n".join(vec[:8])
    prompt  = PromptTemplate(
        template=SEMANTIC_FORMAT_PROMPT,
        input_variables=["context", "question"]
    )
    answer = (prompt | llm_2 | StrOutputParser()).invoke({
        "context":  context,
        "question": question
    })
    return re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip() or context


# Backward compatible alias — run_benchmark.py works with both pipelines
def query_sap(question: str, verbose: bool = False) -> str:
    return query_sap_v6(question, verbose)


# ─── TERMINAL ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  SAP SD Intelligent Agent  —  v6 (MCP Architecture)")
    print(f"  Model 1 : Claude API ({CLAUDE_MODEL}) via MCP tools")
    print(f"  Model 2 : {model2_name} [local, formats data]")
    print(f"  Database: MongoDB Atlas — {DB_NAME}")
    print(f"  Tools   : {', '.join(t['name'] for t in MCP_TOOLS)}")
    print("  Privacy : Claude sees schema only — Llama sees data locally")
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