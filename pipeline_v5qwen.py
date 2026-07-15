"""
SAP SD Intelligent Agent Pipeline v5 — Production
====================================================
Text-to-MQL architecture based on MongoDB + LangChain best practices.
Reference: https://www.mongodb.com/blog/post/technical/natural-language-agents-mongodb-text-mql-langchain

Two-model architecture:
  Model 1: question + schema → MongoDB query + ABAP query
           (sees NO data — only schema — can be swapped to Claude/GPT)
  Model 2: executes query + formats answer
           (sees data — always local — data never leaves machine)
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
    llm_1 = ChatAnthropic(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        temperature=TEMPERATURE,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    print(f"  Model 1 (query gen) : Claude API ({os.getenv('CLAUDE_MODEL','claude-sonnet-4-6')})")
else:
    llm_1 = ChatOllama(model=MODEL_1, base_url=OLLAMA_URL, temperature=TEMPERATURE)
llm_2 = ChatOllama(model=MODEL_2, base_url=OLLAMA_URL, temperature=TEMPERATURE)
embeddings   = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectordb     = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="sap_sd"
)

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
        sample = db[table].find_one({}, {"_id": 0}) or {}
        non_empty = {k: v for k, v in sample.items() if v not in [None, "", 0, "0"]}

        # compute actual date ranges so model knows what data exists
        date_ranges = {}
        for k, v in non_empty.items():
            if isinstance(v, str) and DATE_RE.match(v):
                q   = {k: {"$nin": [None, ""]}}
                mn  = db[table].find_one(q, {k: 1, "_id": 0}, sort=[(k, 1)])
                mx  = db[table].find_one(q, {k: 1, "_id": 0}, sort=[(k, -1)])
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
    question:    str
    mongo_code:  str
    abap_query:  str
    tool_result: str
    exec_status: str   # 'data' | 'empty' | 'error' | 'blocked'
    intent:      str
    final_answer: str
    messages:    Annotated[list, add_messages]

# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — QUERY GENERATOR (Model 1)
# Sees: question + schema only — never sees actual data
# ══════════════════════════════════════════════════════════════════════════════
QUERY_GEN_PROMPT = """You are an expert SAP ERP database analyst with deep MongoDB knowledge.

DATABASE SCHEMA (use ONLY these exact field names and collection names):
{schema}

TASK: Generate a MongoDB Python query for the business question below.

RULES:
1. Use ONLY field names from schema above. Copy casing exactly.
2. $match values must match the type shown in Sample — string fields need quoted values e.g. "1000" not 1000.
3. ONLY add date/$match filters if the question explicitly mentions a year, period, or fiscal year.
   "Top selling product", "highest margin", "most revenue" have NO date filter.
   The data covers ONLY the date ranges shown in schema — do not filter outside those ranges.
4. Indian Fiscal Year (Apr 1 – Mar 31):
   FY 2022-23 → "Created On" >= "2022-04-01 00:00:00" AND < "2023-04-01 00:00:00"
5. For growth across periods: TWO separate aggregations + Python math. Never one pipeline.
6. COLLECTION ROUTING — follow strictly:
   - Customer queries (top customers, customer growth, billing by customer): use VBRK
     VBRK has: "Sold-To Party" (customer ID), "Net Value" (capital V), "Billing Type",
     "Sales Organization", "Distribution Channel", "Tax amount", "Created On"
   - Product/material queries (top products, margins, invoiced qty): use VBRP
     VBRP has: "Material", "Net value" (lowercase v), "Cost", "Invoiced Quantity",
     "Material Group", "Description", "Created On"
    - NEVER use VBRP for customer queries — it has no customer ID field
    - "Sales Office" field only exists in VBAK. For sales office queries use VBAK grouped by "Sales Office" with sum of "Net value"
    - To join VBRP with VBRK use "Billing Document" as join key (localField and foreignField both = "Billing Document")   
7. Margin formula (use ONLY in $project after $group, never in $group):
   margin_pct = (Net value - Cost) / Net value * 100
   In MQL: {{"$multiply": [{{"$divide": [{{"$subtract": ["$rev", "$cost"]}}, "$rev"]}}, 100]}}
   Always add {{"$match": {{"Cost": {{"$gt": 0}}}}}} before grouping to exclude items with no cost data.
8. Always assign the final answer to `result`. For multi-step queries, assign at the end — never initialize `result = None` at the top.
9. If needed field does not exist in schema → INTENT: semantic, leave MONGO_CODE empty.

EXAMPLES (correct patterns):
# Top customers by billing value → VBRK
result = list(db["VBRK"].aggregate([
    {{"$group": {{"_id": "$Sold-To Party", "total": {{"$sum": "$Net Value"}}}}}},
    {{"$sort": {{"total": -1}}}}, {{"$limit": 5}}]))

# Product margins → VBRP, margin in $project
result = list(db["VBRP"].aggregate([
    {{"$group": {{"_id": "$Material", "rev": {{"$sum": "$Net value"}}, "cost": {{"$sum": "$Cost"}}}}}},
    {{"$project": {{"margin_pct": {{"$multiply": [{{"$divide": [{{"$subtract": ["$rev","$cost"]}},"$rev"]}},100]}}}}}},
    {{"$sort": {{"margin_pct": 1}}}}, {{"$limit": 3}}]))

# Customer growth: TWO queries + Python
fy_old = {{r["_id"]: r["rev"] for r in db["VBRK"].aggregate([
    {{"$match": {{"Created On": {{"$gte": "2022-04-01 00:00:00", "$lt": "2023-04-01 00:00:00"}}}}}},
    {{"$group": {{"_id": "$Sold-To Party", "rev": {{"$sum": "$Net Value"}}}}}}])}}
fy_new = {{r["_id"]: r["rev"] for r in db["VBRK"].aggregate([
    {{"$match": {{"Created On": {{"$gte": "2023-04-01 00:00:00", "$lt": "2024-04-01 00:00:00"}}}}}},
    {{"$group": {{"_id": "$Sold-To Party", "rev": {{"$sum": "$Net Value"}}}}}}])}}
result = [{{"customer": k, "growth_pct": round((v-fy_old[k])/fy_old[k]*100,1)}}
          for k,v in fy_new.items() if k in fy_old and fy_old[k]>0
          and (v-fy_old[k])/fy_old[k]*100 > 20]
result.sort(key=lambda x: x["growth_pct"], reverse=True)

# Distinct materials per customer — join VBRP to VBRK on "Billing Document"
result = list(db["VBRP"].aggregate([
    {{"$lookup": {{"from": "VBRK", "localField": "Billing Document",
                  "foreignField": "Billing Document", "as": "header"}}}},
    {{"$unwind": "$header"}},
    {{"$group": {{"_id": "$header.Sold-To Party",
                 "distinct_materials": {{"$addToSet": "$Material"}}}}}},
    {{"$project": {{"_id": 1, "customer": "$_id",
                   "material_count": {{"$size": "$distinct_materials"}}}}}},
    {{"$sort": {{"material_count": -1}}}}]))

CLASSIFY intent:
- aggregate: needs computation, ranking, grouping, totals → generate MONGO_CODE
- semantic:  descriptive/explanatory or needed field missing → leave MONGO_CODE empty

Respond in EXACTLY this format:
INTENT: aggregate
MONGO_CODE:
```python
result = ...
```

Question: {question}"""

ABAP_PROMPT = """You are a senior SAP ABAP developer. Generate an ABAP SELECT query.

SAP SD Tables: VBAK (Sales Orders), VBAP (Order Items), VBRK (Billing Header),
VBRP (Billing Items), KNA1 (Customer Master), LIKP (Delivery Header),
LIPS (Delivery Items), VBFA (Document Flow), MARA (Material), MAKT (Material Desc)

Rules:
- REPORT z_sap_query.
- SELECT ... INTO TABLE @DATA(lt_result)
- GROUP BY for aggregations
- ORDER BY ... DESCENDING for rankings
- UP TO N ROWS for limits
- Add * comments
- Return ONLY the ABAP code, nothing else

Question: {question}
ABAP Query:"""

def node_generate_query(state: AgentState) -> AgentState:
    prompt = PromptTemplate(template=QUERY_GEN_PROMPT, input_variables=["schema", "question"])
    raw = (prompt | llm_1 | StrOutputParser()).invoke({
        "schema":   get_relevant_schema(state["question"]),
        "question": state["question"]
    })
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    intent_m   = re.search(r'INTENT:\s*(aggregate|semantic)', raw, re.I)
    intent     = intent_m.group(1).lower() if intent_m else "semantic"
    code_m     = re.search(r'MONGO_CODE:\s*```python\s*(.*?)```', raw, re.DOTALL)
    mongo_code = code_m.group(1).strip() if code_m else ""

    abap_prompt = PromptTemplate(template=ABAP_PROMPT, input_variables=["question"])
    abap_raw    = (abap_prompt | llm_1 | StrOutputParser()).invoke({"question": state["question"]})
    abap_query  = re.sub(r'<think>.*?</think>', '', abap_raw, flags=re.DOTALL).strip()

    return {**state, "intent": intent, "mongo_code": mongo_code,
            "abap_query": abap_query, "messages": [HumanMessage(content=state["question"])]}

# ══════════════════════════════════════════════════════════════════════════════
# NODE 2A — EXECUTE (Model 2 — always local)
# ══════════════════════════════════════════════════════════════════════════════
def safe_exec(code: str) -> tuple[str, str]:
    """Execute generated MongoDB code. Returns (text, status).
    status ∈ {'data', 'empty', 'error', 'blocked'}
    """
    if not code:
        return "", "error"

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

            # null-group guard: grouping by non-existent field → all _id None
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

    while status in ("error", "blocked") and code and attempts < 2:
        retry_prompt = PromptTemplate(
            template="""The MongoDB query failed.
Error: {error}

Query:
```python
{code}
```

Fix it. Use ONLY these field names (exact casing matters):
{schema}

Return ONLY corrected Python starting with `result = `:
```python
result = ...""",
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
# NODE 2B — RAG SEARCH (fallback for semantic intent only)
# ══════════════════════════════════════════════════════════════════════════════
def rrf(lists: list, k: int = 60) -> list:
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
    question    = state["question"]
    collections = [c for c in db.list_collection_names() if not c.startswith("_")]
    vec, txt    = [], []

    for t in collections:
        for doc in vectordb.similarity_search(question, k=3, filter={"table": t}):
            vec.append({"id": f"{t}_{hash(doc.page_content)}", "text": doc.page_content})

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
# NODE 3 — FORMAT (Model 2)
# ══════════════════════════════════════════════════════════════════════════════
def node_format(state: AgentState) -> AgentState:
    status     = state.get("exec_status", "")
    mongo_code = state.get("mongo_code", "")
    abap       = state.get("abap_query", "")

    # error/blocked — honest failure message, not "no data"
    if status in ("error", "blocked"):
        final = (
            f"⚠️ Could not execute this query (this does NOT mean there is no data):\n\n"
            f"{state['tool_result']}\n\n"
            f"---\nMongoDB Query:\n```python\n{mongo_code}\n```\n\nABAP Query:\n{abap}"
        )
        return {**state, "final_answer": final}

    # empty — honest no-data message
    if status == "empty":
        final = (
            f"No matching records were found for this question.\n\n"
            f"---\nMongoDB Query:\n```python\n{mongo_code}\n```\n\nABAP Query:\n{abap}"
        )
        return {**state, "final_answer": final}

    # semantic — LLM answers from RAG context
    if state["intent"] == "semantic":
        prompt = PromptTemplate(
            template="""You are an SAP SD assistant. Answer using ONLY the records below.
Currency: INR | Fiscal Year: Indian Apr-Mar

SAP Records:
{context}

Question: {question}
Answer:""",
            input_variables=["context", "question"]
        )
        answer = (prompt | llm_2 | StrOutputParser()).invoke({
            "context": state["tool_result"], "question": state["question"]
        })
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

    # aggregate — LLM formats with fabrication guard
    else:
        prompt = PromptTemplate(
            template="""You are presenting SAP query results.
Present ONLY the rows below. Do not add, drop, round, rename or invent any value.
Show customer/material IDs exactly as they appear — do not replace with names.

Results:
{data}

Question: {question}
Answer:""",
            input_variables=["data", "question"]
        )
        answer = (prompt | llm_2 | StrOutputParser()).invoke({
            "data": state["tool_result"], "question": state["question"]
        })
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

        # numeric fabrication guard — if LLM invented numbers not in source, use raw rows
        # fabrication guard — catch invented numbers not present in source
        # normalize floats to 2dp before comparing to handle qwen3.5 rounding differences
        def extract_nums(s):
            nums = set()
            for x in re.findall(r'-?\d[\d,]*\.?\d*', s):
                x = x.replace(",", "")
                try:
                    nums.add(round(float(x), 2))
                except ValueError:
                    nums.add(x)
            return nums

        answer_nums = extract_nums(answer)
        source_nums = extract_nums(state["tool_result"])
        invented    = answer_nums - source_nums

        if not answer.strip():
            answer = "Results:\n" + state["tool_result"]
        elif len(invented) > 2:
            # allow small differences (rounding, percentages) but catch clear fabrication
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
    result = agent.invoke({
        "question":    question,
        "mongo_code":  "",
        "abap_query":  "",
        "tool_result": "",
        "exec_status": "",
        "intent":      "",
        "final_answer": "",
        "messages":    []
    })
    if verbose:
        print(f"  [Intent: {result['intent']} | Status: {result['exec_status']}]")
        print(f"  [MongoDB:\n{result['mongo_code']}]")
    return result["final_answer"]

# ─── TERMINAL ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  SAP SD Intelligent Agent  —  v5")
    print(f"  Model 1 : {MODEL_1}")
    print(f"  Model 2 : {MODEL_2}")
    print(f"  Database: {DB_NAME}")
    print("  Text-to-MQL | LangGraph | Fully Offline")
    print("="*65)
    print("Type 'verbose' to toggle debug | 'quit' to exit\n")

    verbose = False
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not q: continue
        if q.lower() in ["quit", "exit", "q"]: print("Goodbye!"); break
        if q.lower() == "verbose":
            verbose = not verbose
            print(f"  [Verbose: {'ON' if verbose else 'OFF'}]\n")
            continue
        print(f"\nAssistant: {query_sap(q, verbose)}\n")
        print("-" * 65 + "\n")