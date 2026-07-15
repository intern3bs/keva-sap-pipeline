"""
SAP SD Hybrid RAG Pipeline v2 - Final
Generates: ABAP Query + MongoDB Answer + LLM Response
"""
import os
import re
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME")

# ─── INIT ─────────────────────────────────────────────────────────────────────
client = MongoClient(MONGO_URI)
db     = client[DB_NAME]

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.1
)

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="sap_sd"
)

# ─── 1. ABAP QUERY GENERATOR ──────────────────────────────────────────────────
def generate_abap_query(question: str) -> str:
    prompt = PromptTemplate(
        template="""You are an SAP ABAP expert. Generate a proper ABAP SELECT query to answer the business question below.

SAP SD Tables:
- VBAK: Sales Order Header (VBELN, ERDAT, KUNNR, NAME1, NETWR, GBSTK, FISCAL_YEAR, AREA_MGR)
- VBAP: Sales Order Items (VBELN, POSNR, MATNR, ARKTX, KWMENG, MEINS, NETPR, NETWR, COST_PRICE, MARGIN_PCT)
- VBRK: Billing Header (VBELN, ERDAT, KUNNR, NAME1, NETWR, FKSTO, RFBSK, FISCAL_YEAR)
- VBRP: Billing Items (VBELN, MATNR, ARKTX, FKIMG, NETWR, NETPR, COST_PRICE, MARGIN_PCT)
- KNA1: Customer Master (KUNNR, NAME1, ORT01, AREA_MGR, REGION)
- LIKP: Delivery Header (VBELN, KUNNR, NAME1, WADAT, LFGSX)

Rules:
- Use proper ABAP SELECT syntax
- Use INTO TABLE @DATA(lt_result)
- Use GROUP BY for aggregations
- Use ORDER BY for sorting
- Use UP TO N ROWS for limits
- Add brief comments

Question: {question}

ABAP Query:""",
        input_variables=["question"]
    )
    return (prompt | llm | StrOutputParser()).invoke({"question": question})

# ─── 2. MONGODB DIRECT QUERIES ────────────────────────────────────────────────
def parse_amount(text):
    lakh_match  = re.search(r'([\d.]+)\s*lakh', text)
    crore_match = re.search(r'([\d.]+)\s*crore', text)
    num_match   = re.search(r'(?:rs\.?|inr|₹)?\s*([\d,]+)', text)
    if lakh_match:  return float(lakh_match.group(1)) * 100000
    if crore_match: return float(crore_match.group(1)) * 10000000
    if num_match:   return float(num_match.group(1).replace(",", ""))
    return None

def direct_query(question: str) -> str:
    q = question.lower()

    # Q1: Top N customers by invoiced value
    if ("top" in q or "highest" in q) and "customer" in q and ("invoiced" in q or "value" in q or "revenue" in q):
        n = int(re.search(r'top\s+(\d+)', q).group(1)) if re.search(r'top\s+(\d+)', q) else 5
        result = list(db["VBRK"].aggregate([
            {"$match": {"FKSTO": {"$ne": "X"}}},
            {"$group": {"_id": "$KUNNR", "NAME1": {"$first": "$NAME1"}, "total": {"$sum": "$NETWR"}}},
            {"$sort": {"total": -1}}, {"$limit": n}
        ]))
        lines = [f"Top {n} customers by total invoiced value:\n"]
        for i, r in enumerate(result, 1):
            lines.append(f"  {i}. {r['NAME1']:35s} ₹{r['total']:>15,.2f}")
        return "\n".join(lines)

    # Q2: Top selling product by quantity
    if ("top" in q or "best" in q) and ("product" in q or "material" in q) and ("quantity" in q or "qty" in q or "selling" in q):
        result = list(db["VBAP"].aggregate([
            {"$group": {"_id": "$MATNR", "ARKTX": {"$first": "$ARKTX"}, "qty": {"$sum": "$KWMENG"}, "revenue": {"$sum": "$NETWR"}}},
            {"$sort": {"qty": -1}}, {"$limit": 5}
        ]))
        lines = ["Top selling products by quantity:\n"]
        for i, r in enumerate(result, 1):
            lines.append(f"  {i}. {r['ARKTX']:40s} Qty: {r['qty']:>6} | ₹{r['revenue']:>15,.2f}")
        return "\n".join(lines)

    # Q3: Customer growth between fiscal years
    if "growth" in q and ("customer" in q or "fy" in q or "fiscal" in q or "percent" in q or "%" in q):
        fy_matches = re.findall(r'fy\s*(\d{4}[-–]\d{2,4})', q)
        fy1 = f"FY{fy_matches[0]}" if len(fy_matches) > 0 else "FY2023-24"
        fy2 = f"FY{fy_matches[1]}" if len(fy_matches) > 1 else "FY2024-25"
        threshold = float(re.search(r'(\d+)\s*(?:percent|%)', q).group(1)) if re.search(r'(\d+)\s*(?:percent|%)', q) else 20
        fy1_data = {r["_id"]: r for r in db["VBAK"].aggregate([
            {"$match": {"FISCAL_YEAR": fy1}},
            {"$group": {"_id": "$KUNNR", "NAME1": {"$first": "$NAME1"}, "rev": {"$sum": "$NETWR"}}}
        ])}
        fy2_data = {r["_id"]: r for r in db["VBAK"].aggregate([
            {"$match": {"FISCAL_YEAR": fy2}},
            {"$group": {"_id": "$KUNNR", "NAME1": {"$first": "$NAME1"}, "rev": {"$sum": "$NETWR"}}}
        ])}
        growth = []
        for kunnr, r2 in fy2_data.items():
            if kunnr in fy1_data and fy1_data[kunnr]["rev"] > 0:
                r1  = fy1_data[kunnr]["rev"]
                pct = (r2["rev"] - r1) / r1 * 100
                if pct > threshold:
                    growth.append({"name": r2["NAME1"], "fy1": r1, "fy2": r2["rev"], "pct": pct})
        growth.sort(key=lambda x: x["pct"], reverse=True)
        lines = [f"Customers with >{threshold:.0f}% growth ({fy1} → {fy2}):\n"]
        for r in growth:
            lines.append(f"  • {r['name']:35s} {fy1}: ₹{r['fy1']:>12,.0f} → {fy2}: ₹{r['fy2']:>12,.0f} | Growth: {r['pct']:>6.1f}%")
        return "\n".join(lines)

    # Q4: Products with least margins
    if "margin" in q and ("least" in q or "lowest" in q or "worst" in q or "product" in q):
        n = int(re.search(r'(\d+)\s+product', q).group(1)) if re.search(r'(\d+)\s+product', q) else 3
        result = list(db["VBAP"].aggregate([
            {"$group": {"_id": "$MATNR", "ARKTX": {"$first": "$ARKTX"}, "avg_margin": {"$avg": "$MARGIN_PCT"}, "total_rev": {"$sum": "$NETWR"}}},
            {"$sort": {"avg_margin": 1}}, {"$limit": n}
        ]))
        lines = [f"{n} products with least margins:\n"]
        for i, r in enumerate(result, 1):
            lines.append(f"  {i}. {r['ARKTX']:40s} Avg Margin: {r['avg_margin']:>5.1f}% | Revenue: ₹{r['total_rev']:>12,.0f}")
        return "\n".join(lines)

    # Q5: Top company-product combinations
    if ("company" in q or "customer" in q) and ("product" in q or "material" in q) and ("revenue" in q or "combination" in q):
        n = int(re.search(r'top\s+(\d+)', q).group(1)) if re.search(r'top\s+(\d+)', q) else 10
        result = list(db["VBAP"].aggregate([
            {"$group": {"_id": {"kunnr": "$KUNNR", "matnr": "$MATNR"}, "NAME1": {"$first": "$NAME1"}, "ARKTX": {"$first": "$ARKTX"}, "revenue": {"$sum": "$NETWR"}}},
            {"$sort": {"revenue": -1}}, {"$limit": n}
        ]))
        lines = [f"Top {n} company-product combinations by revenue:\n"]
        for i, r in enumerate(result, 1):
            lines.append(f"  {i:2}. {r['NAME1']:30s} | {r['ARKTX']:35s} | ₹{r['revenue']:>12,.0f}")
        return "\n".join(lines)

    # Q6: Area manager growth
    if "area manager" in q or ("manager" in q and ("growth" in q or "slowest" in q or "sales" in q)):
        fy_matches = re.findall(r'fy\s*(\d{4}[-–]\d{2,4})', q)
        fy1 = f"FY{fy_matches[0]}" if len(fy_matches) > 0 else "FY2023-24"
        fy2 = f"FY{fy_matches[1]}" if len(fy_matches) > 1 else "FY2024-25"
        fy1_am = {r["_id"]: r["rev"] for r in db["VBAK"].aggregate([
            {"$match": {"FISCAL_YEAR": fy1}},
            {"$group": {"_id": "$AREA_MGR", "rev": {"$sum": "$NETWR"}}}
        ])}
        fy2_am = {r["_id"]: r["rev"] for r in db["VBAK"].aggregate([
            {"$match": {"FISCAL_YEAR": fy2}},
            {"$group": {"_id": "$AREA_MGR", "rev": {"$sum": "$NETWR"}}}
        ])}
        growth = []
        for am, rev2 in fy2_am.items():
            if am in fy1_am and fy1_am[am] > 0:
                rev1 = fy1_am[am]
                pct  = (rev2 - rev1) / rev1 * 100
                growth.append({"am": am, "fy1": rev1, "fy2": rev2, "pct": pct})
        growth.sort(key=lambda x: x["pct"])
        lines = [f"Area manager growth ({fy1} → {fy2}), slowest first:\n"]
        for r in growth:
            lines.append(f"  {r['am']:25s} {fy1}: ₹{r['fy1']:>12,.0f} → {fy2}: ₹{r['fy2']:>12,.0f} | {r['pct']:>+.1f}%")
        return "\n".join(lines)

    # existing queries
    if ("above" in q or "greater than" in q or "more than" in q) and "order" in q:
        amount = parse_amount(q)
        if amount:
            records = list(db["VBAK"].find({"NETWR": {"$gt": amount}, "GBSTK": "A"}, {"VBELN":1, "NAME1":1, "NETWR":1, "VDATU":1, "_id":0}).sort("NETWR", -1))
            lines = [f"Found {len(records)} open sales orders above ₹{amount:,.0f}:\n"]
            for r in records:
                lines.append(f"  • {r['VBELN']} | {r['NAME1']} | ₹{r['NETWR']:,.2f} | Delivery by: {r['VDATU']}")
            return "\n".join(lines)

    if "completed" in q and ("order" in q or "sales" in q):
        records = list(db["VBAK"].find({"GBSTK": "C"}, {"VBELN":1, "NAME1":1, "NETWR":1, "MWSBP":1, "ERDAT":1, "_id":0}))
        lines = [f"Found {len(records)} completed sales orders:\n"]
        for r in records:
            total = round(r['NETWR'] + r['MWSBP'], 2)
            lines.append(f"  • {r['VBELN']} | {r['NAME1']} | Net: ₹{r['NETWR']:,.2f} | Total with GST: ₹{total:,.2f} | Date: {r['ERDAT']}")
        return "\n".join(lines)

    if "open" in q and ("order" in q or "sales" in q):
        records = list(db["VBAK"].find({"GBSTK": "A"}, {"VBELN":1, "NAME1":1, "NETWR":1, "VDATU":1, "_id":0}))
        lines = [f"Found {len(records)} open sales orders:\n"]
        for r in records:
            lines.append(f"  • {r['VBELN']} | {r['NAME1']} | ₹{r['NETWR']:,.2f} | Delivery by: {r['VDATU']}")
        return "\n".join(lines)

    if ("pending" in q or "not started" in q) and "deliver" in q:
        records = list(db["LIKP"].find({"LFGSX": "A"}, {"VBELN":1, "NAME1":1, "KUNNR":1, "WADAT":1, "VBELN_VK":1, "_id":0}))
        lines = [f"Found {len(records)} pending deliveries:\n"]
        for r in records:
            lines.append(f"  • Delivery {r['VBELN']} | Customer: {r['NAME1']} | Ref Order: {r['VBELN_VK']} | Planned: {r['WADAT']}")
        return "\n".join(lines)

    if ("not cleared" in q or "not yet cleared" in q) and ("bill" in q or "invoice" in q):
        records = list(db["VBRK"].find({"RFBSK": "A", "FKSTO": {"$ne": "X"}}, {"VBELN":1, "NAME1":1, "NETWR":1, "MWSBP":1, "ERDAT":1, "_id":0}))
        lines = [f"Found {len(records)} billing documents not yet cleared:\n"]
        for r in records:
            total = round(r['NETWR'] + r['MWSBP'], 2)
            lines.append(f"  • {r['VBELN']} | {r['NAME1']} | ₹{total:,.2f} | Date: {r['ERDAT']}")
        return "\n".join(lines)

    if "how many" in q and "customer" in q:
        return f"There are {db['KNA1'].count_documents({})} customers in the SAP system."

    return None

# ─── 3. SEMANTIC RAG QUERY ────────────────────────────────────────────────────
def get_tables_for_question(question: str) -> list:
    q = question.lower()
    if any(w in q for w in ["sales order", "order"]):   return ["VBAK", "VBAP"]
    elif any(w in q for w in ["delivery", "shipment"]): return ["LIKP", "LIPS"]
    elif any(w in q for w in ["billing", "invoice"]):   return ["VBRK", "VBRP"]
    elif any(w in q for w in ["customer", "client"]):   return ["KNA1", "KNVV"]
    return ["VBAK", "LIKP", "VBRK", "KNA1"]

def semantic_query(question: str) -> str:
    tables   = get_tables_for_question(question)
    all_docs = []
    for table in tables:
        docs = vectordb.similarity_search(question, k=5, filter={"table": table})
        all_docs.extend(docs)
    context = "\n\n---\n\n".join(d.page_content for d in all_docs)
    prompt  = PromptTemplate(
        template="""You are an SAP SD assistant. Use ONLY the context below to answer.
Be specific with document numbers, amounts in INR, and dates.
Status codes: GBSTK A=Open B=In-Process C=Completed | RFBSK A=Open B=Partial C=Cleared

Context:
{context}

Question: {question}
Answer:""",
        input_variables=["context", "question"]
    )
    return (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

# ─── 4. MAIN QUERY FUNCTION ───────────────────────────────────────────────────
def classify_question(question: str) -> str:
    q = question.lower()
    direct_patterns = [
        "top 5", "top 10", "top selling", "least margin", "slowest growth",
        "most revenue", "invoiced value", "list all", "show all",
        "all completed", "all open", "all pending", "how many", "count",
        "which customers", "which area manager", "not yet cleared",
        "not cleared", "not billed", "above", "below", "greater than",
        "less than", "more than", "growth", "percent", "%",
        "fy20", "fiscal", "margin", "quantity", "revenue", "invoiced",
    ]
    if any(p in q for p in direct_patterns):
        return "direct"
    return "semantic"

def query_sap(question: str):
    print(f"\n{'='*60}")
    print(f"Question: {question}")

    # Step 1: Generate ABAP query
    print("\n--- ABAP Query (generated) ---")
    abap = generate_abap_query(question)
    print(abap)

    # Step 2: Get MongoDB answer
    print("\n--- MongoDB Answer ---")
    mode = classify_question(question)
    print(f"Mode: {mode.upper()}")
    if mode == "direct":
        mongo_result = direct_query(question)
        if mongo_result:
            print(mongo_result)
        else:
            mongo_result = None
    else:
        mongo_result = None

    # Step 3: LLM response
    print("\n--- LLM Response ---")
    if mongo_result:
        llm_prompt = PromptTemplate(
            template="""You are an SAP SD assistant. Based on the data below, give a clear professional answer.

Data:
{data}

Question: {question}
Answer:""",
            input_variables=["data", "question"]
        )
        llm_response = (llm_prompt | llm | StrOutputParser()).invoke({
            "data": mongo_result,
            "question": question
        })
    else:
        llm_response = semantic_query(question)
    print(llm_response)

    return {
        "question":     question,
        "abap_query":   abap,
        "mongo_result": mongo_result,
        "llm_response": llm_response
    }

# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "Who are the top 5 customers by total invoiced value?",
        "What is the top selling product by quantity?",
        "Which customers saw more than 20 percent growth from FY2023-24 to FY2024-25?",
        "Which 3 products have the least margins?",
        "Which company and product combination gets the most revenue top 10?",
        "Which area managers have seen the slowest growth in sales?",
    ]
    for q in questions:
        query_sap(q)