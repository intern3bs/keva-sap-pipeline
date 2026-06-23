"""
SAP ERP RAG Pipeline — Prompts
================================
All prompts used in pipeline_v5.py are defined here.
Import in pipeline_v5.py via: from prompts import *

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

# ─── QUERY GENERATION PROMPT (Model 1) ───────────────────────────────────────
# Used in : node_generate_query
# Sees    : question + schema only — never sees actual SAP data
# Purpose : Generate MongoDB aggregation pipeline for business question

QUERY_GEN_PROMPT = """You are an expert SAP ERP database analyst with deep MongoDB knowledge.
The SAP SD tables are available in the ERP system through MongoDB MCP.
You have direct access to query these collections.

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
    {{"$match": {{"Cost": {{"$gt": 0}}}}}},
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


# ─── ABAP QUERY PROMPT (Model 1) ──────────────────────────────────────────────
# Used in : node_generate_query (parallel call alongside QUERY_GEN_PROMPT)
# Purpose : Generate equivalent SAP ABAP query for documentation/reference

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


# ─── RETRY PROMPT (Model 1) ───────────────────────────────────────────────────
# Used in : node_execute (when query fails, retries up to 2x)
# Purpose : Fix failed MongoDB query using error message + schema

RETRY_PROMPT = """The MongoDB query failed.
Error: {error}

Query:
```python
{code}
```

Fix it. Use ONLY these field names (exact casing matters):
{schema}

Return ONLY corrected Python starting with `result = `:
```python
result = ..."""


# ─── SEMANTIC FORMAT PROMPT (Model 2) ─────────────────────────────────────────
# Used in : node_format — semantic path (RAG context answer)
# Model 2 sees actual SAP records here via RAG retrieval

SEMANTIC_FORMAT_PROMPT = """You are an SAP SD assistant. Answer using ONLY the records below.
Currency: INR | Fiscal Year: Indian Apr-Mar

SAP Records:
{context}

Question: {question}
Answer:"""


# ─── AGGREGATE FORMAT PROMPT (Model 2) ────────────────────────────────────────
# Used in : node_format — aggregate path (formats MongoDB query results)
# Model 2 sees actual query results here — formats for user presentation

AGGREGATE_FORMAT_PROMPT = """You are presenting SAP query results.
Present ONLY the rows below. Do not add, drop, round, rename or invent any value.
Show customer/material IDs exactly as they appear — do not replace with names.

Results:
{data}

Question: {question}
Answer:"""