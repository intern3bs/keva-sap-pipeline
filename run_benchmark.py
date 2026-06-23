"""
SAP Pipeline Benchmark Runner
Runs all benchmark questions across tiers and saves to CSV.

Usage:
  python run_benchmark.py
"""
import os
import csv
import re
from pipeline_v5 import query_sap, MODEL_1, MODEL_2
from pipeline_v5 import USE_CLAUDE


questions = [
    # ── Original 6 benchmark questions ───────────────────────────────────────
    ("Benchmark", "Top 5 customers by total invoiced value."),
    ("Benchmark", "Top selling product by quantity"),
    ("Benchmark", "Which customers saw a more than 20% growth in their sales from FY 24 to FY 25?"),
    ("Benchmark", "Give me the 3 products with the least margins"),
    ("Benchmark", "Which company, product combination gets me the most revenue (top 10)"),
    ("Benchmark", "Which area managers have seen the slowest growth in their sales?"),

    # ── Tier 1 — Field name accuracy ─────────────────────────────────────────
    ("Tier1", "What is the total billing value for Sales Organization 1000?"),
    ("Tier1", "Which billing type appears most frequently in the data?"),
    ("Tier1", "What is the average net value per billing document?"),
    ("Tier1", "Find all billing documents where tax amount exceeds 10000."),

    # ── Tier 2 — Multi-step aggregation ──────────────────────────────────────
    ("Tier2", "Which material has the highest total invoiced quantity across all billing documents?"),
    ("Tier2", "What is the total revenue per distribution channel?"),
    ("Tier2", "Which sales office generates the most revenue?"),
    ("Tier2", "List the top 5 customers by number of billing documents."),

    # ── Tier 3 — Growth and comparison ───────────────────────────────────────
    ("Tier3", "Which customers had more than 5 billing documents in 2013 vs 2014?"),
    ("Tier3", "Compare total revenue between Sales Organization 1000 and 3000."),
    ("Tier3", "Which materials saw an increase in invoiced quantity from 2013 to 2014?"),

    # ── Tier 4 — Margin calculation ───────────────────────────────────────────
    ("Tier4", "Which material group has the worst average margin?"),
    ("Tier4", "Find the top 5 most profitable materials by absolute margin amount."),
    ("Tier4", "What is the overall average margin across all billing line items?"),
    ("Tier4", "Which billing documents have a margin below 20%?"),

    # ── Tier 5 — Cross-collection ─────────────────────────────────────────────
    ("Tier5", "For each customer, how many distinct materials have they ordered?"),
    ("Tier5", "What is the average net value per sales document type?"),

    # ── Tier 6 — Semantic fallback ────────────────────────────────────────────
    ("Tier6", "What payment terms are most common and what do they mean?"),
    ("Tier6", "Which customers are from India and what regions are they in?"),

    # ── Tier 7 — Should return no records ────────────────────────────────────
    ("Tier7", "Which area managers have the highest sales growth?"),
    ("Tier7", "Show me delivery status for all pending shipments."),
    ("Tier7", "What is the profit by sales representative?"),
]

model1_label = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6") if USE_CLAUDE else MODEL_1
model2_label = MODEL_2
model_label  = f"{model1_label}+{model2_label}".replace(":", "_")
output_file  = f"benchmark_v5_{model_label}.csv"

print(f"Pipeline : pipeline_v5")
print(f"Model    : {MODEL_1}")
print(f"Output   : {output_file}")
print(f"Total Qs : {len(questions)}\n")
print("=" * 65)

rows = []
for i, (tier, q) in enumerate(questions, 1):
    print(f"[{tier}] Q{i}: {q}")
    try:
        result = query_sap(q)
    except Exception as e:
        result = f"ERROR: {e}"

    # parse answer, mongodb query, abap query
    parts       = result.split("---")
    answer      = parts[0].strip() if parts else result
    mongo_query = ""
    abap_query  = ""

    if len(parts) > 1:
        rest = parts[1]
        if "MongoDB Query:" in rest and "ABAP Query:" in rest:
            mongo_part  = rest.split("ABAP Query:")[0].replace("MongoDB Query:", "").strip()
            abap_part   = rest.split("ABAP Query:")[1].strip()
            mongo_query = re.sub(r'```python\s*|```', '', mongo_part).strip()
            abap_query  = re.sub(r'```abap\s*|```',  '', abap_part).strip()
        elif "MongoDB Query:" in rest:
            mongo_part  = rest.replace("MongoDB Query:", "").strip()
            mongo_query = re.sub(r'```python\s*|```', '', mongo_part).strip()

    rows.append({
        "Tier":          tier,
        "Question":      q,
        "Model_1":       model1_label,
        "Model_2":       model2_label,
        "Response":      answer,
        "MongoDB_Query": mongo_query,
        "ABAP_Query":    abap_query,
    })
    print(f"  ✅ Done\n")

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Tier", "Question", "Model_1", "Model_2", "Response", "MongoDB_Query", "ABAP_Query"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 65)
print(f"Saved to {output_file}")
print(f"Total questions answered: {len(rows)}")