"""
SAP Pipeline Benchmark Runner — with checkpoint saving
Saves progress every 10 questions so Colab disconnection doesn't lose work.

Usage:
  python run_benchmark.py
  python run_benchmark.py --resume   # resume from checkpoint
"""
import os
import csv
import re
import json
import sys
import time

from pipeline_v6 import query_sap, CLAUDE_MODEL, model2_name, USE_CLAUDE, DB_NAME
MODEL_1 = CLAUDE_MODEL
MODEL_2 = model2_name

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

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHECKPOINT_EVERY = 10   # save every N questions
model1_label  = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6") if USE_CLAUDE else MODEL_1
model2_label  = MODEL_2
model_label   = f"{model1_label}+{model2_label}".replace(":", "_").replace("/", "-").replace(" ", "_")
output_file   = f"benchmark_v6_{model_label}.csv"
checkpoint_file = f"benchmark_v6_{model_label}_checkpoint.json"

print(f"Pipeline  : pipeline_v6")
print(f"Model 1   : {model1_label}")
print(f"Model 2   : {model2_label}")
print(f"Output    : {output_file}")
print(f"Checkpoint: {checkpoint_file}")
print(f"Total Qs  : {len(questions)}")
print("=" * 65)

# ─── LOAD CHECKPOINT if resuming ──────────────────────────────────────────────
resume = "--resume" in sys.argv
rows = []
start_from = 0

if resume and os.path.exists(checkpoint_file):
    with open(checkpoint_file) as f:
        checkpoint = json.load(f)
    rows = checkpoint["rows"]
    start_from = checkpoint["completed"]
    print(f"Resuming from question {start_from + 1} ({len(rows)} already done)\n")
else:
    print(f"Starting fresh\n")

# ─── RUN QUESTIONS ────────────────────────────────────────────────────────────
for i, (tier, q) in enumerate(questions[start_from:], start=start_from + 1):
    print(f"[{tier}] Q{i}/{len(questions)}: {q[:70]}")
    t0 = time.time()

    try:
        result = query_sap(q)
    except Exception as e:
        result = f"ERROR: {e}"

    elapsed = time.time() - t0

    # Parse answer, mongodb query, abap query
    parts       = result.split("<<<SPLIT>>>")
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
        "Time_s":        round(elapsed, 1),
    })
    print(f"  ✅ Done in {elapsed:.1f}s\n")

    # ── Checkpoint save every N questions ─────────────────────────────────────
    if i % CHECKPOINT_EVERY == 0 or i == len(questions):
        with open(checkpoint_file, "w") as f:
            json.dump({"completed": i, "rows": rows}, f)
        print(f"  💾 Checkpoint saved ({i}/{len(questions)} done)\n")

# ─── SAVE FINAL CSV ───────────────────────────────────────────────────────────
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Tier", "Question", "Model_1", "Model_2",
                    "Response", "MongoDB_Query", "ABAP_Query", "Time_s"]
    )
    writer.writeheader()
    writer.writerows(rows)

# Clean up checkpoint
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print("  🗑️  Checkpoint cleaned up")

print("=" * 65)
print(f"✅ Saved to {output_file}")
print(f"   Total questions: {len(rows)}")
print(f"   Avg time/q: {sum(r['Time_s'] for r in rows)/len(rows):.1f}s")