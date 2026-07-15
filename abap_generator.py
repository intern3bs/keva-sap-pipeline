"""
SAP ABAP Query Generator
=========================
Generates ABAP SELECT queries for any business question.
Use this to fill the ABAP Query column in the benchmark sheet.
Output is documentation only — not connected to any SAP system.

Usage: python abap_generator.py
"""
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─── CONFIG FROM .env ─────────────────────────────────────────────────────────
LLM_MODEL   = os.getenv("LLM_MODEL",  "qwen2.5:3b")
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=TEMPERATURE
)

ABAP_PROMPT = PromptTemplate(
    template="""You are a senior SAP ABAP developer. Generate proper production-quality ABAP code.

Available SAP SD Tables and key fields:
- VBAK  : Sales Order Header    (VBELN, ERDAT, KUNNR, NAME1, NETWR, MWSBP, GBSTK, FISCAL_YEAR, AREA_MGR, REGION)
- VBAP  : Sales Order Items     (VBELN, POSNR, MATNR, ARKTX, KWMENG, MEINS, NETPR, NETWR, COST_PRICE, COST_TOTAL, MARGIN_PCT)
- VBRK  : Billing Doc Header    (VBELN, ERDAT, KUNNR, NAME1, NETWR, MWSBP, FKSTO, RFBSK, FISCAL_YEAR, AREA_MGR)
- VBRP  : Billing Doc Items     (VBELN, POSNR, MATNR, ARKTX, FKIMG, VRKME, NETWR, NETPR, COST_PRICE, COST_TOTAL, MARGIN_PCT)
- KNA1  : Customer Master       (KUNNR, NAME1, ORT01, LAND1, AREA_MGR, AREA_MGR_ID, REGION, TERRITORY)
- LIKP  : Delivery Header       (VBELN, ERDAT, KUNNR, NAME1, WADAT, WADAT_IST, LFGSX, VBELN_VK)
- LIPS  : Delivery Items        (VBELN, POSNR, MATNR, ARKTX, LFIMG, VRKME, VBELN_VK)

ABAP Coding Rules:
- Start with REPORT z_sap_query.
- Declare types with TYPES: BEGIN OF ty_result ... END OF ty_result.
- Use SELECT ... INTO TABLE @DATA(lt_result)
- Use GROUP BY for aggregations with SUM(), AVG(), COUNT()
- Use ORDER BY ... DESCENDING for top N queries
- Use UP TO N ROWS for limiting results
- Use WHERE clause for filters
- Add * comments explaining each section
- Status codes: GBSTK A=Open B=In-Process C=Completed
- Accounting status: RFBSK A=Open B=Partial C=Cleared

Question: {question}

ABAP Query:""",
    input_variables=["question"]
)

def generate_abap(question: str) -> str:
    return (ABAP_PROMPT | llm | StrOutputParser()).invoke({"question": question})

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"  SAP ABAP Query Generator")
    print(f"  Model: {LLM_MODEL}")
    print("  Generates ABAP code for benchmark sheet documentation")
    print("="*60)
    print("Type your question to get ABAP code.")
    print("Type 'quit' to exit\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        print("\n--- Generated ABAP Query ---")
        print(generate_abap(question))
        print("="*60 + "\n")