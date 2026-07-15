"""
SAP SD RAG Query Pipeline - v2
Smart table routing + better prompting
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─── INIT ─────────────────────────────────────────────────────────────────────
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
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

# ─── SMART ROUTING ────────────────────────────────────────────────────────────
# Route question to the right SAP tables based on keywords
def get_relevant_tables(question: str) -> list:
    q = question.lower()

    if any(w in q for w in ["sales order", "order status", "order value", "vbak"]):
        return ["VBAK", "VBAP"]
    elif any(w in q for w in ["delivery", "shipment", "shipped", "dispatch"]):
        return ["LIKP", "LIPS"]
    elif any(w in q for w in ["billing", "invoice", "bill", "cleared", "payment"]):
        return ["VBRK", "VBRP"]
    elif any(w in q for w in ["customer", "client", "buyer"]):
        return ["KNA1", "KNVV"]
    else:
        return ["VBAK", "LIKP", "VBRK", "KNA1"]  # broad search

def smart_retrieve(question: str, k: int = 10) -> list:
    tables = get_relevant_tables(question)
    print(f"  Searching tables: {tables}")

    all_docs = []
    per_table = max(3, k // len(tables))

    for table in tables:
        docs = vectordb.similarity_search(
            question,
            k=per_table,
            filter={"table": table}
        )
        all_docs.extend(docs)

    return all_docs[:k]

# ─── PROMPT ───────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an SAP SD data assistant with access to real SAP records.

SAP Tables available:
- VBAK: Sales Order Headers (order number, customer, status, total value)
- VBAP: Sales Order Items (material, quantity, price per order)
- LIKP: Delivery Headers (delivery number, shipping date, status)
- LIPS: Delivery Items (material, delivered quantity)
- VBRK: Billing Documents (invoice number, amount, accounting status)
- KNA1: Customer Master (customer name, city, GST number)

Rules:
- Answer ONLY from the context provided below
- Be specific — include document numbers, amounts in INR, dates
- If listing multiple records, format them clearly
- Overall Status: A=Open, B=In Process, C=Completed
- Accounting Status: A=Open, B=Partial, C=Cleared

Context from SAP records:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ─── QUERY FUNCTION ───────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

def query_sap(question: str) -> dict:
    print(f"\nQuestion: {question}")
    print("Searching SAP records...")

    source_docs = smart_retrieve(question)
    context     = format_docs(source_docs)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })

    print(f"\nAnswer:\n{answer}")
    print(f"\nSources: {len(source_docs)} SAP records from {list(set(d.metadata['table'] for d in source_docs))}")
    print("="*60)

    return {"question": question, "answer": answer, "sources": source_docs}

# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "List all completed sales orders",
        "Which customers have pending deliveries?",
        "Show me billing documents that are not yet cleared",
        "What is the total value of sales order SO00000086?",
    ]

    for q in questions:
        query_sap(q)