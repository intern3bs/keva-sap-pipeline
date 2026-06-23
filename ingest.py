"""
SAP SD Embedding Pipeline
==========================
Reads all collections from MongoDB → converts to text → embeds → stores in ChromaDB.
Fully dynamic — no hardcoded table names.

Usage: python ingest.py
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

MONGO_URI   = os.getenv("MONGO_URI")
DB_NAME     = os.getenv("DB_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
print(f"  Embed model: {EMBED_MODEL}")

def record_to_text(table: str, record: dict) -> str:
    lines = [f"SAP Table: {table}"]
    for key, val in record.items():
        if key.startswith("_") or val in [None, "", 0, "0"]:
            continue
        lines.append(f"{key}: {val}")
    return "\n".join(lines)

def ingest():
    client   = MongoClient(MONGO_URI)
    db       = client[DB_NAME]
    all_docs = []

    # read all collections — skip system/internal ones starting with _
    tables = [c for c in db.list_collection_names() if not c.startswith("_")]

    for table in sorted(tables):
        records = list(db[table].find({}, {"_id": 0}))
        if not records:
            print(f"  {table:6s} → SKIPPED")
            continue
        print(f"  {table:6s} → {len(records)} records", end=" ... ")
        for r in records:
            all_docs.append(Document(
                page_content=record_to_text(table, r),
                metadata={"table": table, "source": f"SAP_{table}"}
            ))
        print("converted")

    print(f"\nTotal: {len(all_docs)} documents")
    print("Embedding in batches of 50...")

    vectordb      = None
    BATCH         = 50
    total_batches = (len(all_docs) + BATCH - 1) // BATCH

    for i in range(0, len(all_docs), BATCH):
        batch = all_docs[i:i + BATCH]
        bn    = i // BATCH + 1
        print(f"  Batch {bn}/{total_batches}...", end=" ", flush=True)
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory="./chroma_db",
                collection_name="sap_sd"
            )
        else:
            vectordb.add_documents(batch)
        print("done")

    print(f"\nDone. {len(all_docs)} documents embedded.")
    client.close()

if __name__ == "__main__":
    ingest()