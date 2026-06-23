"""
SAP Real Data Loader
====================
Scans SAP folder recursively for EXPORT.XLSX files.
Uses parent folder name as MongoDB collection name.
Loads data as-is — no column mapping.

Usage: python load_real_sap.py --path ./SAP
"""
import os
import argparse
import openpyxl
from datetime import datetime, date, time
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db     = client[os.getenv("DB_NAME")]

def clean(val):
    """Convert Excel cell values to MongoDB-safe types"""
    if val is None:               return None
    if isinstance(val, datetime): return val.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(val, date):     return val.strftime("%Y-%m-%d")
    if isinstance(val, time):     return val.strftime("%H:%M:%S")
    if isinstance(val, float):    return int(val) if val == int(val) else val
    return val

def load(sap_path: str):
    print(f"Scanning: {sap_path}\n")
    schema = {}

    # walk all subfolders, find EXPORT.XLSX files
    # use the immediate parent folder name as collection name
    # e.g. SAP/Sales Order Tables/VBAK/EXPORT.XLSX → collection: VBAK
    for root, dirs, files in os.walk(sap_path):
        for filename in files:
            if filename.upper() != "EXPORT.XLSX":
                continue

            table    = os.path.basename(root)   # folder name = SAP table name
            filepath = os.path.join(root, filename)

            wb   = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
            rows = list(wb.active.iter_rows(values_only=True))
            wb.close()

            if len(rows) < 2:
                print(f"  {table:6s} → EMPTY")
                continue

            headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(rows[0])]
            records = [
                {headers[i]: clean(v) for i, v in enumerate(row) if v is not None}
                for row in rows[1:]
                if any(v is not None for v in row)
            ]

            if not records:
                print(f"  {table:6s} → NO DATA")
                continue

            db[table].drop()
            db[table].insert_many(records)
            schema[table] = headers
            print(f"  {table:6s} → {len(records):5d} records, {len(headers)} columns")

    # save schema for LLM to read at query time
    db["_schema"].drop()
    db["_schema"].insert_one({"type": "schema_registry", "tables": schema})
    print(f"\nDone. {len(schema)} tables loaded.")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./SAP")
    load(parser.parse_args().path)