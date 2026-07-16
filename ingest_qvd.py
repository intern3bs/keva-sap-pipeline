"""
QVD → MongoDB Atlas Ingestion Script
======================================
Reads all .qvd / .QVD files from a folder and loads them into MongoDB Atlas.
Each QVD file becomes a MongoDB collection (filename without extension).

Usage:
  python ingest_qvd.py                        # loads from ./qvd_files/
  python ingest_qvd.py --dir /path/to/qvds    # custom folder
  python ingest_qvd.py --file VBRK.qvd        # single file

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import os
import sys
import json
import argparse
import pandas as pd
import pyqvd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, date

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGO_URI")
DB_NAME     = os.getenv("DB_NAME", "sap_erp")
DEFAULT_DIR = "./qvd_files"

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def excel_serial_to_date(serial) -> str:
    """Convert Excel/Qlik serial date (float) to YYYY-MM-DD string."""
    try:
        if serial is None or serial != serial:  # NaN check
            return None
        serial = float(serial)
        if serial <= 0:
            return None
        # Excel epoch is 1899-12-30
        base = datetime(1899, 12, 30)
        delta_days = int(serial)
        result = base + __import__('datetime').timedelta(days=delta_days)
        return result.strftime("%Y-%m-%d")
    except Exception:
        return None

def is_date_column(col_name: str, sample_values) -> bool:
    """Heuristic: column name contains date keywords and values look like serials."""
    date_keywords = ['date', 'dat', '_dt', 'time', 'dob', 'birth', 'in', 'out']
    name_lower = col_name.lower()
    name_hint  = any(kw in name_lower for kw in date_keywords)

    # Check if values look like Excel serials (floats between 20000 and 60000 = ~1954 to ~2064)
    numeric_vals = [v for v in sample_values if v is not None and v == v]
    if not numeric_vals:
        return False
    try:
        nums = [float(v) for v in numeric_vals[:10]]
        looks_serial = all(20000 < n < 70000 for n in nums)
    except Exception:
        looks_serial = False

    return name_hint and looks_serial

def clean_record(record: dict, date_cols: set) -> dict:
    """Clean a single record — handle NaN, dates, types."""
    cleaned = {}
    for k, v in record.items():
        # NaN → None
        if v != v:  # NaN check
            cleaned[k] = None
        elif k in date_cols:
            cleaned[k] = excel_serial_to_date(v)
        elif isinstance(v, (pd.Timestamp, datetime, date)):
            cleaned[k] = str(v)[:10]
        elif isinstance(v, float) and v == int(v):
            cleaned[k] = int(v)  # 1.0 → 1
        else:
            cleaned[k] = v
    return cleaned

def ingest_file(filepath: str, db, verbose: bool = True) -> dict:
    """
    Ingest a single QVD file into MongoDB.
    Returns stats dict.
    """
    fname           = os.path.basename(filepath)
    collection_name = os.path.splitext(fname)[0]  # VBRK.qvd → VBRK

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  File       : {fname}")
        print(f"  Collection : {collection_name}")
        print(f"  Size       : {os.path.getsize(filepath)/1024:.1f} KB")

    # ── Read QVD ──────────────────────────────────────────────────────────────
    try:
        table = pyqvd.QvdTable.from_qvd(filepath)
        df    = table.to_pandas()
    except Exception as e:
        print(f"  ❌ Failed to read QVD: {e}")
        return {"file": fname, "status": "error", "error": str(e)}

    if verbose:
        print(f"  Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns    : {list(df.columns)}")

    if df.empty:
        print(f"  ⚠️  Empty file — skipped")
        return {"file": fname, "status": "skipped", "rows": 0}

    # ── Detect date columns ────────────────────────────────────────────────────
    date_cols = set()
    for col in df.columns:
        sample = df[col].dropna().head(10).tolist()
        if is_date_column(col, sample):
            date_cols.add(col)

    if date_cols and verbose:
        print(f"  Date cols  : {sorted(date_cols)}")

    # ── Clean records ──────────────────────────────────────────────────────────
    records = []
    for _, row in df.iterrows():
        records.append(clean_record(row.to_dict(), date_cols))

    # ── Insert into MongoDB ────────────────────────────────────────────────────
    # Drop existing collection and reload
    db[collection_name].drop()
    db[collection_name].insert_many(records)

    # Rebuild text index for BM25 search
    try:
        text_fields = [
            (col, "text") for col in df.columns
            if df[col].dtype == object and col not in date_cols
        ]
        if text_fields:
            db[collection_name].create_index(text_fields[:10])  # max 10 text fields
    except Exception:
        pass  # text index is optional

    if verbose:
        print(f"  ✅ Inserted {len(records)} records into '{collection_name}'")

    return {
        "file":       fname,
        "collection": collection_name,
        "status":     "success",
        "rows":       len(records),
        "columns":    list(df.columns),
        "date_cols":  sorted(date_cols),
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest QVD files into MongoDB Atlas")
    parser.add_argument("--dir",  default=DEFAULT_DIR, help="Folder containing QVD files")
    parser.add_argument("--file", default=None,        help="Single QVD file to ingest")
    parser.add_argument("--dry-run", action="store_true", help="Read QVDs but don't write to MongoDB")
    args = parser.parse_args()

    # Connect to MongoDB
    if not args.dry_run:
        if not MONGO_URI:
            print("❌ MONGO_URI not set in .env")
            sys.exit(1)
        client = MongoClient(MONGO_URI)
        db     = client[DB_NAME]
        print(f"✅ Connected to MongoDB Atlas — {DB_NAME}")
    else:
        db = None
        print("⚠️  Dry run mode — no writes to MongoDB")

    # Collect files to process
    if args.file:
        files = [args.file]
    else:
        if not os.path.isdir(args.dir):
            print(f"❌ Directory not found: {args.dir}")
            print(f"   Create it and put your QVD files there:")
            print(f"   mkdir -p {args.dir}")
            sys.exit(1)
        files = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if f.lower().endswith(".qvd")
        ]
        if not files:
            print(f"❌ No QVD files found in {args.dir}")
            sys.exit(1)

    print(f"\nFound {len(files)} QVD file(s) to ingest")
    print("=" * 60)

    # Process each file
    results = []
    for filepath in files:
        if args.dry_run:
            # Just read and report
            try:
                table = pyqvd.QvdTable.from_qvd(filepath)
                df    = table.to_pandas()
                print(f"  ✅ {os.path.basename(filepath)}: {df.shape[0]} rows × {df.shape[1]} cols")
                print(f"     Columns: {list(df.columns)}")
                results.append({"file": filepath, "status": "dry-run", "rows": len(df)})
            except Exception as e:
                print(f"  ❌ {os.path.basename(filepath)}: {e}")
        else:
            result = ingest_file(filepath, db)
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success = [r for r in results if r.get("status") == "success"]
    errors  = [r for r in results if r.get("status") == "error"]
    skipped = [r for r in results if r.get("status") == "skipped"]

    print(f"  ✅ Success : {len(success)}")
    print(f"  ❌ Errors  : {len(errors)}")
    print(f"  ⚠️  Skipped : {len(skipped)}")
    print(f"  Total rows: {sum(r.get('rows', 0) for r in success)}")

    if success:
        print(f"\nCollections created:")
        for r in success:
            print(f"  {r['collection']:<20} {r['rows']:>6} rows  {len(r['columns'])} cols")

    if errors:
        print(f"\nErrors:")
        for r in errors:
            print(f"  {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()