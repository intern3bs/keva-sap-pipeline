"""
JSON → QVD Converter
=====================
Converts MongoDB JSON exports to QVD files for Qlik pipeline testing.

Usage:
  python json_to_qvd.py                         # converts all JSON in ./json_exports/
  python json_to_qvd.py --dir ./json_exports     # custom input folder
  python json_to_qvd.py --out ./qvd_files        # custom output folder

OR export directly from MongoDB and convert in one step:
  python json_to_qvd.py --from-mongo

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import os
import sys
import json
import argparse
import pandas as pd
import pyqvd

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def flatten_value(v):
    """Convert complex types to strings for QVD compatibility."""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v

def json_to_df(records: list) -> pd.DataFrame:
    """Convert list of MongoDB records to clean DataFrame."""
    if not records:
        return pd.DataFrame()

    # Flatten any nested values
    cleaned = []
    for rec in records:
        cleaned.append({k: flatten_value(v) for k, v in rec.items()})

    df = pd.DataFrame(cleaned)

    # Convert all columns to string for QVD compatibility
    # (QVD handles mixed types poorly)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str)
        else:
            # Keep numeric columns as-is
            pass

    return df

def convert_file(json_path: str, out_dir: str) -> dict:
    """Convert a single JSON file to QVD."""
    fname    = os.path.basename(json_path)
    name     = os.path.splitext(fname)[0]
    out_path = os.path.join(out_dir, f"{name}.qvd")

    print(f"\n{'─'*55}")
    print(f"  Input  : {fname}")
    print(f"  Output : {name}.qvd")

    try:
        # Load JSON
        with open(json_path) as f:
            records = json.load(f)

        if not records:
            print(f"  ⚠️  Empty — skipped")
            return {"name": name, "status": "skipped", "rows": 0}

        print(f"  Rows   : {len(records)}")
        print(f"  Cols   : {len(records[0])} — {list(records[0].keys())[:8]}...")

        # Convert to DataFrame
        df = json_to_df(records)

        # Convert to QVD
        table = pyqvd.QvdTable.from_pandas(df)
        table.to_qvd(out_path)

        size_kb = os.path.getsize(out_path) / 1024
        print(f"  ✅ Written: {name}.qvd ({size_kb:.1f} KB)")

        return {
            "name":    name,
            "status":  "success",
            "rows":    len(records),
            "cols":    len(records[0]),
            "out":     out_path,
            "size_kb": round(size_kb, 1)
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {"name": name, "status": "error", "error": str(e)}

def export_from_mongo(json_dir: str):
    """Export all MongoDB collections to JSON files."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from pymongo import MongoClient
        import os

        uri     = os.getenv("MONGO_URI")
        db_name = os.getenv("DB_NAME", "sap_erp")
        client  = MongoClient(uri)
        db      = client[db_name]

        cols = [c for c in db.list_collection_names() if not c.startswith("_")]
        print(f"✅ Connected to MongoDB — {db_name}")
        print(f"   Found {len(cols)} collections: {cols}\n")

        os.makedirs(json_dir, exist_ok=True)

        for col in cols:
            docs = list(db[col].find({}, {"_id": 0}))
            path = os.path.join(json_dir, f"{col}.json")
            with open(path, "w") as f:
                json.dump(docs, f, default=str)
            print(f"  ✅ {col}: {len(docs)} docs → {col}.json")

        print(f"\nExported {len(cols)} collections to {json_dir}/")
        return True

    except Exception as e:
        print(f"❌ MongoDB export failed: {e}")
        return False

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Convert JSON exports to QVD files")
    parser.add_argument("--dir",        default="./json_exports", help="Input folder with JSON files")
    parser.add_argument("--out",        default="./qvd_files",    help="Output folder for QVD files")
    parser.add_argument("--from-mongo", action="store_true",      help="Export from MongoDB first then convert")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Step 1 — Export from MongoDB if requested
    if args.from_mongo:
        print("Step 1 — Exporting MongoDB collections to JSON...")
        print("=" * 55)
        success = export_from_mongo(args.dir)
        if not success:
            sys.exit(1)
        print()

    # Step 2 — Find JSON files
    if not os.path.isdir(args.dir):
        print(f"❌ Directory not found: {args.dir}")
        print(f"   Run with --from-mongo to export from MongoDB first")
        sys.exit(1)

    json_files = [
        os.path.join(args.dir, f)
        for f in sorted(os.listdir(args.dir))
        if f.endswith(".json")
    ]

    if not json_files:
        print(f"❌ No JSON files found in {args.dir}")
        sys.exit(1)

    print(f"Step 2 — Converting {len(json_files)} JSON files to QVD")
    print("=" * 55)

    # Step 3 — Convert each file
    results = [convert_file(f, args.out) for f in json_files]

    # Summary
    success = [r for r in results if r["status"] == "success"]
    errors  = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"  ✅ Success : {len(success)}")
    print(f"  ❌ Errors  : {len(errors)}")
    print(f"  ⚠️  Skipped : {len(skipped)}")
    print(f"  Total rows: {sum(r.get('rows',0) for r in success):,}")
    print(f"  Output dir: {args.out}/")

    if success:
        print(f"\nQVD files created:")
        for r in success:
            print(f"  {r['name']:<20} {r['rows']:>5} rows  {r['size_kb']:>8.1f} KB")

    if errors:
        print(f"\nErrors:")
        for r in errors:
            print(f"  {r['name']}: {r['error']}")

    print(f"\n✅ Done! QVD files ready in: {args.out}/")
    print(f"   Next: python mcp_server_qvd.py to test pipeline on QVD files")

if __name__ == "__main__":
    main()