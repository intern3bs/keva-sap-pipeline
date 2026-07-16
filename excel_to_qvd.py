"""
SAP Excel → QVD Converter
===========================
Walks the SAP/ folder structure and converts all EXPORT.xlsx files to QVD.
Collection name = parent folder name (VBRK, VBRP, KNA1 etc.)

Structure expected:
  SAP/
    Billing & Revenue Tables/
      VBRK/EXPORT.xlsx   → qvd_files/VBRK.qvd
      VBRP/EXPORT.xlsx   → qvd_files/VBRP.qvd
    Customer Master Tables/
      KNA1/EXPORT.xlsx   → qvd_files/KNA1.qvd
    ...

Usage:
  python excel_to_qvd.py                        # uses ./SAP/ and ./qvd_files/
  python excel_to_qvd.py --sap ./SAP            # custom SAP folder
  python excel_to_qvd.py --out ./qvd_files      # custom output folder
  python excel_to_qvd.py --dry-run              # preview without converting

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import os
import sys
import argparse
import pandas as pd
import pyqvd

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def find_excel_files(sap_dir: str) -> list:
    """
    Walk SAP/ folder and find all EXPORT.xlsx files.
    Returns list of (collection_name, excel_path) tuples.
    Collection name = immediate parent folder of the xlsx file.
    """
    results = []
    for root, dirs, files in os.walk(sap_dir):
        for fname in files:
            if fname.lower().endswith(('.xlsx', '.xls')):
                excel_path      = os.path.join(root, fname)
                collection_name = os.path.basename(root)  # e.g. VBRK
                results.append((collection_name, excel_path))

    # Sort by collection name
    return sorted(results, key=lambda x: x[0])


def excel_to_qvd(collection: str, excel_path: str, out_dir: str) -> dict:
    """Convert a single Excel file to QVD."""
    out_path = os.path.join(out_dir, f"{collection}.qvd")

    print(f"\n{'─'*55}")
    print(f"  Collection : {collection}")
    print(f"  Input      : {excel_path}")
    print(f"  Output     : {collection}.qvd")

    try:
        # Read Excel — try first sheet
        df = pd.read_excel(excel_path, sheet_name=0, engine='openpyxl')

        print(f"  Shape      : {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Columns    : {list(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")

        if df.empty:
            print(f"  ⚠️  Empty — skipped")
            return {"collection": collection, "status": "skipped", "rows": 0}

        # Clean up DataFrame
        # Convert column names to strings
        df.columns = [str(c).strip() for c in df.columns]

        # Convert datetime columns to strings (QVD handles strings best)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif df[col].dtype == object:
                df[col] = df[col].fillna('').astype(str)
                df[col] = df[col].replace('nan', '')
            elif pd.api.types.is_float_dtype(df[col]):
                # Keep floats as-is — QVD handles them
                pass

        # Convert to QVD
        table = pyqvd.QvdTable.from_pandas(df)
        table.to_qvd(out_path)

        size_kb = os.path.getsize(out_path) / 1024
        print(f"  ✅ Written : {collection}.qvd ({size_kb:.1f} KB)")

        return {
            "collection": collection,
            "status":     "success",
            "rows":       len(df),
            "cols":       len(df.columns),
            "out":        out_path,
            "size_kb":    round(size_kb, 1)
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"collection": collection, "status": "error", "error": str(e)}


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Convert SAP Excel exports to QVD files")
    parser.add_argument("--sap",     default="./SAP",       help="SAP folder with subfolders")
    parser.add_argument("--out",     default="./qvd_files", help="Output folder for QVD files")
    parser.add_argument("--dry-run", action="store_true",   help="Preview without converting")
    args = parser.parse_args()

    # Check SAP folder exists
    if not os.path.isdir(args.sap):
        print(f"❌ SAP folder not found: {args.sap}")
        sys.exit(1)

    # Find all Excel files
    files = find_excel_files(args.sap)

    if not files:
        print(f"❌ No Excel files found in {args.sap}")
        sys.exit(1)

    print(f"Found {len(files)} Excel file(s) in {args.sap}/")
    print("=" * 55)
    for col, path in files:
        rel = os.path.relpath(path, args.sap)
        print(f"  {col:<20} ← {rel}")

    if args.dry_run:
        print("\n⚠️  Dry run — no files written")
        return

    # Create output folder
    os.makedirs(args.out, exist_ok=True)
    print(f"\nConverting to: {args.out}/")
    print("=" * 55)

    # Convert each file
    results = [excel_to_qvd(col, path, args.out) for col, path in files]

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
    print(f"  Total rows : {sum(r.get('rows', 0) for r in success):,}")

    if success:
        print(f"\nQVD files created in {args.out}/:")
        for r in success:
            print(f"  {r['collection']:<20} {r['rows']:>6} rows  {r['size_kb']:>8.1f} KB")

    if errors:
        print(f"\nErrors:")
        for r in errors:
            print(f"  {r['collection']}: {r['error']}")

    print(f"\n✅ Done! Run pipeline with: QVD_DIR={args.out} python pipeline_v6.py")


if __name__ == "__main__":
    main()