"""
SAP ERP QVD MCP Server
========================
Reads directly from QVD files — no MongoDB needed.
Drop-in replacement for mcp_server.py.

TWO USES:
1. Imported by pipeline_v6.py
   from mcp_server_qvd import execute_tool, MCP_TOOLS, SCHEMA_CACHE, db

2. Standalone server for Claude Desktop
   python mcp_server_qvd.py

Set QVD_DIR in .env or pass as env var:
  QVD_DIR=./qvd_files

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import os
import re
import json
import asyncio
import pandas as pd
import pyqvd
from dotenv import load_dotenv

load_dotenv()

QVD_DIR = os.getenv("QVD_DIR", "./qvd_files")
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}')

# ─── LOAD QVD FILES AT STARTUP ────────────────────────────────────────────────
print(f"[QVD] Loading QVD files from: {QVD_DIR}", flush=True)

QVD_CACHE   = {}   # col_name → pandas DataFrame
SCHEMA_CACHE = {}  # col_name → schema info (same format as mcp_server.py)

def _load_qvd_files():
    if not os.path.isdir(QVD_DIR):
        print(f"[QVD] ⚠️  Directory not found: {QVD_DIR}", flush=True)
        return

    for fname in sorted(os.listdir(QVD_DIR)):
        if not fname.lower().endswith(".qvd"):
            continue

        col  = os.path.splitext(fname)[0]
        path = os.path.join(QVD_DIR, fname)

        try:
            table = pyqvd.QvdTable.from_qvd(path)
            df    = table.to_pandas()

            # Clean empty strings → None for consistency
            df = df.replace("", None)

            QVD_CACHE[col] = df

            # Build schema info — same structure as mcp_server.py SCHEMA_CACHE
            sample_row = df.iloc[0].dropna().to_dict() if not df.empty else {}
            fields     = list(df.columns)

            date_ranges = {}
            for k, v in sample_row.items():
                if isinstance(v, str) and DATE_RE.match(str(v)):
                    non_null = df[k].dropna()
                    if not non_null.empty:
                        date_ranges[k] = {
                            "min": str(non_null.min()),
                            "max": str(non_null.max())
                        }

            SCHEMA_CACHE[col] = {
                "fields":      fields,
                "sample":      {k: str(v)[:80] for k, v in sample_row.items()},
                "date_ranges": date_ranges,
                "count":       len(df),
            }

            print(f"[QVD] ✅ {col}: {len(df)} rows × {len(fields)} cols", flush=True)

        except Exception as e:
            print(f"[QVD] ❌ {fname}: {e}", flush=True)

_load_qvd_files()
print(f"[QVD] Loaded: {list(QVD_CACHE.keys())}", flush=True)


# ─── PANDAS AGGREGATION ENGINE ─────────────────────────────────────────────────
# Translates MongoDB aggregation pipeline stages to pandas operations

def _apply_pipeline(df: pd.DataFrame, pipeline: list) -> list:
    """
    Execute a MongoDB aggregation pipeline on a pandas DataFrame.
    Supports: $match, $group, $sort, $limit, $project, $lookup, $unwind, $addFields
    """
    for stage in pipeline:
        if not stage:
            continue
        op = list(stage.keys())[0]

        # ── $match ────────────────────────────────────────────────────────────
        if op == "$match":
            df = _apply_match(df, stage["$match"])

        # ── $group ────────────────────────────────────────────────────────────
        elif op == "$group":
            df = _apply_group(df, stage["$group"])

        # ── $sort ─────────────────────────────────────────────────────────────
        elif op == "$sort":
            sort_spec = stage["$sort"]
            cols  = list(sort_spec.keys())
            asc   = [v == 1 for v in sort_spec.values()]
            valid = [c for c in cols if c in df.columns]
            if valid:
                df = df.sort_values(
                    by=[c for c in cols if c in df.columns],
                    ascending=[asc[i] for i, c in enumerate(cols) if c in df.columns]
                )

        # ── $limit ────────────────────────────────────────────────────────────
        elif op == "$limit":
            df = df.head(int(stage["$limit"]))

        # ── $project ──────────────────────────────────────────────────────────
        elif op == "$project":
            df = _apply_project(df, stage["$project"])

        # ── $addFields ────────────────────────────────────────────────────────
        elif op == "$addFields":
            for field, expr in stage["$addFields"].items():
                df[field] = _eval_expr(df, expr)

        # ── $lookup ───────────────────────────────────────────────────────────
        elif op == "$lookup":
            spec     = stage["$lookup"]
            from_col = spec["from"]
            local    = spec["localField"]
            foreign  = spec["foreignField"]
            as_name  = spec["as"]

            if from_col in QVD_CACHE:
                right = QVD_CACHE[from_col].copy()
                right = right.rename(columns={foreign: local})
                merged = df.merge(right, on=local, how="left", suffixes=("", f"_{from_col}"))
                # Store joined rows as list in as_name column
                df[as_name] = merged.apply(
                    lambda r: [r.to_dict()], axis=1
                )

        # ── $unwind ───────────────────────────────────────────────────────────
        elif op == "$unwind":
            field = stage["$unwind"].lstrip("$")
            if field in df.columns:
                df = df.explode(field).reset_index(drop=True)

    return df.where(pd.notna(df), None).to_dict("records")


def _apply_match(df: pd.DataFrame, match: dict) -> pd.DataFrame:
    """Apply $match filter."""
    mask = pd.Series([True] * len(df), index=df.index)

    for field, condition in match.items():
        if field not in df.columns:
            continue

        col = df[field]

        if isinstance(condition, dict):
            for op, val in condition.items():
                if op == "$gt":
                    mask &= pd.to_numeric(col, errors="coerce") > float(val)
                elif op == "$gte":
                    mask &= pd.to_numeric(col, errors="coerce") >= float(val)
                elif op == "$lt":
                    mask &= pd.to_numeric(col, errors="coerce") < float(val)
                elif op == "$lte":
                    mask &= pd.to_numeric(col, errors="coerce") <= float(val)
                elif op == "$eq":
                    mask &= col == str(val)
                elif op == "$ne":
                    mask &= col != str(val)
                elif op == "$in":
                    mask &= col.isin([str(v) for v in val])
                elif op == "$nin":
                    mask &= ~col.isin([str(v) for v in val])
                elif op == "$exists":
                    mask &= col.notna() if val else col.isna()
        else:
            mask &= col == str(condition)

    return df[mask].reset_index(drop=True)


def _eval_expr(df: pd.DataFrame, expr):
    """Evaluate a MongoDB expression and return a Series."""
    if isinstance(expr, str) and expr.startswith("$"):
        field = expr[1:]
        return pd.to_numeric(df.get(field, pd.Series([None]*len(df))), errors="coerce")

    if isinstance(expr, dict):
        op = list(expr.keys())[0]
        args = expr[op]

        if op == "$sum":
            if args == 1:
                return pd.Series([1] * len(df))
            return _eval_expr(df, args)

        elif op == "$avg":
            return _eval_expr(df, args)

        elif op == "$max":
            return _eval_expr(df, args)

        elif op == "$min":
            return _eval_expr(df, args)

        elif op == "$first":
            return _eval_expr(df, args)

        elif op == "$toDouble":
            return pd.to_numeric(_eval_expr(df, args), errors="coerce")

        elif op == "$toString":
            return _eval_expr(df, args).astype(str)

        elif op == "$subtract":
            a, b = [_eval_expr(df, x) for x in args]
            return pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")

        elif op == "$add":
            a, b = [_eval_expr(df, x) for x in args]
            return pd.to_numeric(a, errors="coerce") + pd.to_numeric(b, errors="coerce")

        elif op == "$multiply":
            a, b = [_eval_expr(df, x) for x in args]
            return pd.to_numeric(a, errors="coerce") * pd.to_numeric(b, errors="coerce")

        elif op == "$divide":
            a, b = [_eval_expr(df, x) for x in args]
            b_num = pd.to_numeric(b, errors="coerce")
            return pd.to_numeric(a, errors="coerce") / b_num.replace(0, float("nan"))

        elif op == "$round":
            val, decimals = args[0], args[1] if len(args) > 1 else 2
            return _eval_expr(df, val).round(int(decimals))

        elif op == "$size":
            col = _eval_expr(df, args)
            return col.apply(lambda x: len(x) if isinstance(x, (list, set)) else 0)

        elif op == "$addToSet":
            return _eval_expr(df, args)

    if isinstance(expr, (int, float)):
        return pd.Series([expr] * len(df))

    return pd.Series([None] * len(df))


def _apply_group(df: pd.DataFrame, group_spec: dict) -> pd.DataFrame:
    """Apply $group aggregation."""
    id_spec  = group_spec.get("_id")
    agg_spec = {k: v for k, v in group_spec.items() if k != "_id"}

    # Determine group keys
    if id_spec is None:
        df["_group_key"] = "_all"
        group_keys = ["_group_key"]
    elif isinstance(id_spec, str) and id_spec.startswith("$"):
        field = id_spec[1:]
        df["_group_key"] = df.get(field, pd.Series([None]*len(df)))
        group_keys = ["_group_key"]
    elif isinstance(id_spec, dict):
        for alias, field_expr in id_spec.items():
            if isinstance(field_expr, str) and field_expr.startswith("$"):
                df[f"_gk_{alias}"] = df.get(field_expr[1:], pd.Series([None]*len(df)))
            else:
                df[f"_gk_{alias}"] = _eval_expr(df, field_expr)
        group_keys = [f"_gk_{k}" for k in id_spec.keys()]
    else:
        df["_group_key"] = str(id_spec)
        group_keys = ["_group_key"]

    grouped = df.groupby(group_keys, dropna=False)

    # Apply aggregations
    result_data = {"_id": []}
    for agg_name, agg_expr in agg_spec.items():
        result_data[agg_name] = []

    id_vals = []
    agg_vals = {k: [] for k in agg_spec}

    for key, group in grouped:
        # Build _id value
        if id_spec is None:
            id_vals.append(None)
        elif isinstance(id_spec, dict):
            keys = list(id_spec.keys())
            if isinstance(key, tuple):
                id_vals.append(dict(zip(keys, key)))
            else:
                id_vals.append({keys[0]: key})
        else:
            id_vals.append(key)

        # Compute aggregations
        for agg_name, agg_expr in agg_spec.items():
            if not isinstance(agg_expr, dict):
                agg_vals[agg_name].append(None)
                continue
            op   = list(agg_expr.keys())[0]
            expr = agg_expr[op]

            series = _eval_expr(group, expr)
            numeric = pd.to_numeric(series, errors="coerce")

            if op == "$sum":
                if expr == 1:
                    agg_vals[agg_name].append(len(group))
                else:
                    agg_vals[agg_name].append(numeric.sum())
            elif op == "$avg":
                agg_vals[agg_name].append(numeric.mean())
            elif op == "$max":
                agg_vals[agg_name].append(numeric.max() if numeric.notna().any() else series.max())
            elif op == "$min":
                agg_vals[agg_name].append(numeric.min() if numeric.notna().any() else series.min())
            elif op == "$first":
                agg_vals[agg_name].append(series.iloc[0] if len(series) > 0 else None)
            elif op == "$addToSet":
                agg_vals[agg_name].append(list(series.dropna().unique()))
            elif op == "$push":
                agg_vals[agg_name].append(series.tolist())
            else:
                agg_vals[agg_name].append(None)

    result = pd.DataFrame({"_id": id_vals, **agg_vals})

    # Drop internal group key columns
    for col in df.columns:
        if col.startswith("_gk_") or col == "_group_key":
            if col in result.columns:
                result = result.drop(columns=[col])

    return result


def _apply_project(df: pd.DataFrame, project_spec: dict) -> pd.DataFrame:
    """Apply $project — include/exclude fields and compute expressions."""
    result = pd.DataFrame(index=df.index)

    for field, spec in project_spec.items():
        if field == "_id":
            if spec == 0:
                continue
            if "_id" in df.columns:
                result["_id"] = df["_id"]
        elif spec == 1:
            if field in df.columns:
                result[field] = df[field]
        elif spec == 0:
            pass  # exclude
        elif isinstance(spec, (dict, str)):
            # Compute expression
            val = _eval_expr(df, spec)
            result[field] = val
        else:
            result[field] = spec

    return result


# ─── MCP TOOL DEFINITIONS ─────────────────────────────────────────────────────
# Identical to mcp_server.py — same tool names and descriptions
MCP_TOOLS = [
    {
        "name": "list_sap_collections",
        "description": (
            "List all SAP collections loaded from QVD files. "
            "Call this first to discover available tables."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_sap_schema",
        "description": (
            "Get exact field names, sample values, and date ranges for a SAP collection. "
            "Always call this before querying to confirm exact field names and casing. "
            "Field names are case-sensitive: 'Net Value' (VBRK) vs 'Net value' (VBRP)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name e.g. VBRK, VBRP, VBAK"
                }
            },
            "required": ["collection"]
        }
    },
    {
        "name": "query_sap_collection",
        "description": (
            "Run a MongoDB-style aggregation pipeline on a SAP QVD collection. "
            "Use for grouping, ranking, totals, margins, growth, joins, filtering. "
            "ROUTING: Customer queries -> VBRK ('Sold-To Party', 'Net Value' capital V). "
            "Product/margin queries -> VBRP ('Material', 'Net value' lowercase v, 'Cost'). "
            "Sales office -> VBAK ('Sales Office'). "
            "Filter queries (find where X > Y) -> VBRK or VBRP with $match. "
            "Join VBRP<->VBRK on 'Billing Document'. "
            "Margin: ($project after $group) (rev-cost)/rev*100. Filter Cost>0 first. "
            "Date filters only if question explicitly mentions a time period. "
            "FORBIDDEN: NEVER query LIKP or LIPS for sales/revenue/growth questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "pipeline":   {"type": "array", "items": {"type": "object"}},
                "limit":      {"type": "integer", "default": 100}
            },
            "required": ["collection", "pipeline"]
        }
    },
    {
        "name": "find_sap_documents",
        "description": (
            "Find documents in a SAP QVD collection with optional filter. "
            "Use for fetching specific records, looking up customers or materials."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "filter":     {"type": "object", "default": {}},
                "fields":     {"type": "array", "items": {"type": "string"}, "default": []},
                "limit":      {"type": "integer", "default": 10}
            },
            "required": ["collection"]
        }
    }
]

# ─── TOOL EXECUTOR ────────────────────────────────────────────────────────────
def execute_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "list_sap_collections":
            summary = {
                col: {"document_count": info["count"], "fields": info["fields"][:15]}
                for col, info in SCHEMA_CACHE.items()
            }
            return json.dumps(summary, indent=2)

        elif tool_name == "get_sap_schema":
            col = tool_input["collection"]
            if col not in SCHEMA_CACHE:
                return json.dumps({
                    "error":     f"Collection '{col}' not found",
                    "available": list(SCHEMA_CACHE.keys())
                })
            info   = SCHEMA_CACHE[col]
            result = {
                "collection":  col,
                "total_docs":  info["count"],
                "fields":      info["fields"],
                "sample":      info["sample"],
                "date_ranges": info["date_ranges"],
            }
            if col in ("LIKP", "LIPS"):
                result["WARNING"] = (
                    "LIKP/LIPS = delivery logistics only. "
                    "Do NOT use for sales, revenue, or growth analysis."
                )
            return json.dumps(result, indent=2)

        elif tool_name == "query_sap_collection":
            col      = tool_input["collection"]
            pipeline = tool_input["pipeline"]
            limit    = int(tool_input.get("limit", 500))

            if col not in QVD_CACHE:
                return json.dumps({"error": f"Collection '{col}' not found"})

            df   = QVD_CACHE[col].copy()
            rows = _apply_pipeline(df, pipeline)

            # Clean None/NaN for JSON serialization
            clean = []
            for r in rows[:limit]:
                clean.append({
                    k: (None if v != v else v)   # NaN → None
                    for k, v in r.items()
                    if k != "_id" or v is not None
                })
            return json.dumps(clean, indent=2, default=str)

        elif tool_name == "find_sap_documents":
            col    = tool_input["collection"]
            filt   = tool_input.get("filter", {})
            fields = tool_input.get("fields", [])
            limit  = int(tool_input.get("limit", 10))

            if col not in QVD_CACHE:
                return json.dumps({"error": f"Collection '{col}' not found"})

            df = QVD_CACHE[col].copy()
            if filt:
                df = _apply_match(df, filt)
            if fields:
                df = df[[f for f in fields if f in df.columns]]

            rows = df.head(limit).where(pd.notna(df), None).to_dict("records")
            return json.dumps(rows, indent=2, default=str)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {e}"})


# Compatibility alias — pipeline_v6.py imports `db` from mcp_server
# For QVD mode we provide a dummy db object for RAG BM25 search fallback
class _DummyDB:
    def list_collection_names(self):
        return list(QVD_CACHE.keys())
    def __getitem__(self, name):
        return _DummyCollection(QVD_CACHE.get(name, pd.DataFrame()))

class _DummyCollection:
    def __init__(self, df):
        self._df = df
    def find(self, query=None, projection=None, **kwargs):
        return iter(self._df.head(10).to_dict("records"))
    def count_documents(self, query=None):
        return len(self._df)

db = _DummyDB()


# ─── STANDALONE MCP SERVER ────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types

        server = Server("sap-erp-qvd")

        @server.list_tools()
        async def list_tools():
            return [
                types.Tool(name=t["name"], description=t["description"],
                           inputSchema=t["input_schema"])
                for t in MCP_TOOLS
            ]

        @server.call_tool()
        async def call_tool(name, arguments):
            result = execute_tool(name, arguments)
            return [types.TextContent(type="text", text=result)]

        async def main():
            print("[QVD] SAP ERP QVD server starting on stdio...", flush=True)
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream,
                                 server.create_initialization_options())

        asyncio.run(main())

    except ImportError:
        print("[QVD] Run as library: from mcp_server_qvd import execute_tool")