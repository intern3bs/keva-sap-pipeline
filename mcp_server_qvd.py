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
from altair import sample
from altair import sample
import pandas as pd
import pyqvd
from dotenv import load_dotenv
from collections import OrderedDict
load_dotenv()

QVD_DIR = os.getenv("QVD_DIR", "./qvd_files")
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}')

# ─── LOAD QVD FILES AT STARTUP ────────────────────────────────────────────────
print(f"[QVD] Loading QVD files from: {QVD_DIR}", flush=True)

MAX_CACHED_COLLECTIONS = int(os.getenv("QVD_MAX_CACHE", "5"))
QVD_CACHE    = OrderedDict()  # LRU cache — evicts least recently used
SCHEMA_CACHE = {}             # schema index always kept (tiny)

QVD_FILES = {}  # col_name → file path (index only, no data loaded)

def _scan_qvd_files():
    """Scan QVD directory and build index — does NOT load data."""
    if not os.path.isdir(QVD_DIR):
        print(f"[QVD] ⚠️  Directory not found: {QVD_DIR}", flush=True)
        return

    for fname in sorted(os.listdir(QVD_DIR)):
        if not fname.lower().endswith(".qvd"):
            continue
        col = os.path.splitext(fname)[0]
        QVD_FILES[col] = os.path.join(QVD_DIR, fname)
        print(f"[QVD] 📋 Found: {col} ({os.path.getsize(QVD_FILES[col])/1024:.1f} KB)", flush=True)

    print(f"[QVD] Index built: {list(QVD_FILES.keys())} — loading on demand", flush=True)

def _load_collection(col: str) -> bool:
    """Load a single QVD file into QVD_CACHE and SCHEMA_CACHE on demand."""
    if col in QVD_CACHE:
        return True  # already loaded

    if col not in QVD_FILES:
        return False  # file doesn't exist

    path = QVD_FILES[col]
    try:
        print(f"[QVD] Loading {col} from {path}...", flush=True)
        table = pyqvd.QvdTable.from_qvd(path)
        df    = table.to_pandas()
        df    = df.replace("", None)
        QVD_CACHE[col] = df
        QVD_CACHE.move_to_end(col)  # mark as recently used

        # Evict oldest if over limit
        while len(QVD_CACHE) > MAX_CACHED_COLLECTIONS:
            evicted = next(iter(QVD_CACHE))
            del QVD_CACHE[evicted]
            print(f"[QVD] ♻️  Evicted {evicted} from cache (limit={MAX_CACHED_COLLECTIONS})", flush=True)

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

# Get real row count efficiently
        full_table = pyqvd.QvdTable.from_qvd(path)
        row_count  = full_table.shape[0]

        SCHEMA_CACHE[col] = {
            "fields":      fields,
            "sample":      {k: str(v)[:80] for k, v in sample_row.items()},
            "date_ranges": date_ranges,
            "count":       row_count,
        }
        print(f"[QVD] ✅ Loaded {col}: {len(df)} rows × {len(fields)} cols", flush=True)
        return True

    except Exception as e:
        print(f"[QVD] ❌ Failed to load {col}: {e}", flush=True)
        return False

_scan_qvd_files()

def _build_schema_index():
    """Build schema info for all collections without loading full data."""
    for col, path in QVD_FILES.items():
        if col in SCHEMA_CACHE:
            continue
        try:
            # Read only first row for schema
            table = pyqvd.QvdTable.from_qvd(path)
            df_head = table.to_pandas().head(1)
            fields  = list(df_head.columns)
            sample  = df_head.iloc[0].dropna().to_dict() if not df_head.empty else {}

            SCHEMA_CACHE[col] = {
                "fields":      fields,
                "sample":      {k: str(v)[:80] for k, v in sample.items()},
                "date_ranges": {},  # will fill on first real load
                "count":       -1,  # unknown until loaded
            }
        except Exception as e:
            print(f"[QVD] ⚠️  Schema preview failed for {col}: {e}", flush=True)

_build_schema_index()


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
            spec = stage["$lookup"]
            as_name = spec.get("as", "_lookup_result")

            if "localField" not in spec or "foreignField" not in spec:
                # Unsupported advanced $lookup (let/pipeline form).
                # Fail safely instead of crashing — leave an empty list so
                # downstream $unwind/$group just see no matches, and the
                # question routes to "no data" rather than erroring out.
                df[as_name] = [[] for _ in range(len(df))]
            else:
                from_col = spec["from"]
                local    = spec["localField"]
                foreign  = spec["foreignField"]

                if from_col in QVD_CACHE:
                    right = QVD_CACHE[from_col].copy()
                    right = right.rename(columns={foreign: local})
                    df[local]    = df[local].astype(str)
                    right[local] = right[local].astype(str)
                    merged = df.merge(right, on=local, how="left", suffixes=("", f"_{from_col}"))
                    df[as_name] = merged.apply(lambda r: [r.to_dict()], axis=1)
                else:
                    df[as_name] = [[] for _ in range(len(df))]

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

        col_str = df[field].astype(str)  # normalized for exact-match ops

        if isinstance(condition, dict):
            for op, val in condition.items():
                if op == "$gt":
                    mask &= pd.to_numeric(df[field], errors="coerce") > float(val)
                elif op == "$gte":
                    mask &= pd.to_numeric(df[field], errors="coerce") >= float(val)
                elif op == "$lt":
                    mask &= pd.to_numeric(df[field], errors="coerce") < float(val)
                elif op == "$lte":
                    mask &= pd.to_numeric(df[field], errors="coerce") <= float(val)
                elif op == "$eq":
                    mask &= col_str == str(val)
                elif op == "$ne":
                    mask &= col_str != str(val)
                elif op == "$in":
                    mask &= col_str.isin([str(v) for v in val])
                elif op == "$nin":
                    mask &= ~col_str.isin([str(v) for v in val])
                elif op == "$exists":
                    mask &= df[field].notna() if val else df[field].isna()
        else:
            mask &= col_str == str(condition)

    return df[mask].reset_index(drop=True)


def _eval_expr(df: pd.DataFrame, expr):
    """Evaluate a MongoDB expression and return a Series."""
    if isinstance(expr, str) and expr.startswith("$"):
        field = expr[1:]
        if "." in field:
            base_field, nested_path = field.split(".", 1)
            base_series = df.get(base_field, pd.Series([None] * len(df)))
            keys = nested_path.split(".")
            def _dig(v):
                cur = v
                for k in keys:
                    if isinstance(cur, dict):
                        cur = cur.get(k)
                    else:
                        return None
                return cur
            return base_series.apply(_dig)
        # Return raw values — numeric coercion happens at call sites
        # that need it ($sum/$avg/etc already re-coerce in _apply_group).
        return df.get(field, pd.Series([None] * len(df)))

    if isinstance(expr, dict):
        op = list(expr.keys())[0]
        args = expr[op]

        if op == "$toLong":
            return pd.to_numeric(_eval_expr(df, args), errors="coerce").astype("Int64")

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


# ─── GENERALIZED AUTO-ENRICHMENT ──────────────────────────────────────────────
# Any collection listed here is a "master/reference" table. After ANY query
# result comes back, every column is checked for value-overlap against each
# reference table's ID column. On a match, the description column is attached
# automatically — regardless of what the column is named (_id, Customer,
# Sold-To Party...) and regardless of which question was asked.
# Add a new master table by adding one entry here — no other code changes.

REFERENCE_TABLES = [
    {
        "collection": "KNA1",
        "id_field_candidates": ["customer", "kunnr", "cust."],
        "desc_field_keywords": ["name 1"],
        "output_field": "Customer Name",
    },
    {
        "collection": "MAKT",
        "id_field_candidates": ["material", "matnr"],
        "desc_field_keywords": ["material description", "maktx", "desc"],
        "output_field": "Material Description",
    },
    # Add more master tables here, e.g.:
    # {"collection": "TVKO", "id_field_candidates": ["sales organization", "vkorg"],
    #  "desc_field_keywords": ["name"], "output_field": "Sales Org Name"},
]

def _norm_id(v):
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) == 1 else v
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _find_join_column(rows, ref_df, ref_id_col, min_matches=1):
    """Find which column in `rows` best overlaps with ref_df[ref_id_col]'s values,
    by absolute match count — works even if the reference table only covers
    a subset of real IDs (partial/fake master data)."""
    if ref_df is None or ref_id_col not in ref_df.columns or not rows:
        return None
    ref_ids = set(ref_df[ref_id_col].dropna().map(_norm_id))
    if not ref_ids:
        return None
    best_col, best_count = None, 0
    for col in rows[0].keys():
        vals = [_norm_id(r.get(col)) for r in rows if r.get(col) not in (None, "")]
        if not vals:
            continue
        matched = sum(1 for v in vals if v in ref_ids)
        if matched > best_count:
            best_count, best_col = matched, col
    return best_col if best_count >= min_matches else None

def _enrich_with_lookup(rows, ref_collection, id_field_candidates,
                         desc_field_keywords, output_field):
    if not rows:
        return
    try:
        _load_collection(ref_collection)
        ref_df = QVD_CACHE.get(ref_collection)
    except Exception:
        ref_df = None
    if ref_df is None:
        return

    id_col = next((c for c in ref_df.columns if c.lower() in id_field_candidates), None)
    val_col = next((c for c in ref_df.columns
                     if any(k in c.lower() for k in desc_field_keywords)), None)
    if not id_col or not val_col:
        return

    join_col = _find_join_column(rows, ref_df, id_col)
    if not join_col:
        return

    lookup = dict(zip(ref_df[id_col].map(_norm_id), ref_df[val_col].astype(str)))
    for row in rows:
        key = _norm_id(row.get(join_col))
        if key in lookup and lookup[key] not in ("nan", "None", ""):
            row[output_field] = lookup[key]

def auto_enrich(rows):
    """Run every registered reference-table enrichment against `rows` in place.
    Safe to call on ANY result set from ANY question — reference tables that
    don't match anything in this result simply add nothing."""
    if not rows or not isinstance(rows[0], dict):
        return rows
    for ref in REFERENCE_TABLES:
        try:
            _enrich_with_lookup(
                rows, ref["collection"],
                ref["id_field_candidates"],
                ref["desc_field_keywords"],
                ref["output_field"],
            )
        except Exception:
            continue  # one bad reference table should never break the others
    return rows


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
    },
    {
        "name": "cache_status",
        "description": "Show which collections are currently loaded in memory.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
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

            # Lazy load — only load this collection if not already in cache
            if col not in QVD_CACHE:
                if not _load_collection(col):
                    return json.dumps({"error": f"Collection '{col}' not found"})
                else:
                    QVD_CACHE.move_to_end(col)

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
                if not _load_collection(col):
                    return json.dumps({"error": f"Collection '{col}' not found"})

            df = QVD_CACHE[col].copy()
            if filt:
                df = _apply_match(df, filt)
            if fields:
                df = df[[f for f in fields if f in df.columns]]

            rows = df.head(limit).where(pd.notna(df), None).to_dict("records")
            return json.dumps(rows, indent=2, default=str)
        elif tool_name == "cache_status":
            status = {
                "loaded":     list(QVD_CACHE.keys()),
                "available":  list(QVD_FILES.keys()),
                "not_loaded": [c for c in QVD_FILES if c not in QVD_CACHE],
                "max_cache":  MAX_CACHED_COLLECTIONS,
            }
            return json.dumps(status, indent=2)
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