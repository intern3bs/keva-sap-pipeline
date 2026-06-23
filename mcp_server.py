"""
SAP ERP MongoDB MCP Server
===========================
Exposes SAP SD MongoDB collections as MCP tools.
Claude calls these tools directly instead of generating Python code strings.

Why this is more deterministic (per team lead):
  - Claude provides structured JSON params → no Python syntax errors
  - MongoDB executes directly → no safe_exec needed
  - Schema validation built in → wrong field names caught immediately

Architecture:
  Claude API → calls MCP tools (JSON pipeline) → this server → MongoDB Atlas

Usage:
  pip install mcp pymongo python-dotenv
  python mcp_server.py

Claude Desktop config (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "sap-erp": {
        "command": "python",
        "args": ["/path/to/mcp_server.py"],
        "env": {
          "MONGO_URI": "mongodb+srv://...",
          "DB_NAME": "sap_erp"
        }
      }
    }
  }

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import asyncio
import json
import os
import re
from pymongo import MongoClient
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

load_dotenv()

# ─── CONNECTIONS ──────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME", "sap_erp")

mongo_client = MongoClient(MONGO_URI)
db           = mongo_client[DB_NAME]
server       = Server("sap-erp-mongodb")

DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}')

# ─── SCHEMA CACHE — built once at startup ─────────────────────────────────────
def build_schema_cache() -> dict:
    """Load schema + date ranges for all SAP collections at startup."""
    cache = {}
    for col in db.list_collection_names():
        if col.startswith("_"):
            continue
        sample    = db[col].find_one({}, {"_id": 0}) or {}
        non_empty = {k: v for k, v in sample.items() if v not in [None, "", 0, "0"]}
        fields    = list(sample.keys())

        date_ranges = {}
        for k, v in non_empty.items():
            if isinstance(v, str) and DATE_RE.match(v):
                mn = db[col].find_one({k: {"$nin": [None, ""]}}, {k: 1, "_id": 0}, sort=[(k,  1)])
                mx = db[col].find_one({k: {"$nin": [None, ""]}}, {k: 1, "_id": 0}, sort=[(k, -1)])
                if mn and mx:
                    date_ranges[k] = {"min": mn.get(k), "max": mx.get(k)}

        cache[col] = {
            "fields":      fields,
            "sample":      {k: str(v)[:80] for k, v in non_empty.items()},
            "date_ranges": date_ranges,
            "count":       db[col].count_documents({}),
        }
    return cache

SCHEMA_CACHE = build_schema_cache()
print(f"[MCP] Schema loaded: {list(SCHEMA_CACHE.keys())}", flush=True)

# ─── TOOL DEFINITIONS ─────────────────────────────────────────────────────────
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [

        types.Tool(
            name="list_sap_collections",
            description=(
                "List all SAP ERP collections in MongoDB with their field names and document counts. "
                "Call this first to discover available SAP tables (VBRK, VBRP, VBAK, VBAP, KNA1 etc)."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),

        types.Tool(
            name="get_sap_schema",
            description=(
                "Get exact field names, sample values, data types, and date ranges for a SAP collection. "
                "Always call this before querying to know the exact field names and their casing. "
                "Field names are case-sensitive — e.g. 'Net Value' vs 'Net value' are different fields."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "SAP collection name e.g. VBRK, VBRP, VBAK, VBAP, KNA1, KNVV, MARA"
                    }
                },
                "required": ["collection"]
            }
        ),

        types.Tool(
            name="query_sap_collection",
            description=(
                "Run a MongoDB aggregation pipeline on a SAP collection and return results. "
                "Use for: grouping, ranking, totals, margins, growth calculations, joins. "
                "COLLECTION ROUTING: "
                "  - Customer queries → VBRK (has Sold-To Party, Net Value capital V) "
                "  - Product/margin queries → VBRP (has Material, Net value lowercase v, Cost) "
                "  - Sales office queries → VBAK (has Sales Office) "
                "  - Join VBRP to VBRK on 'Billing Document' field "
                "Always use exact field names from get_sap_schema."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "SAP collection name"
                    },
                    "pipeline": {
                        "type": "array",
                        "description": "MongoDB aggregation pipeline stages as JSON array",
                        "items": {"type": "object"}
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 100)",
                        "default": 100
                    }
                },
                "required": ["collection", "pipeline"]
            }
        ),

        types.Tool(
            name="find_sap_documents",
            description=(
                "Find documents in a SAP collection with optional filter. "
                "Use for: fetching specific records, looking up customers, materials, payment terms. "
                "For large result sets use query_sap_collection with aggregation instead."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "SAP collection name"
                    },
                    "filter": {
                        "type": "object",
                        "description": "MongoDB filter query (default {} returns all)",
                        "default": {}
                    },
                    "fields": {
                        "type": "array",
                        "description": "Field names to return (default: all fields)",
                        "items": {"type": "string"},
                        "default": []
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max documents to return (default 10)",
                        "default": 10
                    }
                },
                "required": ["collection"]
            }
        ),

    ]

# ─── TOOL HANDLERS ────────────────────────────────────────────────────────────
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:

        # ── list_sap_collections ──────────────────────────────────────────────
        if name == "list_sap_collections":
            summary = {
                col: {
                    "document_count": info["count"],
                    "fields":         info["fields"][:20],
                }
                for col, info in SCHEMA_CACHE.items()
            }
            return [types.TextContent(type="text", text=json.dumps(summary, indent=2))]

        # ── get_sap_schema ────────────────────────────────────────────────────
        elif name == "get_sap_schema":
            col = arguments["collection"]
            if col not in SCHEMA_CACHE:
                return [types.TextContent(type="text",
                    text=f"Collection '{col}' not found. Available: {list(SCHEMA_CACHE.keys())}")]
            info   = SCHEMA_CACHE[col]
            result = {
                "collection":  col,
                "total_docs":  info["count"],
                "fields":      info["fields"],
                "sample":      info["sample"],
                "date_ranges": info["date_ranges"],
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        # ── query_sap_collection ──────────────────────────────────────────────
        elif name == "query_sap_collection":
            col      = arguments["collection"]
            pipeline = arguments["pipeline"]
            limit    = int(arguments.get("limit", 100))

            if col not in SCHEMA_CACHE:
                return [types.TextContent(type="text",
                    text=f"Collection '{col}' not found. Available: {list(SCHEMA_CACHE.keys())}")]

            rows = list(db[col].aggregate(pipeline))
            for r in rows:
                r.pop("_id", None)
            rows = rows[:limit]

            return [types.TextContent(
                type="text",
                text=json.dumps(rows, indent=2, default=str)
            )]

        # ── find_sap_documents ────────────────────────────────────────────────
        elif name == "find_sap_documents":
            col    = arguments["collection"]
            filt   = arguments.get("filter", {})
            fields = arguments.get("fields", [])
            limit  = int(arguments.get("limit", 10))

            if col not in SCHEMA_CACHE:
                return [types.TextContent(type="text",
                    text=f"Collection '{col}' not found. Available: {list(SCHEMA_CACHE.keys())}")]

            projection = {"_id": 0}
            if fields:
                for f in fields:
                    projection[f] = 1

            rows = list(db[col].find(filt, projection).limit(limit))
            return [types.TextContent(
                type="text",
                text=json.dumps(rows, indent=2, default=str)
            )]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [types.TextContent(type="text", text=f"Error in {name}: {e}")]

# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def main():
    print("[MCP] SAP ERP MongoDB server starting on stdio...", flush=True)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())