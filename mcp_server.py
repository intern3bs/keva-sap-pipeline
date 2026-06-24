"""
SAP ERP MongoDB MCP Server
===========================
Exposes SAP SD MongoDB collections as MCP tools.

TWO USES:
1. Imported by pipeline_v6.py
   from mcp_server import execute_tool, MCP_TOOLS, SCHEMA_CACHE, db

2. Standalone server for Claude Desktop
   python mcp_server.py
   Config in claude_desktop_config.json:
   {
     "mcpServers": {
       "sap-erp": {
         "command": "python",
         "args": ["/path/to/mcp_server.py"]
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

load_dotenv()

# ─── CONNECTIONS ──────────────────────────────────────────────────────────────
MONGO_URI    = os.getenv("MONGO_URI")
DB_NAME      = os.getenv("DB_NAME", "sap_erp")
mongo_client = MongoClient(MONGO_URI)
db           = mongo_client[DB_NAME]
DATE_RE      = re.compile(r'^\d{4}-\d{2}-\d{2}')

# ─── SCHEMA CACHE — built once at startup ─────────────────────────────────────
def build_schema_cache() -> dict:
    """Load schema + date ranges for all SAP collections."""
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

# ─── MCP TOOL DEFINITIONS ─────────────────────────────────────────────────────
# Shared between pipeline_v6.py (as Anthropic tool specs) and standalone server
MCP_TOOLS = [
    {
        "name": "list_sap_collections",
        "description": (
            "List all SAP ERP collections in MongoDB with field names and document counts. "
            "Call this first to discover available SAP tables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
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
                    "description": "SAP collection name e.g. VBRK, VBRP, VBAK, VBAP, KNA1"
                }
            },
            "required": ["collection"]
        }
    },
    {
        "name": "query_sap_collection",
        "description": (
            "Run a MongoDB aggregation pipeline on a SAP collection. "
            "Use for grouping, ranking, totals, margins, growth, joins. "
            "ROUTING: Customer queries -> VBRK ('Sold-To Party', 'Net Value' capital V). "
            "Product/margin queries -> VBRP ('Material', 'Net value' lowercase v, 'Cost'). "
            "Sales office -> VBAK ('Sales Office'). "
            "Join VBRP<->VBRK on 'Billing Document'. "
            "Margin: ($project after $group) (rev-cost)/rev*100. Filter Cost>0 first. "
            "Date filters only if question explicitly mentions a time period."
        ),
        "input_schema": {
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
    },
    {
        "name": "find_sap_documents",
        "description": (
            "Find documents in a SAP collection with optional filter. "
            "Use for fetching specific records, looking up customers or materials."
        ),
        "input_schema": {
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
                    "description": "Field names to return (default: all)",
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
    }
]

# ─── TOOL EXECUTOR ────────────────────────────────────────────────────────────
# Called by pipeline_v6.py directly (no server process needed)
# Also called by standalone MCP server when running for Claude Desktop

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute an MCP tool call against MongoDB Atlas.
    Returns JSON string result.
    """
    try:
        if tool_name == "list_sap_collections":
            summary = {
                col: {
                    "document_count": info["count"],
                    "fields":         info["fields"][:15],
                }
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
            info = SCHEMA_CACHE[col]
            return json.dumps({
                "collection":  col,
                "total_docs":  info["count"],
                "fields":      info["fields"],
                "sample":      info["sample"],
                "date_ranges": info["date_ranges"],
            }, indent=2)

        elif tool_name == "query_sap_collection":
            col      = tool_input["collection"]
            pipeline = tool_input["pipeline"]
            limit    = int(tool_input.get("limit", 100))
            if col not in SCHEMA_CACHE:
                return json.dumps({"error": f"Collection '{col}' not found"})
            rows = list(db[col].aggregate(pipeline))
            for r in rows:
                r.pop("_id", None)
            return json.dumps(rows[:limit], indent=2, default=str)

        elif tool_name == "find_sap_documents":
            col    = tool_input["collection"]
            filt   = tool_input.get("filter", {})
            fields = tool_input.get("fields", [])
            limit  = int(tool_input.get("limit", 10))
            if col not in SCHEMA_CACHE:
                return json.dumps({"error": f"Collection '{col}' not found"})
            proj = {"_id": 0}
            if fields:
                for f in fields:
                    proj[f] = 1
            rows = list(db[col].find(filt, proj).limit(limit))
            return json.dumps(rows, indent=2, default=str)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {e}"})


# ─── STANDALONE MCP SERVER — for Claude Desktop only ──────────────────────────
if __name__ == "__main__":
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types

        server = Server("sap-erp-mongodb")

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["input_schema"]
                )
                for t in MCP_TOOLS
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            result = execute_tool(name, arguments)
            return [types.TextContent(type="text", text=result)]

        async def main():
            print("[MCP] SAP ERP server starting on stdio...", flush=True)
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options()
                )

        asyncio.run(main())

    except ImportError:
        print("[MCP] 'mcp' package not installed.")
        print("      Install: pip install mcp")
        print("      For pipeline_v6.py usage, mcp package is not needed.")