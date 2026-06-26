"""
SAP ERP RAG Pipeline — Prompts Loader
=======================================
Loads prompts from skills/ folder.
Pipeline v6 only — edit prompts in skills/*.md, not here.

Skill files:
  skills/mcp_system.md        — Claude Model 1 system prompt
  skills/abap_generation.md   — ABAP query generation
  skills/semantic_format.md   — RAG semantic answer formatting
  skills/aggregate_format.md  — MongoDB results formatting

Author  : Rohit Kumar
Project : SAP ERP RAG Pipeline — Keva Fragrances Internship
"""

import os

_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")

def _load(filename: str) -> str:
    path = os.path.join(_SKILLS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

MCP_SYSTEM_PROMPT       = _load("mcp_system.md")
ABAP_PROMPT             = _load("abap_generation.md")
SEMANTIC_FORMAT_PROMPT  = _load("semantic_format.md")
AGGREGATE_FORMAT_PROMPT = _load("aggregate_format.md")