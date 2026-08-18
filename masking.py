"""
Local-only masking layer. Detects identifying values BEFORE they reach any
cloud call, replaces them with opaque tokens, and substitutes real values
back in afterward. The mapping is per-request, in-memory only, never logged
or persisted.
"""
import re
import json
from mcp_server_qvd import _load_collection, QVD_CACHE

CUSTOMER_ID_RE   = re.compile(r'\b\d{6,10}\b')
MATERIAL_CODE_RE = re.compile(r'\b[A-Z0-9]+/[A-Z0-9]+/[A-Z]{2}\b')

class MaskMap:
    def __init__(self):
        self.token_to_real = {}
        self.real_to_token = {}
        self._n = {"CUST": 0, "MAT": 0, "VAL": 0}

    def get_token(self, real_value, kind):
        if real_value in self.real_to_token:
            return self.real_to_token[real_value]
        self._n[kind] += 1
        token = f"<<{kind}_{self._n[kind]}>>"
        self.token_to_real[token] = real_value
        self.real_to_token[real_value] = token
        return token

    def unmask(self, text: str) -> str:
        for token, real in self.token_to_real.items():
            text = text.replace(token, real)
        return text

    def unmask_json(self, obj):
        """Recursively substitute tokens back into a parsed pipeline/filter
        object, so the query actually runs against the real value."""
        if isinstance(obj, str):
            return self.unmask(obj)
        if isinstance(obj, list):
            return [self.unmask_json(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.unmask_json(v) for k, v in obj.items()}
        return obj


def _customer_name_lookup():
    try:
        _load_collection('KNA1')
        df = QVD_CACHE.get('KNA1')
        name_col = next((c for c in df.columns if 'name 1' in c.lower()), None)
        if name_col:
            return [str(v) for v in df[name_col].dropna().unique()]
    except Exception:
        pass
    return []

def _material_desc_lookup():
    try:
        _load_collection('MAKT')
        df = QVD_CACHE.get('MAKT')
        desc_col = next((c for c in df.columns if 'desc' in c.lower()), None)
        if desc_col:
            return [str(v) for v in df[desc_col].dropna().unique()]
    except Exception:
        pass
    return []

_ner_pipeline = None
def _get_ner():
    """Small local NER model — CPU-friendly, offline. Catches identifying
    phrasing that ID patterns and reference-table lookups miss."""
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        _ner_pipeline = pipeline(
            "ner", model="dslim/bert-base-NER", aggregation_strategy="simple"
        )
    return _ner_pipeline

def mask_question(question: str):
    mm = MaskMap()
    masked = question

    for m in set(CUSTOMER_ID_RE.findall(masked)):
        masked = masked.replace(m, mm.get_token(m, "CUST"))
    for m in set(MATERIAL_CODE_RE.findall(masked)):
        masked = masked.replace(m, mm.get_token(m, "MAT"))

    for name in _customer_name_lookup():
        if name and name.lower() in masked.lower():
            masked = re.sub(re.escape(name), mm.get_token(name, "VAL"), masked, flags=re.IGNORECASE)
    for desc in _material_desc_lookup():
        if desc and desc.lower() in masked.lower():
            masked = re.sub(re.escape(desc), mm.get_token(desc, "VAL"), masked, flags=re.IGNORECASE)

    try:
        for ent in _get_ner()(masked):
            if ent["entity_group"] in ("ORG", "PER") and ent["score"] > 0.80:
                val = ent["word"]
                if val not in mm.real_to_token and len(val) > 2:
                    masked = masked.replace(val, mm.get_token(val, "VAL"))
    except Exception:
        pass  # NER is a best-effort catch-all, never blocks the pipeline

    return masked, mm


def summarize_for_model1(tool_result_json: str) -> str:
    """What gets echoed back to Claude in the tool-use loop instead of the
    real data — shape and success only, never actual values."""
    try:
        parsed = json.loads(tool_result_json)
        if isinstance(parsed, list):
            fields = list(parsed[0].keys()) if parsed and isinstance(parsed[0], dict) else []
            return json.dumps({"status": "ok", "row_count": len(parsed), "fields": fields})
        if isinstance(parsed, dict) and "error" in parsed:
            return tool_result_json  # errors are safe/necessary to see in full
        return json.dumps({"status": "ok"})
    except Exception:
        return json.dumps({"status": "ok"})