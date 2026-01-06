# app/main.py
import json
import re
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime, date

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

# ----------------------------
# JSON + SSE helpers
# ----------------------------

def _json_default(o):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, set):
        return list(o)
    try:
        import asyncpg
        if isinstance(o, asyncpg.Record):
            return dict(o)
    except Exception:
        pass
    return str(o)


def sse(obj: dict) -> str:
    payload = json.dumps(obj, default=_json_default)
    return f"data: {payload}\n\n"


# ----------------------------
# Session helpers
# ----------------------------

def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in [
        "zipcode",
        "payer_like",
        "payment_mode",
        "service_query",
        "variant_cpt_code",
        "variant_name",
    ]:
        v = intent.get(k)
        if v is not None and v != "":
            out[k] = v
    return out


def apply_variant_choice(message: str, merged: Dict[str, Any]) -> Dict[str, Any]:
    if merged.get("_awaiting") != "variant":
        return merged

    opts = merged.get("_variant_options") or []
    msg = (message or "").strip()

    if msg.isdigit():
        idx = int(msg)
        if 1 <= idx <= len(opts):
            chosen = opts[idx - 1]
            merged["variant_cpt_code"] = chosen["cpt_code"]
            merged["variant_name"] = chosen["variant_name"]
            merged.pop("_awaiting", None)
            merged.pop("_variant_options", None)
    return merged


# ----------------------------
# Chat endpoint
# ----------------------------

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    message = body.get("message", "")
    state = body.get("state", {})

    intent = {}
    merged = merge_state(state, intent)
    merged = apply_variant_choice(message, merged)

    async def event_gen():
        if not merged.get("zipcode"):
            yield sse({"type": "delta", "text": "What’s your 5-digit ZIP code?"})
            yield sse({"type": "final"})
            return

        if not merged.get("payment_mode"):
            yield sse({
                "type": "delta",
                "text": "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"
            })
            yield sse({"type": "final"})
            return

        yield sse({"type": "delta", "text": "Pricing flow continues here."})
        yield sse({"type": "final"})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
