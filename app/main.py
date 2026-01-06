import os
import json
import re
import uuid
import time
import logging
from typing import Optional, Any, Dict, List, Tuple

import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI, OpenAIError

# Python fallback refiners (file: app/service_refiners.py)
# If this file is missing, the code will default to an empty registry without crashing.
try:
    from app.service_refiners import refiners_registry
except ImportError:
    def refiners_registry(): return {"refiners": []}

# ----------------------------
# Config
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("costsavvy")

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

APP_API_KEY = os.getenv("APP_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MIN_DB_RESULTS_BEFORE_WEB = int(os.getenv("MIN_DB_RESULTS_BEFORE_WEB", "3"))
MIN_FACILITIES_TO_DISPLAY = int(os.getenv("MIN_FACILITIES_TO_DISPLAY", "5"))
REFINERS_CACHE_TTL_SECONDS = int(os.getenv("REFINERS_CACHE_TTL_SECONDS", "300"))

INTENT_OVERRIDE_FORCE_PRICE_ENABLED = os.getenv("INTENT_OVERRIDE_FORCE_PRICE_ENABLED", "true").lower() in ("1", "true", "yes", "y", "on")
INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS = [s.strip().lower() for s in os.getenv("INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS", "cost,price,how much,pricing,estimate,rate,charge,fee").split(",") if s.strip()]
RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION = os.getenv("RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION", "true").lower() in ("1", "true", "yes", "y", "on")

# ----------------------------
# App
# ----------------------------
app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)
pool: asyncpg.Pool | None = None

# Ensure static directory exists to prevent crash on startup
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

_ip_hits: Dict[str, List[float]] = {}

def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    window_start = now - 60
    hits = _ip_hits.get(ip, [])
    hits = [t for t in hits if t >= window_start]
    if len(hits) >= RATE_LIMIT_PER_MINUTE:
        _ip_hits[ip] = hits
        return False
    hits.append(now)
    _ip_hits[ip] = hits
    return True

def require_auth(request: Request):
    if APP_API_KEY and request.headers.get("X-API-Key", "") != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.on_event("startup")
async def startup():
    global pool
    # Create connection pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)

@app.on_event("shutdown")
async def shutdown():
    if pool:
        await pool.close()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def home():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "CostSavvy API is running. (No static/index.html found)"}

# ----------------------------
# Helpers
# ----------------------------
def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"

def stream_llm_to_sse(system: str, user_content: str, out_text_parts: List[str]):
    try:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
            stream=True,
            timeout=30,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                out_text_parts.append(delta)
                yield sse({"type": "delta", "text": delta})
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield sse({"type": "error", "message": str(e)})

def _coerce_jsonb_to_dict(val) -> dict:
    if isinstance(val, str):
        try: return json.loads(val)
        except: return {}
    return val if isinstance(val, dict) else {}

async def get_or_create_session(conn: asyncpg.Connection, session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if not session_id: session_id = str(uuid.uuid4())
    row = await conn.fetchrow("SELECT session_state FROM public.chat_session WHERE id = $1", session_id)
    if row:
        await conn.execute("UPDATE public.chat_session SET last_seen = now() WHERE id = $1", session_id)
        return session_id, _coerce_jsonb_to_dict(row["session_state"])
    await conn.execute("INSERT INTO public.chat_session (id, session_state) VALUES ($1, $2::jsonb)", session_id, json.dumps({}))
    return session_id, {}

async def save_message(conn: asyncpg.Connection, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
    await conn.execute("INSERT INTO public.chat_message (session_id, role, content, metadata) VALUES ($1,$2,$3,$4::jsonb)", session_id, role, content, json.dumps(metadata or {}))

async def update_session_state(conn: asyncpg.Connection, session_id: str, state: Dict[str, Any]):
    await conn.execute("UPDATE public.chat_session SET session_state = $2::jsonb, last_seen = now() WHERE id = $1", session_id, json.dumps(state))

async def log_query(conn: asyncpg.Connection, session_id: str, question: str, intent: dict, used_radius: Optional[float], result_count: int, used_web: bool, answer_text: str):
    await conn.execute("INSERT INTO public.query_log (session_id, question, intent_json, used_radius_miles, result_count, used_web_search, answer_text) VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7)", session_id, question, json.dumps(intent), used_radius, result_count, used_web, answer_text)

# ----------------------------
# Intent
# ----------------------------
INTENT_RULES = """
Return ONLY JSON with:
mode: "general" | "price" | "hybrid" | "clarify"
zipcode: 5-digit ZIP or null
payer_like: string or null
payment_mode: "cash" | "insurance" | null
service_query: short phrase or null
"""

def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in ["zipcode", "payer_like", "payment_mode", "service_query", "code_type", "code", "refiner_id", "refiner_choice"]:
        v = intent.get(k)
        if v is not None and v != "": out[k] = v
    return out

def _normalize_payment_mode(merged: Dict[str, Any]) -> None:
    if merged.get("payment_mode") == "cash":
        merged["payer_like"] = None
    if merged.get("payment_mode") == "insurance":
        pass

async def extract_intent(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    msg = (message or "").strip()
    msg_l = msg.lower()
    st = state or {}
    awaiting = st.get("_awaiting")
    
    # 0) ZIP
    zip_match = re.search(r"\b(\d{5})\b", msg)
    if zip_match:
        return {"mode": "price", "zipcode": zip_match.group(1)}

    # Helpers
    cash_terms = {"cash", "self pay", "out of pocket"}
    ins_terms = {"insurance", "insured", "use insurance"}

    carrier_map = {
        "aetna": "Aetna", "cigna": "Cigna", "anthem": "Anthem", "blue cross": "Blue Cross Blue Shield",
        "bcbs": "Blue Cross Blue Shield", "united": "UnitedHealthcare", "uhc": "UnitedHealthcare",
        "humana": "Humana", "kaiser": "Kaiser Permanente", "medicare": "Medicare", "medicaid": "Medicaid"
    }

    def extract_carrier(m: str) -> Optional[str]:
        ml = m.lower()
        for k, v in carrier_map.items():
            if k in ml: return v
        clean = m
        for stop in ["i have", "i use", "use", "with", "insurance", "my", "have", "paying"]:
            clean = re.sub(r'\b' + re.escape(stop) + r'\b', '', clean, flags=re.IGNORECASE)
        tokens = re.findall(r"[a-zA-Z]+", clean)
        if len(tokens) == 1 and len(tokens[0]) >= 3: return tokens[0].title()
        if 2 <= len(tokens) <= 3: return " ".join([t.title() for t in tokens])
        return None

    # 1) Awaiting Payment
    if awaiting == "payment":
        if any(t in msg_l for t in cash_terms):
            return {"mode": "price", "payment_mode": "cash"}
        carrier = extract_carrier(msg)
        if any(t in msg_l for t in ins_terms) or carrier:
            return {"mode": "price", "payment_mode": "insurance", "payer_like": carrier}
        return {"mode": "clarify", "clarifying_question": "Cash or Insurance?"}

    # 2) Awaiting Payer
    if awaiting == "payer":
        carrier = extract_carrier(msg)
        if carrier: return {"mode": "price", "payer_like": carrier}
        # If they just say "I have insurance", we still need the name
        return {"mode": "clarify", "clarifying_question": "Which carrier?"}

    # 3) Fallback
    inferred = infer_service_query_from_message(msg)
    if inferred:
        return {"mode": "price", "service_query": inferred}

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0,
            messages=[{"role": "system", "content": INTENT_RULES}, {"role": "user", "content": json.dumps({"message": msg, "state": st})}],
            timeout=10
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except:
        return {"mode": "clarify"}

def infer_service_query_from_message(message: str) -> Optional[str]:
    msg = (message or "").lower()
    if "colonoscopy" in msg: return "colonoscopy"
    if "mammogram" in msg: return "mammogram"
    if "ultrasound" in msg: return "ultrasound"
    if "ct scan" in msg or "cat scan" in msg: return "ct scan"
    if "mri" in msg: return "mri"
    if "x-ray" in msg: return "x-ray"
    if "lab" in msg or "blood" in msg: return "lab test"
    if "visit" in msg: return "office visit"
    return None

def reset_gating_fields_for_new_price_question(message: str, merged: Dict[str, Any]) -> None:
    if not re.search(r"\b\d{5}\b", message): merged.pop("zipcode", None)
    if not any(t in message.lower() for t in ["cash", "insurance", "pay"]):
        merged.pop("payment_mode", None)
        merged.pop("payer_like", None)

# ----------------------------
# Pricing & Refiners
# ----------------------------
async def resolve_service_code(conn: asyncpg.Connection, merged: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if merged.get("code_type") and merged.get("code"): return merged["code_type"], merged["code"]
    q = (merged.get("service_query") or "").strip()
    if not q: return None
    rows = await conn.fetch("SELECT code_type, code FROM public.services WHERE service_description ILIKE '%' || $1 || '%' LIMIT 1", q)
    if rows: return rows[0]["code_type"], rows[0]["code"]
    rows = await conn.fetch("SELECT code_type, code FROM public.stg_hospital_rates WHERE service_description ILIKE '%' || $1 || '%' LIMIT 1", q)
    if rows: return rows[0]["code_type"], rows[0]["code"]
    return None

async def price_lookup_progressive(conn: asyncpg.Connection, zipcode: str, code_type: str, code: str, service_query: str, payer_like: str, payment_mode: str) -> Tuple[List[dict], int]:
    radius_attempts = [10, 25, 50, 100]
    z = await conn.fetchrow("SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1", zipcode)
    if not z: return [], 0
    zlat, zlon = float(z["latitude"]), float(z["longitude"])

    # Get service IDs
    s_ids = []
    if service_query:
        r = await conn.fetch("SELECT id FROM public.services WHERE service_description ILIKE '%' || $1 || '%' LIMIT 25", service_query)
        s_ids.extend([x["id"] for x in r])
    if code_type and code:
        r = await conn.fetchrow("SELECT id FROM public.services WHERE code_type=$1 AND code=$2", code_type, code)
        if r: s_ids.append(r["id"])
    
    if not s_ids: return [], 0

    for r in radius_attempts:
        if payment_mode == "cash":
            # CASH: Prefer standard_charge_cash (using production column name)
            q = """
                WITH user_zip AS (SELECT $1::float AS lat, $2::float AS lon)
                SELECT h.name, h.address, h.phone, h.zipcode,
                       (3959 * acos(cos(radians((SELECT lat FROM user_zip))) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians((SELECT lon FROM user_zip))) + sin(radians((SELECT lat FROM user_zip))) * sin(radians(h.latitude)))) AS distance_miles,
                       nr.standard_charge_cash, nr.estimated_amount, nr.standard_charge_gross
                FROM public.hospitals h
                LEFT JOIN LATERAL (
                    SELECT MIN(standard_charge_cash) as standard_charge_cash,
                           MIN(estimated_amount) as estimated_amount,
                           MIN(standard_charge_gross) as standard_charge_gross
                    FROM public.negotiated_rates WHERE hospital_id = h.id AND service_id = ANY($3::int[])
                ) nr ON TRUE
                WHERE h.latitude IS NOT NULL AND (3959 * acos(cos(radians((SELECT lat FROM user_zip))) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians((SELECT lon FROM user_zip))) + sin(radians((SELECT lat FROM user_zip))) * sin(radians(h.latitude)))) <= $4
                ORDER BY distance_miles LIMIT 10
            """
            rows = await conn.fetch(q, zlat, zlon, s_ids, r)
        else:
            # INSURANCE: Prefer negotiated_dollar (using production column name)
            payer_pat = f"%{payer_like}%" if payer_like else "%"
            q = """
                WITH user_zip AS (SELECT $1::float AS lat, $2::float AS lon)
                SELECT h.name, h.address, h.phone, h.zipcode,
                       (3959 * acos(cos(radians((SELECT lat FROM user_zip))) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians((SELECT lon FROM user_zip))) + sin(radians((SELECT lat FROM user_zip))) * sin(radians(h.latitude)))) AS distance_miles,
                       pick.negotiated_dollar, pick.estimated_amount, pick.standard_charge_cash
                FROM public.hospitals h
                LEFT JOIN LATERAL (
                    SELECT nr.negotiated_dollar, nr.estimated_amount, nr.standard_charge_cash
                    FROM public.negotiated_rates nr
                    JOIN public.insurance_plans ip ON ip.id = nr.plan_id
                    WHERE nr.hospital_id = h.id AND nr.service_id = ANY($3::int[]) AND ip.payer_name ILIKE $5
                    ORDER BY nr.negotiated_dollar NULLS LAST LIMIT 1
                ) pick ON TRUE
                WHERE h.latitude IS NOT NULL AND (3959 * acos(cos(radians((SELECT lat FROM user_zip))) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians((SELECT lon FROM user_zip))) + sin(radians((SELECT lat FROM user_zip))) * sin(radians(h.latitude)))) <= $4
                ORDER BY distance_miles LIMIT 10
            """
            rows = await conn.fetch(q, zlat, zlon, s_ids, r, payer_pat)

        if len(rows) >= 5:
            return [dict(x) for x in rows], r
    
    return [dict(x) for x in rows] if 'rows' in locals() and rows else [], 0

# ----------------------------
# Refiner Logic
# ----------------------------
async def get_refiners(conn: asyncpg.Connection) -> dict:
    try:
        rows = await conn.fetch("SELECT id, title, keywords, require_choice_before_pricing, preview_code_type, preview_code, question_text FROM public.service_refiner WHERE is_active = true")
        if not rows: return refiners_registry()
        ids = [r["id"] for r in rows]
        crows = await conn.fetch("SELECT refiner_id, choice_key, choice_label, code_type, code FROM public.service_refiner_choice WHERE is_active = true AND refiner_id = ANY($1::text[])", ids)
        choices = {}
        for c in crows: choices.setdefault(c["refiner_id"], []).append(dict(c))
        res = []
        for r in rows:
            res.append({
                "id": r["id"], "keywords": r["keywords"], "require_choice_before_pricing": r["require_choice_before_pricing"],
                "question": r["question_text"], "choices": choices.get(r["id"], [])
            })
        return {"refiners": res}
    except: return refiners_registry()

def match_refiner(service_query: str, doc: dict) -> Optional[dict]:
    q = (service_query or "").lower()
    for r in doc.get("refiners", []):
        if any(k in q for k in (r.get("keywords") or [])): return r
    return None

def apply_refiner_choice(message: str, merged: dict, refiner: Optional[dict]) -> dict:
    if not refiner: return merged
    key = (message or "").strip()
    for c in refiner.get("choices", []):
        if str(c.get("choice_key", c.get("key"))) == key:
            merged["code_type"] = c.get("code_type")
            merged["code"] = c.get("code")
            merged["refiner_choice"] = key
            return merged
    return merged

def get_refinement_prompt(refiner: dict) -> str:
    lines = [refiner.get("question", ""), ""]
    for c in refiner.get("choices", []):
        lines.append(f"{c.get('choice_key', c.get('key'))}) {c.get('choice_label', c.get('label'))}")
    return "\n".join(lines)

# ----------------------------
# Main Chat Stream (FIXED)
# ----------------------------
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    require_auth(request)
    if not rate_limit_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, detail="Limit exceeded")
    if not pool: raise HTTPException(500, detail="DB not ready")

    async def event_gen():
        try:
            async with pool.acquire() as conn:
                session_id, state = await get_or_create_session(conn, req.session_id)
                yield sse({"type": "session", "session_id": session_id})
                await save_message(conn, session_id, "user", req.message)

                intent = await extract_intent(req.message, state)
                merged = merge_state(state, intent)
                _normalize_payment_mode(merged)
                
                # Check for new query
                if RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION and intent.get("mode") in ["price", "hybrid"]:
                    if not merged.get("_awaiting"):
                        inferred = infer_service_query_from_message(req.message)
                        if inferred and any(k in req.message.lower() for k in INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS):
                            reset_gating_fields_for_new_price_question(req.message, merged)

                # Refiner
                refiners = await get_refiners(conn)
                refiner = match_refiner(merged.get("service_query") or "", refiners)
                merged = apply_refiner_choice(req.message, merged, refiner)

                mode = intent.get("mode") or "hybrid"

                if mode == "general":
                    parts = []
                    for chunk in stream_llm_to_sse("You are CostSavvy.", req.message, parts): yield chunk
                    await save_message(conn, session_id, "assistant", "".join(parts), {"mode": "general"})
                    await update_session_state(conn, session_id, merged)
                    yield sse({"type": "final"})
                    return

                if mode in ["price", "hybrid"]:
                    # Gate 1: ZIP
                    if not merged.get("zipcode"):
                        merged["_awaiting"] = "zip"
                        msg = "What’s your 5-digit ZIP code?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {})
                        await update_session_state(conn, session_id, merged)
                        yield sse({"type": "final"})
                        return
                    if merged.get("_awaiting") == "zip": merged.pop("_awaiting", None)

                    # Gate 2: Payment
                    if not merged.get("payment_mode"):
                        merged["_awaiting"] = "payment"
                        msg = "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {})
                        await update_session_state(conn, session_id, merged)
                        yield sse({"type": "final"})
                        return
                    if merged.get("_awaiting") == "payment": merged.pop("_awaiting", None)
                    
                    # Gate 3: Payer (if insurance)
                    if merged.get("payment_mode") == "insurance" and not (merged.get("payer_like") or "").strip():
                         merged["_awaiting"] = "payer"
                         msg = "Which insurance carrier should I match prices for (e.g., Aetna, UnitedHealthcare)?"
                         yield sse({"type": "delta", "text": msg})
                         await save_message(conn, session_id, "assistant", msg, {})
                         await update_session_state(conn, session_id, merged)
                         yield sse({"type": "final"})
                         return
                    if merged.get("_awaiting") == "payer": merged.pop("_awaiting", None)

                    # Refiner: Choice Required?
                    if refiner and refiner.get("require_choice_before_pricing") and not merged.get("refiner_choice"):
                        msg = get_refinement_prompt(refiner)
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {})
                        await update_session_state(conn, session_id, merged)
                        yield sse({"type": "final"})
                        return

                    # Resolve Code
                    if not (merged.get("code_type") and merged.get("code")):
                        resolved = await resolve_service_code(conn, merged)
                        if resolved: merged["code_type"], merged["code"] = resolved

                    # Missing Service?
                    if not (merged.get("service_query") or "").strip() and not (merged.get("code_type") and merged.get("code")):
                         msg = "What service are you pricing?"
                         yield sse({"type": "delta", "text": msg})
                         await save_message(conn, session_id, "assistant", msg, {})
                         await update_session_state(conn, session_id, merged)
                         yield sse({"type": "final"})
                         return

                    # Execute Pricing
                    results, used_radius = await price_lookup_progressive(
                        conn, merged["zipcode"], merged.get("code_type"), merged.get("code"),
                        merged.get("service_query") or "", merged.get("payer_like"), merged.get("payment_mode") or "cash"
                    )

                    # Build Response
                    lines = []
                    lines.append(f"Here are nearby options for **{merged.get('service_query') or 'service'}** ({merged.get('payment_mode')}):")
                    for i, r in enumerate(results[:5], 1):
                        p = r.get("negotiated_dollar") if merged.get("payment_mode") == "insurance" else r.get("standard_charge_cash")
                        if p is None: p = r.get("estimated_amount") or r.get("standard_charge_gross")
                        price_fmt = f"${p:,.0f}" if p else "Estimate not available"
                        lines.append(f"{i}) **{r.get('name')}** ({r.get('distance_miles', 0):.1f} mi)")
                        lines.append(f"   - Price: **{price_fmt}**")
                    
                    if not results:
                        lines.append("No facilities found nearby with pricing.")

                    txt = "\n".join(lines)
                    yield sse({"type": "delta", "text": txt})
                    yield sse({"type": "results", "results": results[:25]})
                    
                    await save_message(conn, session_id, "assistant", txt, {"mode": "price_result"})
                    await update_session_state(conn, session_id, merged)
                    yield sse({"type": "final"})
                    return
                
                # Fallback
                msg = "I’m not sure what you need. Can you rephrase?"
                yield sse({"type": "delta", "text": msg})
                await save_message(conn, session_id, "assistant", msg, {"mode": "fallback"})
                yield sse({"type": "final"})

        except Exception as e:
            logger.exception("Error")
            yield sse({"type": "error", "message": str(e)})
            yield sse({"type": "final"})

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

# ----------------------------
# Legacy Endpoint (Redirect)
# ----------------------------
@app.post("/chat")
async def chat_legacy(req: ChatRequest, request: Request):
    """
    Prevents 404/Not Found if frontend uses the old non-streaming endpoint.
    Redirects logic to chat_stream but unfortunately we can't 'redirect' a POST
    body easily in 307. Instead, we return a message or handle it.
    """
    return await chat_stream(req, request)