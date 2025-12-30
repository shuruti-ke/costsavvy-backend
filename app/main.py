# app/main.py
import os
import json
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

# Optional: enable debug logging during dev
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("costsavvy")

# =========================
# Config (Render env vars)
# =========================
DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # Enforce presence
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # gpt-4.1-mini is invalid

APP_API_KEY = os.getenv("APP_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MIN_DB_RESULTS_BEFORE_WEB = int(os.getenv("MIN_DB_RESULTS_BEFORE_WEB", "3"))

# =========================
# App
# =========================
app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)
pool: asyncpg.Pool | None = None

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# Models
# =========================
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

# =========================
# Rate limit (in-memory)
# =========================
_ip_hits: Dict[str, List[float]] = {}

def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    window_start = now - 60
    hits = _ip_hits.get(ip, [])[::]  # copy
    hits = [t for t in hits if t >= window_start]
    if len(hits) >= RATE_LIMIT_PER_MINUTE:
        _ip_hits[ip] = hits
        return False
    hits.append(now)
    _ip_hits[ip] = hits
    return True

def require_auth(request: Request):
    if not APP_API_KEY:
        return
    if request.headers.get("X-API-Key", "") != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# =========================
# Startup / Shutdown
# =========================
@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    logger.info("✅ DB pool ready")

@app.on_event("shutdown")
async def shutdown():
    if pool:
        await pool.close()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def home():
    return FileResponse("static/index.html")

# =========================
# SSE helpers
# =========================
def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"

def stream_llm_to_sse(system: str, user_content: str, out_text_parts: List[str]):
    """
    Streams OpenAI chat completions as SSE events.
    Appends delta text to out_text_parts for logging.
    """
    try:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            timeout=30,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                out_text_parts.append(delta)
                yield sse({"type": "delta", "text": delta})

    except OpenAIError as e:
        logger.error(f"OpenAI error: {e}")
        yield sse({"type": "error", "message": "LLM service unavailable. Please try again."})
        return
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield sse({"type": "error", "message": "An unexpected error occurred."})
        return

# =========================
# DB helpers
# =========================
def _coerce_jsonb_to_dict(val) -> dict:
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except:
            return {}
    return {}

async def get_or_create_session(conn: asyncpg.Connection, session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if not session_id:
        session_id = str(uuid.uuid4())

    row = await conn.fetchrow("SELECT session_state FROM public.chat_session WHERE id = $1", session_id)
    if row:
        await conn.execute("UPDATE public.chat_session SET last_seen = now() WHERE id = $1", session_id)
        return session_id, _coerce_jsonb_to_dict(row["session_state"])

    await conn.execute(
        "INSERT INTO public.chat_session (id, session_state) VALUES ($1, $2::jsonb)",
        session_id, json.dumps({})
    )
    return session_id, {}

async def save_message(conn: asyncpg.Connection, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
    await conn.execute(
        "INSERT INTO public.chat_message (session_id, role, content, metadata) VALUES ($1,$2,$3,$4::jsonb)",
        session_id, role, content, json.dumps(metadata or {})
    )

async def update_session_state(conn: asyncpg.Connection, session_id: str, state: Dict[str, Any]):
    await conn.execute(
        "UPDATE public.chat_session SET session_state = $2::jsonb, last_seen = now() WHERE id = $1",
        session_id, json.dumps(state)
    )

async def log_query(conn: asyncpg.Connection, session_id: str, question: str, intent: dict,
                    used_radius: Optional[float], result_count: int, used_web: bool, answer_text: str):
    await conn.execute(
        """
        INSERT INTO public.query_log
          (session_id, question, intent_json, used_radius_miles, result_count, used_web_search, answer_text)
        VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7)
        """,
        session_id, question, json.dumps(intent), used_radius, result_count, used_web, answer_text
    )

# =========================
# Intent extraction
# =========================
INTENT_RULES = """
Return ONLY valid JSON (no markdown). Keys:
mode: "general"|"price"|"hybrid"|"clarify"
zipcode: 5-digit ZIP string or null
radius_miles: number or null
payer_like: string (e.g., "Aetna") or null
plan_like: string (e.g., "PPO") or null
service_query: phrase like "colonoscopy" or null
code_type: "CPT"|"HCPCS"|null
code: string like "45378" or null
clarifying_question: string or null
cash_only: true|false
"""

def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in ["zipcode", "radius_miles", "payer_like", "plan_like", "service_query", "code_type", "code", "cash_only"]:
        v = intent.get(k)
        if v is not None and v != "":
            out[k] = v
    return out

async def extract_intent(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You extract intent for healthcare Q&A and price lookup. Be conservative."},
                {"role": "system", "content": INTENT_RULES},
                {"role": "user", "content": json.dumps({"message": message, "session_state": state})},
            ],
            temperature=0.0,
            timeout=10,
        )
        content = resp.choices[0].message.content or ""
        return json.loads(content)
    except (json.JSONDecodeError, OpenAIError, Exception) as e:
        logger.warning(f"Intent extraction failed: {e}")
        return {
            "mode": "clarify",
            "clarifying_question": "What 5-digit ZIP code should I search near?"
        }

# =========================
# DB: resolve code + price lookup
# =========================
async def resolve_service_code(conn: asyncpg.Connection, merged: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if merged.get("code_type") and merged.get("code"):
        return merged["code_type"], merged["code"]

    q = (merged.get("service_query") or "").strip().lower()
    if not q:
        return None

    rows = await conn.fetch(
        """
        SELECT code_type, code
        FROM public.services
        WHERE 
            lower(cpt_explanation) LIKE '%' || $1 || '%'
            OR lower(service_description) LIKE '%' || $1 || '%'
        ORDER BY code_type, code
        LIMIT 5
        """,
        q
    )
    if not rows:
        return None
    return rows[0]["code_type"], rows[0]["code"]

async def price_lookup_v3(conn: asyncpg.Connection, zipcode: str, code_type: str, code: str,
                          payer_like: Optional[str], plan_like: Optional[str]) -> List[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT *
        FROM public.get_prices_by_zip_radius_v2(
          $1, $2, $3, $4, $5,
          ARRAY[10,25,50], 10, 25
        );
        """,
        zipcode, code_type, code, payer_like, plan_like
    )
    return [dict(r) for r in rows]

# =========================
# Web search fallback (LLM-based estimate)
# =========================
def web_search_fallback_text(question: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are CostSavvy.health. Provide realistic U.S. cost estimates for procedures. "
                        "Be clear about uncertainty. Do NOT invent exact prices. Mention typical ranges, cash vs insured."
                    )
                },
                {"role": "user", "content": f"Estimate costs for: {question}"}
            ],
            timeout=15,
        )
        return resp.choices[0].message.content or "No estimate available."
    except Exception as e:
        logger.error(f"Web fallback failed: {e}")
        return "I couldn’t find sufficient pricing data. Try specifying ZIP code and insurance."

# =========================
# Main streaming endpoint
# =========================
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    require_auth(request)
    ip = request.client.host if request.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(429, detail="Rate limit exceeded.")
    if not pool:
        raise HTTPException(500, detail="Database not ready")

    async def event_gen():
        try:
            async with pool.acquire() as conn:
                session_id, state = await get_or_create_session(conn, req.session_id)
                yield sse({"type": "session", "session_id": session_id})
                await save_message(conn, session_id, "user", req.message)

                intent = await extract_intent(req.message, state)
                merged = merge_state(state, intent)
                mode = intent.get("mode") or "hybrid"

                if merged.get("cash_only") is True:
                    merged["payer_like"] = None
                    merged["plan_like"] = None

                # --------------------
                # GENERAL mode
                # --------------------
                if mode == "general":
                    system = (
                        "You are CostSavvy.health. Answer general healthcare questions clearly in plain language. "
                        "Do not give medical advice. End with: 'This is general info, not a quote. Confirm with your provider.'"
                    )
                    parts: List[str] = []
                    for chunk in stream_llm_to_sse(system, req.message, parts):
                        yield chunk

                    full_answer = "".join(parts).strip() or "I couldn’t generate a response."
                    await save_message(conn, session_id, "assistant", full_answer, {"mode": "general", "intent": intent})
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, None, 0, False, full_answer)
                    yield sse({"type": "final", "used_web_search": False})
                    return

                # --------------------
                # PRICE / HYBRID mode
                # --------------------
                if mode in ["price", "hybrid"]:
                    code_type, code = merged.get("code_type"), merged.get("code")
                    if not (code_type and code):
                        resolved = await resolve_service_code(conn, merged)
                        if resolved:
                            code_type, code = resolved
                            merged["code_type"] = code_type
                            merged["code"] = code

                    zipcode = merged.get("zipcode")
                    if not zipcode or not (code_type and code):
                        cq = intent.get("clarifying_question") or "Could you share your 5-digit ZIP code and procedure name (e.g., 'colonoscopy')?"
                        yield sse({"type": "delta", "text": cq})
                        await save_message(conn, session_id, "assistant", cq, {"mode": "clarify", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, cq)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    results = await price_lookup_v3(conn, zipcode, code_type, code, merged.get("payer_like"), merged.get("plan_like"))
                    used_web = len(results) < MIN_DB_RESULTS_BEFORE_WEB
                    web_notes = web_search_fallback_text(req.message) if used_web else None

                    system = (
                        "You are CostSavvy.health. Be transparent and helpful.\n"
                        "- If DB results: summarize top 5 cheapest (hospital, distance, price).\n"
                        "- If web_notes: say 'Our database had limited results, so I supplemented with general estimates.'\n"
                        "- Always add: 'Prices vary. Confirm with the facility and your insurer.'"
                    )

                    payload = {
                        "User question": req.message,
                        "ZIP": zipcode,
                        "Procedure": f"{code_type} {code}",
                        "DB results": [
                            {k: v for k, v in r.items() if k in ["hospital_name", "city", "state", "distance_miles", "best_price"]}
                            for r in results[:5]
                        ],
                        "Web supplement": web_notes if used_web else None,
                    }

                    parts: List[str] = []
                    for chunk in stream_llm_to_sse(system, json.dumps(payload, indent=2), parts):
                        yield chunk

                    full_answer = "".join(parts).strip() or "I couldn’t generate a response."
                    await save_message(conn, session_id, "assistant", full_answer, {
                        "mode": mode, "intent": intent, "result_count": len(results), "used_web_search": used_web
                    })
                    await update_session_state(conn, session_id, merged)
                    used_radius = results[0].get("used_radius_miles") if results else None
                    await log_query(conn, session_id, req.message, intent, used_radius, len(results), used_web, full_answer)
                    yield sse({"type": "final", "used_web_search": used_web})
                    return

                # Fallback
                msg = "I’m not sure how to help. Could you rephrase or ask about a procedure, cost, or insurance?"
                yield sse({"type": "delta", "text": msg})
                await save_message(conn, session_id, "assistant", msg, {"mode": "fallback", "intent": intent})
                await update_session_state(conn, session_id, merged)
                await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                yield sse({"type": "final", "used_web_search": False})

        except Exception as e:
            logger.exception("Unhandled error in chat_stream")
            yield sse({"type": "error", "message": f"Server error: {type(e).__name__}"})
            yield sse({"type": "final", "used_web_search": False})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

@app.post("/chat")
async def chat(_req: ChatRequest, _request: Request):
    raise HTTPException(410, detail="Use /chat_stream for streaming UI.")