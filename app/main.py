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

# --------------------
# Config
# --------------------
DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # Enforce presence
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # gpt-4.1-mini doesn't exist — use gpt-4o-mini

APP_API_KEY = os.getenv("APP_API_KEY", "").strip()  # if empty, auth disabled
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# DB sufficiency threshold for web fallback
MIN_DB_RESULTS_BEFORE_WEB = 3

# --------------------
# Logging (optional — comment out in prod if too verbose)
# --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("costsavvy")

# --------------------
# App + clients
# --------------------
app = FastAPI(title="CostSavvy.health API", version="0.2-beta")
client = OpenAI(api_key=OPENAI_API_KEY)
pool: asyncpg.Pool | None = None

# Serve UI + static
app.mount("/static", StaticFiles(directory="static"), name="static")


# --------------------
# Simple in-memory rate limiter (per-IP)
# Note: good for 1 instance. If you scale horizontally, move to Redis.
# --------------------
_ip_hits: Dict[str, List[float]] = {}  # ip -> timestamps


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
    if not APP_API_KEY:
        return  # auth disabled
    key = request.headers.get("X-API-Key", "")
    if key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# --------------------
# Models
# --------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


# --------------------
# Startup / Shutdown
# --------------------
@app.on_event("startup")
async def startup():
    global pool
    try:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        logger.info("✅ Database pool created")
    except Exception as e:
        logger.error(f"❌ Failed to connect to DB: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    if pool:
        await pool.close()
        logger.info("CloseOperation: DB pool closed")


@app.get("/health")
async def health():
    return {"status": "ok", "model": OPENAI_MODEL}


@app.get("/")
async def home():
    return FileResponse("static/index.html")


# --------------------
# DB helpers: sessions/messages/logs
# --------------------
async def get_or_create_session(conn: asyncpg.Connection, session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if not session_id:
        session_id = str(uuid.uuid4())

    row = await conn.fetchrow("SELECT session_state FROM public.chat_session WHERE id = $1", session_id)
    if row:
        await conn.execute("UPDATE public.chat_session SET last_seen = now() WHERE id = $1", session_id)
        state = dict(row["session_state"] or {})
        logger.debug(f"🔁 Loaded session {session_id}: {list(state.keys())}")
        return session_id, state

    await conn.execute(
        "INSERT INTO public.chat_session (id, session_state) VALUES ($1, $2::jsonb)",
        session_id, json.dumps({})
    )
    logger.info(f"🆕 Created session {session_id}")
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
        INSERT INTO public.query_log (session_id, question, intent_json, used_radius_miles, result_count, used_web_search, answer_text)
        VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7)
        """,
        session_id, question, json.dumps(intent), used_radius, result_count, used_web, answer_text
    )


# --------------------
# LLM: intent extraction (multi-turn aware)
# --------------------
INTENT_RULES = """Return ONLY valid JSON. No markdown. Keys:
- mode: "general" | "price" | "hybrid" | "clarify"
- zipcode: 5-digit ZIP string or null
- radius_miles: number or null
- payer_like: string (e.g., "Aetna") or null — no wildcards
- plan_like: string (e.g., "PPO") or null
- service_query: short phrase (e.g., "colonoscopy") or null
- code_type: "CPT" | "HCPCS" | null
- code: e.g., "45378" or null
- clarifying_question: string or null
- cash_only: true | false
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
                {"role": "user", "content": f"Current state: {json.dumps(state)}\nUser message: {message}"},
            ],
            temperature=0.0,
            timeout=10,
        )
        content = resp.choices[0].message.content or ""
        logger.debug(f"🧠 Raw intent: {content[:200]}...")
        parsed = json.loads(content)
        logger.info(f"✅ Intent: mode={parsed.get('mode')}, zip={parsed.get('zipcode')}, svc={parsed.get('service_query')}")
        return parsed
    except (json.JSONDecodeError, OpenAIError, Exception) as e:
        logger.warning(f"⚠️ Intent extraction failed: {e}")
        return {
            "mode": "clarify",
            "clarifying_question": "What 5-digit ZIP code should I search near?"
        }


# --------------------
# DB: resolve code + price query
# --------------------
async def resolve_service_code(conn: asyncpg.Connection, intent: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if intent.get("code_type") and intent.get("code"):
        return intent["code_type"], intent["code"]

    q = (intent.get("service_query") or "").strip().lower()
    if not q:
        return None

    # Try fuzzy match in DB
    rows = await conn.fetch(
        """
        SELECT code_type, code
        FROM public.services
        WHERE 
            lower(cpt_explanation) LIKE '%' || $1 || '%'
            OR lower(service_description) LIKE '%' || $1 || '%'
        ORDER BY 
            CASE WHEN lower(code) = $1 THEN 0 ELSE 1 END,  -- exact code match first
            code_type, code
        LIMIT 5
        """,
        q
    )
    if not rows:
        return None
    top = rows[0]
    logger.info(f"🔍 Resolved {q} → {top['code_type']} {top['code']}")
    return top["code_type"], top["code"]


async def price_lookup_v3(
    conn: asyncpg.Connection,
    zipcode: str,
    code_type: str,
    code: str,
    payer_like: Optional[str],
    plan_like: Optional[str]
) -> List[Dict[str, Any]]:
    try:
        rows = await conn.fetch(
            """
            SELECT *
            FROM public.get_prices_by_zip_radius_v3(
              $1, $2, $3, $4, $5,
              ARRAY[10,25,50], 10, 25
            );
            """,
            zipcode, code_type, code, payer_like, plan_like
        )
        logger.info(f"💰 DB returned {len(rows)} price rows")
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"❌ DB query failed: {e}")
        return []


# --------------------
# Web search fallback (LLM-based estimate only)
# --------------------
def web_search_fallback_text(question: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are CostSavvy.health. Provide realistic U.S. cost estimates for healthcare procedures. "
                    "Be clear about uncertainty. Do NOT invent exact prices or hospitals. "
                    "Mention typical ranges, cash vs insured, and factors affecting price."
                )},
                {"role": "user", "content": f"Estimate costs for: {question}"}
            ],
            timeout=15,
        )
        return resp.choices[0].message.content or "No estimate available."
    except Exception as e:
        logger.error(f"🌐 Web fallback failed: {e}")
        return "I couldn’t find sufficient pricing data. Try specifying ZIP code and insurance."


# --------------------
# Streaming helpers
# --------------------
def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def stream_llm(system: str, user_content: str):
    """
    Stream chat.completions deltas as SSE 'delta' events.
    Yields SSE strings.
    Returns full text (not used in streaming response, but logged).
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

        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield sse({"type": "delta", "text": delta})

        return full_text

    except Exception as e:
        error_msg = f"LLM streaming error: {e}"
        logger.error(error_msg)
        yield sse({"type": "error", "message": "Sorry, I encountered an error generating a response."})
        return ""


# --------------------
# STREAMING chat endpoint
# --------------------
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    require_auth(request)
    ip = request.client.host if request.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(429, detail="Rate limit exceeded. Please wait a moment.")

    if not pool:
        raise HTTPException(500, detail="Database pool not ready")

    async def event_gen():
        async with pool.acquire() as conn:
            session_id, state = await get_or_create_session(conn, req.session_id)
            yield sse({"type": "session", "session_id": session_id})
            await save_message(conn, session_id, "user", req.message)

            # Extract intent
            intent = await extract_intent(req.message, state)
            merged = merge_state(state, intent)
            mode = intent.get("mode", "hybrid")

            # Cash-only → clear payer/plan
            if merged.get("cash_only") is True:
                merged["payer_like"] = None
                merged["plan_like"] = None

            # ✅ GENERAL QUESTIONS
            if mode == "general":
                system_prompt = (
                    "You are CostSavvy.health — a helpful, transparent assistant for U.S. healthcare costs and info.\n"
                    "- Use plain, empathetic language.\n"
                    "- Never give medical advice.\n"
                    "- Cite uncertainty; add disclaimer: \"This is general info, not a quote. Confirm with provider.\"\n"
                )
                full_answer = ""
                for event in stream_llm(system_prompt, req.message):
                    if isinstance(event, str) and event.startswith("data: "):
                        yield event
                        # Extract text for logging
                        try:
                            payload = json.loads(event[6:])  # skip "data: "
                            if payload.get("type") == "delta":
                                full_answer += payload.get("text", "")
                        except:
                            pass

                await save_message(conn, session_id, "assistant", full_answer, {"intent": intent, "mode": "general"})
                await update_session_state(conn, session_id, merged)
                await log_query(conn, session_id, req.message, intent, None, 0, False, full_answer)
                yield sse({"type": "final", "used_web_search": False})
                return

            # ✅ PRICE / HYBRID MODE
            if mode in ["price", "hybrid"]:
                # Resolve service code
                code_type, code = merged.get("code_type"), merged.get("code")
                if not (code_type and code):
                    resolved = await resolve_service_code(conn, merged)
                    if resolved:
                        code_type, code = resolved
                        merged["code_type"] = code_type
                        merged["code"] = code

                zipcode = merged.get("zipcode")
                payer_like = merged.get("payer_like")
                plan_like = merged.get("plan_like")

                # Clarify if missing critical info
                if not zipcode or not (code_type and code):
                    cq = intent.get("clarifying_question") or "Could you share your 5-digit ZIP code and the procedure name (e.g., 'colonoscopy')?"
                    yield sse({"type": "delta", "text": cq})
                    await save_message(conn, session_id, "assistant", cq, {"intent": intent, "mode": "clarify"})
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, None, 0, False, cq)
                    yield sse({"type": "final", "used_web_search": False})
                    return

                # Query DB
                results = await price_lookup_v3(conn, zipcode, code_type, code, payer_like, plan_like)
                used_web = False
                web_notes = None

                if len(results) < MIN_DB_RESULTS_BEFORE_WEB:
                    used_web = True
                    web_notes = web_search_fallback_text(req.message)

                # Build answer
                system = (
                    "You are CostSavvy.health. Be transparent and helpful.\n"
                    "- If DB results exist: summarize top 5 cheapest (show distance, price, hospital).\n"
                    "- If web_notes exist: say 'Our database had limited results, so I supplemented with general estimates.'\n"
                    "- Always add: 'Prices vary. Confirm with the facility and your insurer.'\n"
                    "- Keep it concise and scannable."
                )

                payload = {
                    "User question": req.message,
                    "ZIP": zipcode,
                    "Procedure": f"{code_type} {code}",
                    "Insurance filter": f"{payer_like} {plan_like}".strip() or "Any/Unspecified",
                    "DB results count": len(results),
                    "DB results": [
                        {
                            "hospital": r.get("hospital_name"),
                            "city": r.get("city"),
                            "state": r.get("state"),
                            "distance_mi": round(r.get("distance_miles", 999), 1),
                            "price": f"${r.get('best_price'):,.0f}" if r.get("best_price") else "N/A",
                        }
                        for r in results[:5]
                    ],
                    "Web supplement": web_notes if used_web else None,
                }

                full_answer = ""
                for event in stream_llm(system, json.dumps(payload, indent=2)):
                    if isinstance(event, str) and event.startswith("data: "):
                        yield event
                        try:
                            payload_inner = json.loads(event[6:])
                            if payload_inner.get("type") == "delta":
                                full_answer += payload_inner.get("text", "")
                        except:
                            pass

                await save_message(conn, session_id, "assistant", full_answer, {
                    "intent": intent,
                    "result_count": len(results),
                    "used_web_search": used_web
                })
                await update_session_state(conn, session_id, merged)
                used_radius = results[0].get("used_radius_miles") if results else None
                await log_query(conn, session_id, req.message, intent, used_radius, len(results), used_web, full_answer)

                yield sse({"type": "final", "used_web_search": used_web})
                return

            # Fallback
            msg = "I’m not sure how to help with that. Could you rephrase or ask about a medical procedure, cost, or insurance?"
            yield sse({"type": "delta", "text": msg})
            await save_message(conn, session_id, "assistant", msg, {"intent": intent})
            yield sse({"type": "final", "used_web_search": False})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# Keep non-streaming endpoint as deprecated
@app.post("/chat")
async def chat(_req: ChatRequest, _request: Request):
    raise HTTPException(410, detail="This endpoint is deprecated. Use /chat_stream instead.")