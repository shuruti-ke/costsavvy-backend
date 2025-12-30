# app/main.py
import os
import json
import uuid
import time
import openai
print("OpenAI SDK version:", openai.__version__)
from typing import Optional, Any, Dict, List, Tuple

import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI




# =========================
# Config (Render env vars)
# =========================
DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

APP_API_KEY = os.getenv("APP_API_KEY", "").strip()          # if empty, auth disabled
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

MIN_DB_RESULTS_BEFORE_WEB = int(os.getenv("MIN_DB_RESULTS_BEFORE_WEB", "3"))

# =========================
# App
# =========================
app = FastAPI()
client = OpenAI()  # uses OPENAI_API_KEY in env
pool: asyncpg.Pool | None = None

# Serve UI + static assets
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
    return f"data: {json.dumps(obj)}\n\n"

def _event_type(ev: Any) -> Optional[str]:
    if hasattr(ev, "type"):
        return getattr(ev, "type")
    if isinstance(ev, dict):
        return ev.get("type")
    return None

def _event_delta(ev: Any) -> str:
    if hasattr(ev, "delta"):
        return getattr(ev, "delta") or ""
    if isinstance(ev, dict):
        return ev.get("delta", "") or ""
    return ""

def _event_error(ev: Any) -> str:
    if hasattr(ev, "error"):
        return str(getattr(ev, "error"))
    if isinstance(ev, dict):
        return str(ev.get("error", "Unknown error"))
    return "Unknown error"

def stream_llm_to_sse(system: str, user_content: str, out_text_parts: List[str]):
    """
    Streams OpenAI Responses output_text deltas as SSE events.
    Appends delta text to out_text_parts for logging.
    """
    stream = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        stream=True
    )

    for ev in stream:
        et = _event_type(ev)

        if et == "response.output_text.delta":
            delta = _event_delta(ev)
            if delta:
                out_text_parts.append(delta)
                yield sse({"type": "delta", "text": delta})

        elif et == "error":
            yield sse({"type": "error", "message": _event_error(ev)})
            return

# =========================
# DB helpers: sessions/messages/logs
# =========================
def _coerce_jsonb_to_dict(val) -> dict:
    """
    asyncpg can return json/jsonb as dict, or sometimes as str depending on
    driver/config. This prevents ValueError from dict(<string>).
    """
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
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
Return ONLY JSON with:
mode: "general" | "price" | "hybrid" | "clarify"
zipcode: 5-digit ZIP or null
radius_miles: number or null
payer_like: string like "%Aetna%" or null
plan_like: string like "%PPO%" or null
service_query: short phrase like "chest x-ray" or null
code_type: string or null (usually "CPT")
code: string or null (like "71046")
clarifying_question: string or null (ask ONE question only if needed)
cash_only: boolean (true if user wants cash/self-pay prices)
"""

def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in ["zipcode", "radius_miles", "payer_like", "plan_like", "service_query", "code_type", "code", "cash_only"]:
        v = intent.get(k)
        if v is not None and v != "":
            out[k] = v
    return out

async def extract_intent(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "You extract intent for healthcare Q&A and price lookup. Be conservative."},
            {"role": "system", "content": INTENT_RULES},
            {"role": "user", "content": json.dumps({"message": message, "session_state": state})}
        ]
    )
    try:
        return json.loads(resp.output_text)
    except Exception:
        return {"mode": "clarify", "clarifying_question": "What 5-digit ZIP code should I search near?"}

# =========================
# DB: resolve code + query v3 function
# =========================
async def resolve_service_code(conn: asyncpg.Connection, merged: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if merged.get("code_type") and merged.get("code"):
        return merged["code_type"], merged["code"]

    q = (merged.get("service_query") or "").strip()
    if not q:
        return None

    rows = await conn.fetch(
        """
        SELECT code_type, code
        FROM public.services
        WHERE (cpt_explanation ILIKE '%' || $1 || '%'
            OR service_description ILIKE '%' || $1 || '%')
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
        FROM public.get_prices_by_zip_radius_v3(
          $1, $2, $3, $4, $5,
          ARRAY[10,25,50], 10, 25
        );
        """,
        zipcode, code_type, code, payer_like, plan_like
    )
    return [dict(r) for r in rows]

# =========================
# Web search fallback (DB insufficient only)
# =========================
def web_search_fallback_text(question: str) -> str:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": question}],
        tools=[{"type": "web_search"}],
    )
    return resp.output_text

# =========================
# Main streaming endpoint
# =========================
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    require_auth(request)
    ip = request.client.host if request.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(429, detail="Rate limit exceeded. Please slow down.")
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

                # cash-only: clear insurance filters
                if merged.get("cash_only") is True:
                    merged["payer_like"] = None
                    merged["plan_like"] = None

                # --------------------
                # GENERAL mode
                # --------------------
                if mode == "general":
                    system = (
                        "You are CostSavvy.health. Answer general healthcare questions clearly in plain language. "
                        "Avoid medical advice. End with a brief educational disclaimer."
                    )

                    parts: List[str] = []
                    for chunk in stream_llm_to_sse(system, req.message, parts):
                        yield chunk

                    full_answer = "".join(parts).strip() or "I couldn’t generate a response. Please try again."
                    await save_message(conn, session_id, "assistant", full_answer, {"mode": "general", "intent": intent})
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, None, 0, False, full_answer)

                    yield sse({"type": "final", "used_web_search": False})
                    return

                # --------------------
                # PRICE / HYBRID mode
                # --------------------
                results: List[Dict[str, Any]] = []
                used_web = False
                web_notes = None

                zipcode = merged.get("zipcode")
                payer_like = merged.get("payer_like")
                plan_like = merged.get("plan_like")
                code_type = merged.get("code_type")
                code = merged.get("code")

                if mode in ["price", "hybrid"]:
                    # resolve code if missing
                    if not (code_type and code):
                        resolved = await resolve_service_code(conn, merged)
                        if resolved:
                            code_type, code = resolved
                            merged["code_type"] = code_type
                            merged["code"] = code

                    # if missing required info, ask one clarifying question
                    if not zipcode or not (code_type and code):
                        cq = intent.get("clarifying_question")
                        if not cq:
                            missing = []
                            if not zipcode:
                                missing.append("your 5-digit ZIP code")
                            if not (code_type and code):
                                missing.append("the procedure (or CPT code)")
                            cq = "What is " + " and ".join(missing) + "?"
                        yield sse({"type": "delta", "text": cq})

                        await save_message(conn, session_id, "assistant", cq, {"mode": "clarify", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, cq)

                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # DB lookup
                    results = await price_lookup_v3(conn, zipcode, code_type, code, payer_like, plan_like)

                    # Web fallback only if DB insufficient
                    if len(results) < MIN_DB_RESULTS_BEFORE_WEB:
                        used_web = True
                        web_notes = web_search_fallback_text(req.message)

                    system = (
                        "You are CostSavvy.health. Be honest about sources.\n"
                        "- If results are present, say they come from our hospital price database.\n"
                        "- If web_notes are present, say you used online search because DB was insufficient.\n"
                        "- If the user asked general info, include a short explanation too.\n"
                        "- Summarize top 5 cheapest with distance and best_price.\n"
                        "- End with a brief disclaimer: prices vary; confirm with hospital/insurer.\n"
                    )

                    payload = {
                        "question": req.message,
                        "state": merged,
                        "top_results": results[:10],
                        "web_notes": web_notes
                    }

                    parts: List[str] = []
                    for chunk in stream_llm_to_sse(system, json.dumps(payload), parts):
                        yield chunk

                    full_answer = "".join(parts).strip() or "I couldn’t generate a response. Please try again."
                    await save_message(
                        conn, session_id, "assistant", full_answer,
                        {"mode": mode, "intent": intent, "result_count": len(results), "used_web_search": used_web}
                    )
                    await update_session_state(conn, session_id, merged)

                    used_radius = results[0].get("used_radius_miles") if results else None
                    await log_query(conn, session_id, req.message, intent, used_radius, len(results), used_web, full_answer)

                    yield sse({"type": "final", "used_web_search": used_web})
                    return

                # If we somehow land here, ask to rephrase
                msg = "I’m not sure what you need. Can you rephrase your question?"
                yield sse({"type": "delta", "text": msg})
                await save_message(conn, session_id, "assistant", msg, {"mode": "fallback", "intent": intent})
                await update_session_state(conn, session_id, merged)
                await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                yield sse({"type": "final", "used_web_search": False})

        except Exception as e:
            # Never fail silently
            yield sse({"type": "error", "message": f"{type(e).__name__}: {str(e)}"})
            yield sse({"type": "final", "used_web_search": False})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

# Optional: non-streaming endpoint (disabled)
@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    require_auth(request)
    raise HTTPException(410, detail="Use /chat_stream for streaming UI.")
