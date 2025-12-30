import os
import json
import uuid
import time
from typing import Optional, Any, Dict, List, Tuple

import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

# --------------------
# Config
# --------------------
DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

APP_API_KEY = os.getenv("APP_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# DB sufficiency thresholds
MIN_DB_RESULTS_BEFORE_WEB = 3

# --------------------
# App + clients
# --------------------
app = FastAPI()
client = OpenAI()  # uses OPENAI_API_KEY
pool: asyncpg.Pool | None = None

# Serve UI
app.mount("/static", StaticFiles(directory="static"), name="static")


# --------------------
# Simple in-memory rate limiter (per-IP)
# NOTE: Fine for 1 Render instance. If you scale to multiple instances,
# move this to Redis.
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
        return  # auth disabled if not set

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


# --------------------
# DB helpers: sessions/messages/logs
# --------------------
async def get_or_create_session(conn: asyncpg.Connection, session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if not session_id:
        session_id = str(uuid.uuid4())

    row = await conn.fetchrow("SELECT session_state FROM chat_session WHERE id = $1", session_id)
    if row:
        await conn.execute("UPDATE chat_session SET last_seen = now() WHERE id = $1", session_id)
        return session_id, dict(row["session_state"] or {})

    await conn.execute(
        "INSERT INTO chat_session (id, session_state) VALUES ($1, $2::jsonb)",
        session_id, json.dumps({})
    )
    return session_id, {}


async def save_message(conn: asyncpg.Connection, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
    await conn.execute(
        "INSERT INTO chat_message (session_id, role, content, metadata) VALUES ($1,$2,$3,$4::jsonb)",
        session_id, role, content, json.dumps(metadata or {})
    )


async def update_session_state(conn: asyncpg.Connection, session_id: str, state: Dict[str, Any]):
    await conn.execute(
        "UPDATE chat_session SET session_state = $2::jsonb, last_seen = now() WHERE id = $1",
        session_id, json.dumps(state)
    )


async def log_query(conn: asyncpg.Connection, session_id: str, question: str, intent: dict,
                    used_radius: Optional[float], result_count: int, used_web: bool, answer_text: str):
    await conn.execute(
        """
        INSERT INTO query_log (session_id, question, intent_json, used_radius_miles, result_count, used_web_search, answer_text)
        VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7)
        """,
        session_id, question, json.dumps(intent), used_radius, result_count, used_web, answer_text
    )


# --------------------
# LLM: intent extraction (multi-turn aware)
# --------------------
INTENT_RULES = """
Return ONLY JSON with:
mode: "general" | "price" | "hybrid" | "clarify"
zipcode: 5-digit ZIP or null
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
    for k in ["zipcode","payer_like","plan_like","service_query","code_type","code","cash_only"]:
        v = intent.get(k)
        if v is not None and v != "":
            out[k] = v
    return out


async def extract_intent(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "You extract intent for healthcare questions and price lookup. Be conservative."},
            {"role": "system", "content": INTENT_RULES},
            {"role": "user", "content": json.dumps({"message": message, "session_state": state})}
        ]
    )
    try:
        return json.loads(resp.output_text)
    except Exception:
        return {"mode": "clarify", "clarifying_question": "What 5-digit ZIP code should I search near?"}


# --------------------
# DB: resolve code + price query
# --------------------
async def resolve_service_code(conn: asyncpg.Connection, intent: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if intent.get("code_type") and intent.get("code"):
        return intent["code_type"], intent["code"]

    q = (intent.get("service_query") or "").strip()
    if not q:
        return None

    rows = await conn.fetch(
        """
        SELECT code_type, code
        FROM services
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


# --------------------
# Web search fallback (only when DB insufficient)
# Uses OpenAI Responses tool: web_search :contentReference[oaicite:1]{index=1}
# --------------------
def web_search_fallback_text(question: str) -> str:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role":"user","content": question}],
        tools=[{"type":"web_search"}],
    )
    return resp.output_text


# --------------------
# Streaming answer generator (SSE-ish over HTTP)
# Uses Responses streaming: stream=True :contentReference[oaicite:2]{index=2}
# --------------------
def stream_answer(question: str, state: Dict[str, Any], results: List[Dict[str, Any]], web_notes: Optional[str]):
    system = (
        "You are CostSavvy.health. Be honest about sources.\n"
        "- If results are present, say they come from our hospital price database.\n"
        "- If web_notes are present, say you used online search because DB was insufficient.\n"
        "- Summarize top 5 cheapest with distance and best_price.\n"
        "- Add a brief disclaimer: prices vary; confirm with hospital/insurer.\n"
    )
    payload = {"question": question, "state": state, "top_results": results[:10], "web_notes": web_notes}

    stream = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role":"system","content": system},
            {"role":"user","content": json.dumps(payload)}
        ],
        stream=True
    )

    full_text = []
    for event in stream:
        # We only forward text deltas to the browser.
        # Event types documented: response.output_text.delta, etc. :contentReference[oaicite:3]{index=3}
        et = getattr(event, "type", None) or event.get("type") if isinstance(event, dict) else None

        if et == "response.output_text.delta":
            delta = event.delta if hasattr(event, "delta") else event.get("delta", "")
            if delta:
                full_text.append(delta)
                yield f"data: {json.dumps({'type':'delta','text': delta})}\n\n"

        if et == "error":
            msg = event.error if hasattr(event, "error") else event.get("error", "Unknown error")
            yield f"data: {json.dumps({'type':'error','message': str(msg)})}\n\n"
            return

    return "".join(full_text)


# --------------------
# STREAMING chat endpoint
# --------------------
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    # Auth + rate limiting
    require_auth(request)
    ip = request.client.host if request.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(429, detail="Rate limit exceeded. Please slow down.")

    if not pool:
        raise HTTPException(500, detail="Database not ready")

    async def event_gen():
        async with pool.acquire() as conn:
            session_id, state = await get_or_create_session(conn, req.session_id)

            # send session_id first
            yield f"data: {json.dumps({'type':'session','session_id': session_id})}\n\n"

            await save_message(conn, session_id, "user", req.message)

            intent = await extract_intent(req.message, state)
            merged = merge_state(state, intent)

            mode = intent.get("mode") or "hybrid"

            # Cash-only toggle: clear payer filter
            if merged.get("cash_only") is True:
                merged["payer_like"] = None
                merged["plan_like"] = None

            # Try to resolve code + query DB if needed
            results: List[Dict[str, Any]] = []
            used_web = False
            web_notes = None

            zipcode = merged.get("zipcode")
            payer_like = merged.get("payer_like")
            plan_like = merged.get("plan_like")

            code_type = merged.get("code_type")
            code = merged.get("code")

            if mode in ["price", "hybrid"]:
                if not (code_type and code):
                    resolved = await resolve_service_code(conn, merged)
                    if resolved:
                        code_type, code = resolved
                        merged["code_type"] = code_type
                        merged["code"] = code

                if zipcode and code_type and code:
                    results = await price_lookup_v3(conn, zipcode, code_type, code, payer_like, plan_like)
                else:
                    cq = intent.get("clarifying_question")
                    if not cq:
                        missing = []
                        if not zipcode: missing.append("your 5-digit ZIP code")
                        if not (code_type and code): missing.append("the procedure (or CPT code)")
                        cq = "What is " + " and ".join(missing) + "?"
                    # Stream the clarifying question as a single delta
                    yield f"data: {json.dumps({'type':'delta','text': cq})}\n\n"
                    await save_message(conn, session_id, "assistant", cq, {"intent": intent})
                    await update_session_state(conn, session_id, merged)
                    yield f"data: {json.dumps({'type':'final','used_web_search': False})}\n\n"
                    return

            # Web fallback only when DB is insufficient
            if mode in ["price", "hybrid"] and len(results) < MIN_DB_RESULTS_BEFORE_WEB:
                used_web = True
                web_notes = web_search_fallback_text(req.message)

            # Stream the model answer
            full_answer = ""
            for chunk in stream_answer(req.message, merged, results, web_notes):
                # chunk is already "data: ...\n\n"
                # Collect final text only for logging (roughly)
                try:
                    obj = json.loads(chunk.split("data: ", 1)[1])
                    if obj.get("type") == "delta":
                        full_answer += obj.get("text", "")
                except Exception:
                    pass
                yield chunk

            await save_message(conn, session_id, "assistant", full_answer, {"intent": intent, "result_count": len(results), "used_web_search": used_web})
            await update_session_state(conn, session_id, merged)

            used_radius = results[0].get("used_radius_miles") if results else None
            await log_query(conn, session_id, req.message, intent, used_radius, len(results), used_web, full_answer)

            yield f"data: {json.dumps({'type':'final','used_web_search': used_web})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# --------------------
# Non-streaming version (optional)
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    require_auth(request)
    ip = request.client.host if request.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(429, detail="Rate limit exceeded. Please slow down.")
    raise HTTPException(410, detail="Use /chat_stream for streaming UI.")
