# app/main.py
import os
import json
import re
import uuid
import time
import logging
import httpx
from typing import Optional, Any, Dict, List, Tuple

import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI, OpenAIError

# Python fallback refiners (file: app/service_refiners.py)
from app.service_refiners import refiners_registry

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

# Always try to show at least this many facilities in the answer
MIN_FACILITIES_TO_DISPLAY = int(os.getenv("MIN_FACILITIES_TO_DISPLAY", "5"))

# Web search configuration for health education
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes", "y", "on")
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "15"))
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()

# Care navigation: symptom keywords that trigger care-finding mode (not pricing)
SYMPTOM_KEYWORDS = [
    "hurts", "hurt", "pain", "ache", "aching", "sore", "swollen", "swelling",
    "bleeding", "dizzy", "diziness", "nausea", "vomiting", "fever", "cough",
    "headache", "chest pain", "shortness of breath", "can't breathe", "broken",
    "sprain", "twisted", "injured", "injury", "sick", "unwell", "feel bad",
    "emergency", "urgent", "help me find", "where can i go", "need a doctor",
    "need to see", "should i go to"
]

# Health education keywords that trigger web search for explanations
HEALTH_EDUCATION_KEYWORDS = [
    "what is", "what's", "explain", "tell me about", "how does", "why do",
    "symptoms of", "causes of", "treatment for", "recovery from", "risks of",
    "side effects", "preparation for", "after a", "before a", "difference between",
    "is it safe", "should i worry", "normal to", "how long does"
]

# Refiners cache TTL (DB-first, Python fallback)
REFINERS_CACHE_TTL_SECONDS = int(os.getenv("REFINERS_CACHE_TTL_SECONDS", "300"))

# ---- Intent override controls (env-driven) ----
INTENT_OVERRIDE_FORCE_PRICE_ENABLED = os.getenv("INTENT_OVERRIDE_FORCE_PRICE_ENABLED", "true").lower() in (
    "1", "true", "yes", "y", "on"
)
INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS = [
    s.strip().lower()
    for s in os.getenv(
        "INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS",
        "cost,price,how much,pricing,estimate,rate,charge,fee",
    ).split(",")
    if s.strip()
]

# For new pricing questions, do NOT reuse old ZIP/payment fields from session
RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION = os.getenv("RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION", "true").lower() in (
    "1", "true", "yes", "y", "on"
)

# Progressive radius attempts for priced search (miles)
PRICE_RADIUS_ATTEMPTS = [
    int(x) for x in os.getenv("PRICE_RADIUS_ATTEMPTS", "10,25,50,100,200").split(",") if x.strip().isdigit()
] or [10, 25, 50, 100, 200]

# ----------------------------
# App
# ----------------------------
app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)
pool: asyncpg.Pool | None = None

app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------------------
# Models
# ----------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


# ----------------------------
# Rate limiting
# ----------------------------
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
    if not APP_API_KEY:
        return
    if request.headers.get("X-API-Key", "") != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ----------------------------
# Startup / Shutdown
# ----------------------------
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


# ----------------------------
# SSE helpers
# ----------------------------
def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def stream_llm_to_sse(system: str, user_content: str, out_text_parts: List[str]):
    """
    Streams OpenAI chat.completions deltas as SSE 'delta' events.
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
        yield sse({"type": "error", "message": f"OpenAI error: {type(e).__name__}"})
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield sse({"type": "error", "message": f"Streaming error: {type(e).__name__}"})


# ----------------------------
# DB helpers
# ----------------------------
def _coerce_jsonb_to_dict(val) -> dict:
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
        session_id,
        json.dumps({}),
    )
    return session_id, {}


async def save_message(conn: asyncpg.Connection, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
    await conn.execute(
        "INSERT INTO public.chat_message (session_id, role, content, metadata) VALUES ($1,$2,$3,$4::jsonb)",
        session_id,
        role,
        content,
        json.dumps(metadata or {}),
    )


async def update_session_state(conn: asyncpg.Connection, session_id: str, state: Dict[str, Any]):
    await conn.execute(
        "UPDATE public.chat_session SET session_state = $2::jsonb, last_seen = now() WHERE id = $1",
        session_id,
        json.dumps(state),
    )


async def log_query(
    conn: asyncpg.Connection,
    session_id: str,
    question: str,
    intent: dict,
    used_radius: Optional[float],
    result_count: int,
    used_web: bool,
    answer_text: str,
):
    await conn.execute(
        """
        INSERT INTO public.query_log
          (session_id, question, intent_json, used_radius_miles, result_count, used_web_search, answer_text)
        VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7)
        """,
        session_id,
        question,
        json.dumps(intent),
        used_radius,
        result_count,
        used_web,
        answer_text,
    )


# ----------------------------
# Web Search for Health Education
# ----------------------------
async def web_search_health_info(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for health education information.
    Prioritizes authoritative medical sources.
    Returns a list of search results with title, snippet, and url.
    """
    if not WEB_SEARCH_ENABLED:
        return []
    
    results = []
    
    # Try Brave Search API first
    if BRAVE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": BRAVE_API_KEY},
                    params={
                        "q": f"{query} site:mayoclinic.org OR site:webmd.com OR site:healthline.com OR site:medlineplus.gov OR site:cdc.gov",
                        "count": num_results,
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("web", {}).get("results", [])[:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("description", ""),
                            "url": item.get("url", ""),
                            "source": "brave"
                        })
        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
    
    # Fallback to Serper API
    if not results and SERPER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as client:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={
                        "q": f"{query} health medical",
                        "num": num_results,
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("organic", [])[:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", ""),
                            "source": "serper"
                        })
        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
    
    return results


def is_symptom_query(message: str) -> bool:
    """Check if the message describes symptoms or care needs (not pricing)."""
    msg = (message or "").lower()
    return any(kw in msg for kw in SYMPTOM_KEYWORDS)


def is_health_education_query(message: str) -> bool:
    """Check if the user is asking for health education/explanation."""
    msg = (message or "").lower()
    return any(kw in msg for kw in HEALTH_EDUCATION_KEYWORDS)


def extract_health_topic(message: str) -> str:
    """Extract the health topic from a user's question for web search."""
    msg = (message or "").strip()
    # Remove common question prefixes
    prefixes = [
        "what is", "what's", "what are", "explain", "tell me about",
        "how does", "why do", "can you explain", "i want to know about",
        "help me understand"
    ]
    msg_lower = msg.lower()
    for prefix in prefixes:
        if msg_lower.startswith(prefix):
            msg = msg[len(prefix):].strip()
            break
    # Remove trailing question marks and clean up
    msg = msg.rstrip("?").strip()
    return msg


async def lookup_service_explanation(conn: asyncpg.Connection, topic: str) -> Optional[Dict[str, Any]]:
    """
    Look up a service explanation from the database before falling back to web search.
    Returns service info with patient_summary, cpt_explanation if found.
    """
    if not topic:
        return None
    
    # First try service_variants for patient summaries
    row = await conn.fetchrow(
        """
        SELECT sv.cpt_code, sv.variant_name, sv.patient_summary,
               s.cpt_explanation, s.service_description, s.category
        FROM public.service_variants sv
        LEFT JOIN public.services s ON s.code = sv.cpt_code AND s.code_type = 'CPT'
        WHERE sv.variant_name ILIKE '%' || $1 || '%'
           OR sv.patient_summary ILIKE '%' || $1 || '%'
           OR s.cpt_explanation ILIKE '%' || $1 || '%'
           OR s.patient_summary ILIKE '%' || $1 || '%'
        LIMIT 1
        """,
        topic,
    )
    
    if row:
        return {
            "cpt_code": row["cpt_code"],
            "variant_name": row["variant_name"],
            "patient_summary": row["patient_summary"],
            "cpt_explanation": row["cpt_explanation"],
            "service_description": row["service_description"],
            "category": row["category"],
            "source": "database"
        }
    
    # Fallback to services table directly
    row = await conn.fetchrow(
        """
        SELECT code, code_type, service_description, cpt_explanation, patient_summary, category
        FROM public.services
        WHERE cpt_explanation ILIKE '%' || $1 || '%'
           OR service_description ILIKE '%' || $1 || '%'
           OR patient_summary ILIKE '%' || $1 || '%'
        LIMIT 1
        """,
        topic,
    )
    
    if row:
        return {
            "cpt_code": row["code"],
            "code_type": row["code_type"],
            "service_description": row["service_description"],
            "cpt_explanation": row["cpt_explanation"],
            "patient_summary": row["patient_summary"],
            "category": row["category"],
            "source": "database"
        }
    
    return None


async def generate_health_education_response(
    query: str,
    search_results: List[Dict[str, Any]],
    original_message: str,
    db_service_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a patient-friendly health education response using LLM + web search results.
    Optionally incorporates database service info if available.
    """
    # If we have database info, incorporate it
    db_context = ""
    if db_service_info:
        parts = []
        if db_service_info.get("patient_summary"):
            parts.append(f"Patient Summary: {db_service_info['patient_summary']}")
        if db_service_info.get("cpt_explanation"):
            parts.append(f"Medical Description: {db_service_info['cpt_explanation']}")
        if db_service_info.get("category"):
            parts.append(f"Category: {db_service_info['category']}")
        if parts:
            db_context = "\n\nFrom our medical database:\n" + "\n".join(parts)
    
    if not search_results and not db_service_info:
        # Fallback to LLM-only response with appropriate caveats
        system = (
            "You are CostSavvy.health, a helpful healthcare assistant. "
            "Provide clear, patient-friendly explanations of medical topics. "
            "Always include appropriate caveats that you are not providing medical advice "
            "and that users should consult their healthcare provider for personal medical questions. "
            "Be accurate and cite authoritative sources when possible."
        )
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": original_message}
                ],
                timeout=20,
            )
            answer = (resp.choices[0].message.content or "").strip()
            return answer + "\n\n*This is general health information, not medical advice. Please consult your healthcare provider for questions about your specific situation.*"
        except Exception as e:
            logger.warning(f"LLM health education failed: {e}")
            return "I couldn't find detailed information on that topic. Please consult a healthcare provider or visit trusted sources like MayoClinic.org or MedlinePlus.gov."
    
    # Build context from search results
    context_parts = []
    sources = []
    
    # Add database info first if available
    if db_context:
        context_parts.append(db_context)
    
    for i, r in enumerate(search_results[:3], 1):
        context_parts.append(f"Source {i}: {r.get('title', 'Unknown')}\n{r.get('snippet', '')}")
        if r.get('url'):
            sources.append(f"- [{r.get('title', 'Source')}]({r.get('url')})")
    
    context = "\n\n".join(context_parts)
    
    system = (
        "You are CostSavvy.health, a helpful healthcare assistant. "
        "Use the provided information to give an accurate, patient-friendly explanation. "
        "Prioritize information from our medical database when available. "
        "Synthesize the information clearly. Do not make up information not in the sources. "
        "If the sources don't fully answer the question, say so. "
        "Keep the response concise but complete."
    )
    
    user_prompt = f"""User question: {original_message}

Search results:
{context}

Provide a clear, helpful response based on these sources. End with a brief disclaimer about consulting healthcare providers."""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            timeout=20,
        )
        answer = (resp.choices[0].message.content or "").strip()
        
        # Add sources
        if sources:
            answer += "\n\n**Learn more:**\n" + "\n".join(sources[:3])
        
        return answer
    except Exception as e:
        logger.warning(f"LLM health education with search failed: {e}")
        # Return a summary of search results
        lines = ["Here's what I found:\n"]
        for r in search_results[:3]:
            lines.append(f"**{r.get('title', 'Source')}**")
            lines.append(r.get('snippet', ''))
            if r.get('url'):
                lines.append(f"[Read more]({r.get('url')})\n")
        lines.append("\n*Please consult your healthcare provider for personalized medical advice.*")
        return "\n".join(lines)


# ----------------------------
# Intent extraction
# ----------------------------
INTENT_RULES = """
Return ONLY JSON with:
mode: "general" | "price" | "hybrid" | "clarify" | "care" | "education"
zipcode: 5-digit ZIP or null
radius_miles: number or null
payer_like: string like "Aetna" or null
plan_like: string like "PPO" or null
payment_mode: "cash" | "insurance" | null
service_query: short phrase like "chest x-ray" or null
code_type: string or null (usually "CPT")
code: string or null (like "71046")
clarifying_question: string or null
cash_only: boolean
health_topic: string or null (for education mode, the topic to explain)

Notes:
- If user says "cash price", "self-pay", "out of pocket" => payment_mode="cash" and cash_only=true.
- If user mentions insurance or a carrier name => payment_mode="insurance".
- If user describes symptoms like pain, injury, illness => mode="care" (help find care, not price).
- If user asks "what is", "explain", "tell me about" a medical topic => mode="education".
- mode="price" is for explicit pricing questions about specific services.
"""


def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in ["zipcode", "radius_miles", "payer_like", "plan_like", "payment_mode", "service_query", "code_type", "code", "cash_only"]:
        v = intent.get(k)
        if v is not None and v != "":
            out[k] = v
    return out


def _is_zip_only_message(msg: str) -> bool:
    s = (msg or "").strip()
    return len(s) == 5 and s.isdigit()


def _normalize_payment_mode(merged: Dict[str, Any]) -> None:
    if merged.get("cash_only") is True:
        merged["payment_mode"] = "cash"
        merged["payer_like"] = None
        merged["plan_like"] = None
        return

    pm = merged.get("payment_mode")
    if pm == "cash":
        merged["cash_only"] = True
        merged["payer_like"] = None
        merged["plan_like"] = None
    elif pm == "insurance":
        merged["cash_only"] = False


async def extract_intent(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic session-aware intent extraction.

    Key goals:
      - Detect symptom/care queries and route to care navigation (not pricing).
      - Detect health education queries and route to web search + LLM explanation.
      - Keep the user in the pricing flow once they started it.
      - Ask for ZIP first if missing (geo-first).
      - Accept short replies like "Cash", "Self pay", "Insurance", "Aetna" as continuations.
      - Allow the user to change ZIP/payment/payer mid-session and rerun pricing accordingly.
    """
    msg = (message or "").strip()
    msg_l = msg.lower()

    # Pull existing context
    st = state or {}
    awaiting = st.get("_awaiting")
    have_service = bool(st.get("service_query") or st.get("code"))
    have_zip = bool(st.get("zipcode"))

    # EARLY DETECTION: Symptom/care queries (e.g., "my knee hurts")
    # ALWAYS trigger for symptoms - this overrides any pricing flow
    # Symptoms indicate a NEW concern that needs care navigation, not pricing
    if is_symptom_query(msg):
        return {
            "mode": "care",
            "zipcode": None,  # Always ask for ZIP fresh for care queries
            "symptom_description": msg,
            "clarifying_question": None,
            "_reset_session": True,  # Signal to reset pricing state
        }

    # EARLY DETECTION: Health education queries (e.g., "what is a colonoscopy")
    # Trigger if this looks like an education question (even mid-flow)
    if is_health_education_query(msg) and awaiting not in {"variant_choice", "variant_confirm_yesno"}:
        health_topic = extract_health_topic(msg)
        return {
            "mode": "education",
            "health_topic": health_topic,
            "original_question": msg,
            "clarifying_question": None,
        }

    # 0) ZIP detection anywhere in the message
    zip_match = re.search(r"\b(\d{5})\b", msg)
    if zip_match:
        z = zip_match.group(1)
        # If we are awaiting a ZIP for care mode, return to care mode with the ZIP
        if awaiting == "care_zip":
            return {
                "mode": "care",
                "zipcode": z,
                "symptom_description": st.get("_symptom_description"),
                "clarifying_question": None,
            }
        # If we already have a service in session, treat this as continuation of pricing flow.
        if have_service or awaiting in {"zip", "payment", "payer"}:
            return {
                "mode": "price",
                "zipcode": z,
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": st.get("payment_mode"),
                "payer_like": st.get("payer_like"),
                "plan_like": st.get("plan_like"),
                "clarifying_question": None,
            }

    # Helpers
    cash_terms = {
        "cash", "self pay", "self-pay", "selfpay", "out of pocket", "oop",
        "paying cash", "pay cash", "cash pay", "self pay patient", "selfpay patient",
    }
    insurance_terms = {"insurance", "insured", "use insurance", "with insurance"}

    carrier_map = {
        "aetna": "Aetna",
        "cigna": "Cigna",
        "anthem": "Anthem",
        "blue cross": "Blue Cross Blue Shield",
        "bcbs": "Blue Cross Blue Shield",
        "united": "UnitedHealthcare",
        "uhc": "UnitedHealthcare",
        "humana": "Humana",
        "kaiser": "Kaiser Permanente",
        "molina": "Molina",
        "centene": "Centene",
        "wellcare": "Wellcare",
        "medicaid": "Medicaid",
        "medicare": "Medicare",
    }

    def extract_carrier(m: str) -> Optional[str]:
        ml = (m or "").lower()
        # direct map matches
        for k, v in carrier_map.items():
            if k in ml:
                return v

        # Filter out common filler words so we don't accidentally treat
        # "I have insurance" or "use insurance" as the carrier name "I Have Insurance".
        clean = m
        for stop in ["i have", "i use", "use", "with", "insurance", "my", "have", "paying"]:
            # Word-boundary aware case-insensitive replacement
            clean = re.sub(r'\b' + re.escape(stop) + r'\b', '', clean, flags=re.IGNORECASE)

        clean = clean.strip()
        tokens = re.findall(r"[a-zA-Z]+", clean)

        # if the user replies with just a single word, treat it as a carrier candidate
        if len(tokens) == 1 and len(tokens[0]) >= 3:
            return tokens[0].title()
        
        # two-word carrier (e.g., "Harvard Pilgrim")
        if 2 <= len(tokens) <= 3:
            return " ".join([t.title() for t in tokens])
            
        return None


    # 0b) If we are awaiting a service variant choice, accept a number only.
    if awaiting == "variant_choice":
        if msg.strip().isdigit():
            return {"mode": "price", "variant_choice": int(msg.strip())}
        return {"mode": "clarify", "clarifying_question": "Please reply with the number for the option that matches your service."}

    # 1) If we are explicitly awaiting ZIP, only accept ZIP (or a cancel/new question handled by LLM below).
    if awaiting == "zip":
        # If they didn't provide a ZIP, keep asking.
        return {"mode": "clarify", "clarifying_question": "What’s your 5-digit ZIP code?"}

    # 2) If we are awaiting payment, accept cash/insurance quickly.
    if awaiting == "payment":
        if any(t in msg_l for t in cash_terms):
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "cash",
                "payer_like": None,
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": True,
            }

        if any(t in msg_l for t in insurance_terms) or extract_carrier(msg):
            payer_like = extract_carrier(msg)
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "insurance",
                "payer_like": payer_like,
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": False,
            }

        return {"mode": "clarify", "clarifying_question": "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"}

    # 3) Session-aware continuation even when not awaiting:
    # If the session already has service + zip, treat "cash"/"insurance"/carrier-only as updates and rerun pricing.
    if have_service and have_zip:
        if any(t in msg_l for t in cash_terms):
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "cash",
                "payer_like": None,
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": True,
            }

        carrier = extract_carrier(msg)
        if any(t in msg_l for t in insurance_terms) or carrier:
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "insurance",
                "payer_like": carrier or st.get("payer_like"),
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": False,
            }

    # 4) If the user is asking a new price question but we lack ZIP, we will ask for ZIP (geo-first).
    inferred_service = infer_service_query_from_message(msg)
    if inferred_service and not have_zip:
        # Force clarifying ZIP instead of letting the LLM ask for payment first.
        return {"mode": "price", "service_query": inferred_service, "clarifying_question": None, "_new_price_question": True}

    # 5) Otherwise fall back to LLM-based intent extraction (general Q&A, non-price).
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You extract intent for healthcare Q&A and price lookup. Be conservative."},
                {"role": "system", "content": INTENT_RULES},
                {"role": "user", "content": json.dumps({"message": msg, "session_state": st})},
            ],
            timeout=10,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Intent extraction failed: {e}")
        return {"mode": "clarify", "clarifying_question": "What 5-digit ZIP code should I search near?"}

def infer_service_query_from_message(message: str) -> Optional[str]:
    msg = (message or "").lower()
    if "colonoscopy" in msg:
        return "colonoscopy"
    if "mammogram" in msg or "mammo" in msg:
        return "mammogram"
    if "ultrasound" in msg:
        return "ultrasound"
    if "cat scan" in msg or "ct scan" in msg or (" ct " in f" {msg} "):
        return "ct scan"
    if "mri" in msg:
        return "mri"
    if "x-ray" in msg or "xray" in msg:
        return "x-ray"
    if "blood test" in msg or "lab test" in msg or "labs" in msg:
        return "lab test"
    if "office visit" in msg or "doctor visit" in msg:
        return "office visit"
    return None


def should_force_price_mode(message: str, merged: Dict[str, Any]) -> bool:
    if not INTENT_OVERRIDE_FORCE_PRICE_ENABLED:
        return False

    # If we already have a pricing context (service + ZIP) and are still collecting
    # payment details, keep the conversation in price mode even if the user's reply
    # doesn't include explicit cost keywords (e.g., "I have Aetna insurance").
    if (merged or {}).get("service_query") and (merged or {}).get("zipcode") and not (merged or {}).get("payment_mode"):
        return True
    if (merged or {}).get("_awaiting") in {"zip", "payment"}:
        return True

    msg = (message or "").lower()
    return any(kw in msg for kw in INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS)


def apply_intent_override_if_needed(intent: Dict[str, Any], message: str, merged: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    if should_force_price_mode(message, merged):
        prev = intent.get("mode")
        if prev != "price":
            logger.info(
                "Intent override applied: forcing mode=price",
                extra={"session_id": session_id, "prev_mode": prev, "keywords": INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS},
            )
        intent["mode"] = "price"
        intent["intent_overridden"] = True
        intent["override_reason"] = "cost_keyword"

        if not (merged.get("service_query") or "").strip():
            inferred = infer_service_query_from_message(message)
            if inferred:
                merged["service_query"] = inferred
                logger.info("Service query inferred from message", extra={"session_id": session_id, "service_query": inferred})
    else:
        intent["intent_overridden"] = False

    return intent


def message_contains_zip(message: str) -> bool:
    msg = (message or "").strip()
    tokens = [t.strip(",.()[]{}") for t in msg.split()]
    return any(len(t) == 5 and t.isdigit() for t in tokens)


def message_contains_payment_info(message: str) -> bool:
    msg = (message or "").lower()
    cash_terms = ["cash", "self pay", "self-pay", "out of pocket", "out-of-pocket", "uninsured", "no insurance"]
    ins_terms = ["insurance", "insured", "copay", "coinsurance", "deductible"]
    return any(t in msg for t in cash_terms) or any(t in msg for t in ins_terms)


def reset_gating_fields_for_new_price_question(message: str, merged: Dict[str, Any]) -> None:
    if not message_contains_zip(message):
        merged.pop("zipcode", None)

    if not message_contains_payment_info(message):
        merged.pop("payment_mode", None)
        merged.pop("payer_like", None)
        merged.pop("plan_like", None)
        merged.pop("cash_only", None)


# ----------------------------
# DB: resolve code + price lookup
# ----------------------------
async def resolve_service_code(conn: asyncpg.Connection, merged: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if merged.get("code_type") and merged.get("code"):
        return merged["code_type"], merged["code"]

    q = (merged.get("service_query") or "").strip()
    if not q:
        return None

    # 1) Canonical lookup from `services`
    rows = await conn.fetch(
        """
        SELECT code_type, code
        FROM public.services
        WHERE (cpt_explanation ILIKE '%' || $1 || '%'
            OR service_description ILIKE '%' || $1 || '%'
            OR patient_summary ILIKE '%' || $1 || '%')
        ORDER BY code_type, code
        LIMIT 5
        """,
        q,
    )
    if rows:
        return rows[0]["code_type"], rows[0]["code"]

    # 2) Fallback: staged hospital file often has the right CPT even when `services` lacks the keyword
    srows = await conn.fetch(
        """
        SELECT code_type, code, COUNT(*) AS n
        FROM public.stg_hospital_rates
        WHERE code_type IS NOT NULL AND code IS NOT NULL
          AND (
                service_description ILIKE '%' || $1 || '%'
             OR code ILIKE '%' || $1 || '%'
          )
        GROUP BY code_type, code
        ORDER BY n DESC, code_type, code
        LIMIT 5
        """,
        q,
    )
    if srows:
        return srows[0]["code_type"], srows[0]["code"]

    return None

async def price_lookup_v3(
    conn: asyncpg.Connection,
    zipcode: str,
    code_type: str,
    code: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
    radius_array: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    radius_array = radius_array or [10, 25, 50]
    rows = await conn.fetch(
        """
        SELECT *
        FROM public.get_prices_by_zip_radius_v3(
          $1, $2, $3, $4, $5,
          $6::int[], 10, 25
        );
        """,
        zipcode,
        code_type,
        code,
        payer_like,
        plan_like,
        radius_array,
    )
    return [dict(r) for r in rows]


async def price_lookup_staging(
    conn: asyncpg.Connection,
    zipcode: str,
    service_query: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """
    DB-first fallback when we cannot resolve a code or the pricing function returns no rows.
    It searches `stg_hospital_rates` by service_description and joins to `hospitals` + `zip_locations`
    to compute distance (when possible). Returns a row shape compatible with `build_facility_block`.
    """
    q = (service_query or "").strip()
    if not q:
        return []

    z = await conn.fetchrow(
        "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1 LIMIT 1",
        zipcode,
    )
    zlat = float(z["latitude"]) if z and z["latitude"] is not None else None
    zlon = float(z["longitude"]) if z and z["longitude"] is not None else None

    where_bits = ["r.service_description ILIKE '%' || $2 || '%'"]
    args = [zipcode, q, payer_like, plan_like, limit]
    # payer/plan filters are optional
    payer_clause = "($3::text IS NULL OR r.payer_name ILIKE '%' || $3 || '%')"
    plan_clause = "($4::text IS NULL OR r.plan_name ILIKE '%' || $4 || '%')"

    if zlat is not None and zlon is not None:
        sql = """
        SELECT
          h.name AS hospital_name,
          h.address AS address,
          h.state AS state,
          h.zipcode AS zipcode,
          h.phone AS phone,
          h.latitude AS latitude,
          h.longitude AS longitude,
          (3959 * acos(
              cos(radians($6)) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians($7)) +
              sin(radians($6)) * sin(radians(h.latitude))
          )) AS distance_miles,
          r.code_type,
          r.code,
          r.payer_name,
          r.plan_name,
          r.standard_charge_discounted_cash AS standard_charge_cash,
          r.standard_charge_negotiated_dollar AS negotiated_dollar,
          r.estimated_amount AS estimated_amount
        FROM public.stg_hospital_rates r
        LEFT JOIN public.hospitals h ON h.id = r.hospital_id
        WHERE r.service_description ILIKE '%' || $2 || '%'
          AND """ + payer_clause + """ 
          AND """ + plan_clause + """ 
          AND h.latitude IS NOT NULL AND h.longitude IS NOT NULL
        ORDER BY distance_miles ASC NULLS LAST
        LIMIT $5
        """
        rows = await conn.fetch(sql, zipcode, q, payer_like, plan_like, limit, zlat, zlon)
        return [dict(r) for r in rows]

# ----------------------------
# Nearest-facility pricing (geo-first, ensures a facility list)
# ----------------------------
async def get_service_id(conn: asyncpg.Connection, code_type: str, code: str) -> Optional[int]:
    row = await conn.fetchrow(
        "SELECT id FROM public.services WHERE code_type = $1 AND code = $2 LIMIT 1",
        code_type,
        code,
    )
    return int(row["id"]) if row else None


async def get_service_ids(conn: asyncpg.Connection, service_query: str, code_type: Optional[str] = None, code: Optional[str] = None, limit: int = 25) -> List[int]:
    """Return a list of candidate service_ids for a user query.

    We prefer keyword matches in `services` (description/explanations), and we also include the
    explicitly-resolved (code_type, code) id when provided.
    """
    q = (service_query or "").strip()
    ids: List[int] = []

    # 1) Keyword matches
    if q:
        rows = await conn.fetch(
            """
            SELECT id
            FROM public.services
            WHERE (cpt_explanation ILIKE '%' || $1 || '%'
                OR service_description ILIKE '%' || $1 || '%'
                OR patient_summary ILIKE '%' || $1 || '%')
            ORDER BY id
            LIMIT $2
            """,
            q,
            limit,
        )
        ids.extend([int(r["id"]) for r in rows if r and r["id"] is not None])

    # 2) Explicit code match (ensures at least one id if the keyword search is sparse)
    if code_type and code:
        sid = await get_service_id(conn, code_type, code)
        if sid is not None:
            ids.append(int(sid))

    # de-dup, preserve order
    out: List[int] = []
    seen = set()
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out



async def price_lookup_nearest_facilities(
    conn: asyncpg.Connection,
    zipcode: str,
    code_type: str,
    code: str,
    service_query: str,
    payment_mode: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
    limit: int = 10,
    radius_array: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns nearest facilities (by distance to ZIP centroid) with best-available price fields.

    For cash: pulls MIN(standard_charge_cash) (fallback to MIN(estimated_amount), MIN(standard_charge_gross))
    For insurance: pulls the first matching plan row by payer/plan filters with negotiated_dollar preferred.
    """
    radius_array = radius_array or [10, 25, 50, 100]
    z = await conn.fetchrow("SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1", zipcode)
    if not z:
        return []

    zlat, zlon = float(z["latitude"]), float(z["longitude"])
    service_ids = await get_service_ids(conn, service_query=service_query, code_type=code_type, code=code, limit=25)
    if not service_ids:
        return []

    # We expand radius until we have enough rows, then return the best set.
    last_rows: List[Dict[str, Any]] = []
    for r in radius_array:
        if payment_mode.lower() == "cash":
            rows = await conn.fetch(
                """
                WITH user_zip AS (
                    SELECT $1::double precision AS zlat, $2::double precision AS zlon
                )
                SELECT
                    h.id AS hospital_id,
                    h.name AS hospital_name,
                    h.address,
                    h.state,
                    h.zipcode,
                    h.phone,
                    (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                    )) AS distance_miles,
                    nr.best_price,
                    nr.standard_charge_cash,
                    nr.estimated_amount,
                    nr.standard_charge_gross
                FROM public.hospitals h
                JOIN LATERAL (
                    SELECT
                        COALESCE(
                            MIN(standard_charge_cash),
                            MIN(estimated_amount),
                            MIN(standard_charge_gross)
                        ) AS best_price,
                        MIN(standard_charge_cash) AS standard_charge_cash,
                        MIN(estimated_amount) AS estimated_amount,
                        MIN(standard_charge_gross) AS standard_charge_gross
                    FROM public.negotiated_rates
                    WHERE hospital_id = h.id
                      AND service_id = ANY($3::int[])
                ) nr ON TRUE
                WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
                  AND (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                  )) <= $4
                  AND nr.best_price IS NOT NULL
                ORDER BY distance_miles ASC
                LIMIT $5
                """,
                zlat, zlon, service_ids, r, limit,
            )
        else:
            payer_pat = f"%{payer_like}%" if payer_like else "%"
            plan_pat = f"%{plan_like}%" if plan_like else "%"
            rows = await conn.fetch(
                """
                WITH user_zip AS (
                    SELECT $1::double precision AS zlat, $2::double precision AS zlon
                )
                SELECT
                    h.id AS hospital_id,
                    h.name AS hospital_name,
                    h.address,
                    h.state,
                    h.zipcode,
                    h.phone,
                    (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                    )) AS distance_miles,
                    pick.best_price,
                    pick.negotiated_dollar,
                    pick.estimated_amount,
                    pick.standard_charge_cash,
                    pick.payer_name,
                    pick.plan_name
                FROM public.hospitals h
                JOIN LATERAL (
                    SELECT
                        COALESCE(nr.negotiated_dollar, nr.estimated_amount, nr.standard_charge_cash) AS best_price,
                        nr.negotiated_dollar,
                        nr.estimated_amount,
                        nr.standard_charge_cash,
                        ip.payer_name,
                        ip.plan_name
                    FROM public.negotiated_rates nr
                    JOIN public.insurance_plans ip ON ip.id = nr.plan_id
                    WHERE nr.hospital_id = h.id
                      AND nr.service_id = ANY($3::int[])
                      AND ip.payer_name ILIKE $4
                      AND ip.plan_name ILIKE $5
                      AND (nr.negotiated_dollar IS NOT NULL OR nr.estimated_amount IS NOT NULL OR nr.standard_charge_cash IS NOT NULL)
                    ORDER BY
                        nr.negotiated_dollar NULLS LAST,
                        nr.estimated_amount NULLS LAST,
                        nr.standard_charge_cash NULLS LAST
                    LIMIT 1
                ) pick ON TRUE
                WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
                  AND (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                  )) <= $6
                ORDER BY distance_miles ASC
                LIMIT $7
                """,
                zlat, zlon, service_ids, payer_pat, plan_pat, r, limit,
            )

        last_rows = [dict(rr) for rr in rows] if rows else last_rows
        if len(last_rows) >= MIN_FACILITIES_TO_DISPLAY:
            return last_rows

    return last_rows


    # No lat/lon available, return without distance ordering
    sql2 = """
    SELECT
      COALESCE(h.name, r.hospital_name) AS hospital_name,
      h.address AS address,
      h.state AS state,
      h.zipcode AS zipcode,
      h.phone AS phone,
      NULL::float AS distance_miles,
      r.code_type,
      r.code,
      r.payer_name,
      r.plan_name,
      r.standard_charge_discounted_cash AS standard_charge_cash,
      r.standard_charge_negotiated_dollar AS negotiated_dollar,
      r.estimated_amount AS estimated_amount
    FROM public.stg_hospital_rates r
    LEFT JOIN public.hospitals h ON h.id = r.hospital_id
    WHERE r.service_description ILIKE '%' || $2 || '%'
      AND """ + payer_clause + """ 
      AND """ + plan_clause + """ 
    ORDER BY COALESCE(h.state, ''), COALESCE(h.zipcode, ''), COALESCE(h.name, r.hospital_name, '')
    LIMIT $5
    """
    rows = await conn.fetch(sql2, zipcode, q, payer_like, plan_like, limit)
    return [dict(r) for r in rows]


async def price_lookup_progressive(
    conn: asyncpg.Connection,
    zipcode: str,
    code_type: str,
    code: str,
    service_query: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
    payment_mode: str,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Geo-first nearest-facility pricing lookup with expanding radius."""
    radius_array = [10, 25, 50, 100]

    rows = await price_lookup_nearest_facilities(
        conn,
        zipcode=zipcode,
        code_type=code_type,
        code=code,
        service_query=service_query,
        payment_mode=payment_mode,
        payer_like=payer_like,
        plan_like=plan_like,
        limit=max(MIN_FACILITIES_TO_DISPLAY, 10),
        radius_array=radius_array,
    )

    used_radius = None
    if rows:
        used_radius = radius_array[-1]
    return rows, used_radius

# ----------------------------
# Nearby hospitals (for facility list even when no prices)
# ----------------------------
async def get_nearby_hospitals(conn: asyncpg.Connection, zipcode: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Returns nearby hospitals with name/address/state/zipcode/phone (+ distance if possible).
    Uses public.hospital_details for facility directory fields:
      name, address, state, zipcode, phone, latitude, longitude
    """
    z = await conn.fetchrow(
        "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1 LIMIT 1",
        zipcode,
    )
    zlat = float(z["latitude"]) if z and z["latitude"] is not None else None
    zlon = float(z["longitude"]) if z and z["longitude"] is not None else None

    if zlat is not None and zlon is not None:
        q = """
        SELECT
          h.hospital_name AS hospital_name,
          h.address AS address,
          h.state AS state,
          h.zipcode AS zipcode,
          h.phone AS phone,
          h.latitude AS latitude,
          h.longitude AS longitude,
          (3959 * acos(
              cos(radians($2)) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians($3)) +
              sin(radians($2)) * sin(radians(h.latitude))
          )) AS distance_miles
        FROM public.hospital_details h
        WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
        ORDER BY distance_miles ASC
        LIMIT $1
        """
        rows = await conn.fetch(q, limit, zlat, zlon)
        # If no results from hospital_details, try hospitals table
        if not rows:
            q_fallback = """
            SELECT
              h.name AS hospital_name,
              h.address AS address,
              h.state AS state,
              h.zipcode AS zipcode,
              h.phone AS phone,
              h.latitude AS latitude,
              h.longitude AS longitude,
              (3959 * acos(
                  cos(radians($2)) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians($3)) +
                  sin(radians($2)) * sin(radians(h.latitude))
              )) AS distance_miles
            FROM public.hospitals h
            WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
            ORDER BY distance_miles ASC
            LIMIT $1
            """
            rows = await conn.fetch(q_fallback, limit, zlat, zlon)
        
        return [dict(r) for r in rows]

    # If we can't compute distance, still return facilities (join both tables)
    q2 = """
    SELECT
      COALESCE(hd.hospital_name, h.name) AS hospital_name,
      COALESCE(hd.address, h.address) AS address,
      COALESCE(hd.state, h.state) AS state,
      COALESCE(hd.zipcode, h.zipcode) AS zipcode,
      COALESCE(hd.phone, h.phone) AS phone,
      NULL::float AS distance_miles
    FROM public.hospitals h
    LEFT JOIN public.hospital_details hd ON hd.hospital_id = h.id
    ORDER BY h.state, h.zipcode, h.name
    LIMIT $1
    """
    rows = await conn.fetch(q2, limit)
    return [dict(r) for r in rows]


# ----------------------------
# Estimated range (when DB price missing)
# ----------------------------
def estimate_cost_range(service_query: str, payment_mode: str) -> str:
    """
    Returns a short '$X–$Y' range, explicitly an estimate.
    This is used ONLY when DB pricing is missing.
    """
    system = (
        "You output ONLY a short numeric range for a U.S. healthcare service.\n"
        "Format: '$X–$Y'. No extra text.\n"
        "Be conservative and plausible."
    )
    user = json.dumps({"service": service_query, "payment_mode": payment_mode})
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=12,
        )
        txt = (resp.choices[0].message.content or "").strip()
        # basic sanity
        if "$" in txt and any(ch.isdigit() for ch in txt):
            return txt
        return "$1,000–$3,000"
    except Exception:
        return "$1,000–$3,000"


# ----------------------------
# Facility formatting
# ----------------------------
def _pick_price_fields(row: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort: try common column names returned by your SQL function.
    """
    for k in ["best_price","standard_charge_cash","standard_charge_discounted_cash","cash_price","cash","negotiated_dollar","standard_charge_negotiated_dollar","negotiated_percentage","standard_charge_negotiated_percentage","estimated_amount","standard_charge","standard_charge_gross","rate","price","allowed_amount"]:
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        # sometimes Decimal
        try:
            if v is not None and str(v).replace(".", "", 1).isdigit():
                return float(v)
        except Exception:
            pass
    return None


def _pick_hospital_name(row: Dict[str, Any]) -> str:
    return (row.get("hospital_name") or row.get("name") or "Unknown facility").strip()

def _pick_phone(row: Dict[str, Any]) -> str:
    return (row.get("phone") or "").strip()


def _pick_address(row: Dict[str, Any]) -> str:
    # Build a readable one-liner from common fields
    addr = row.get("address") or row.get("street_address") or ""
    city = row.get("city") or ""
    state = row.get("state") or ""
    z = row.get("zipcode") or row.get("zip") or ""
    parts = [p for p in [addr, city, state, z] if p]
    return ", ".join(parts).strip()


def _format_money(v: Optional[float]) -> str:
    if v is None:
        return ""
    try:
        return f"${v:,.0f}"
    except Exception:
        return f"${v}"



def build_service_education_bullets(service_query: str, payment_mode: str) -> List[str]:
    """
    Patient-facing, service-appropriate caveats. Keep generic unless we are confident.
    """
    s = (service_query or "").lower()
    bullets: List[str] = []

    # Imaging / diagnostics
    if any(k in s for k in ["mri", "magnetic resonance"]):
        bullets += [
            "- MRI prices vary by **body part** and whether it’s **with or without contrast**.",
            "- Many totals include both the **technical** fee (scanner/facility) and the **professional** fee (radiologist read).",
            "- If sedation is used, that can add to the total.",
        ]
    elif any(k in s for k in ["ct", "cat scan", "computed tomography"]):
        bullets += [
            "- CT prices vary by **body part** and whether it’s **with or without contrast**.",
            "- If contrast is used, there may be extra charges (supplies and monitoring).",
            "- Facility and radiologist interpretation fees may be billed separately.",
        ]
    elif any(k in s for k in ["x-ray", "xray", "radiograph"]):
        bullets += [
            "- X-ray prices vary by **body part** and the number of views.",
            "- There may be separate charges for the **facility** and the **radiologist interpretation**.",
        ]
    elif "ultrasound" in s or "sonogram" in s:
        bullets += [
            "- Ultrasound prices vary by **body part** and whether it’s **limited** vs **complete**.",
            "- If Doppler/vascular components are included, the total can be higher.",
        ]
    # Common procedures
    elif "colonoscopy" in s:
        bullets += [
            "- Colonoscopies can be **screening** (preventive) or **diagnostic** (symptoms/abnormal finding).",
            "- Facility setting matters: **outpatient endoscopy centers** often differ from **hospital outpatient** pricing.",
            "- If a biopsy or polyp removal happens, total cost can increase.",
        ]
    else:
        bullets += [
            "- Prices can vary by **facility**, how the service is billed, and what’s included.",
            "- The total may include separate facility and professional fees.",
        ]

    # Payment nuance
    if (payment_mode or "").lower().startswith("insur"):
        bullets.append("- Your out-of-pocket cost depends on your plan (deductible, copays, coinsurance), and whether the facility is in-network.")
    return bullets


def build_facility_block(
    service_query: str,
    payment_mode: str,
    priced_results: List[Dict[str, Any]],
    fallback_hospitals: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (text_answer, facilities_payload_for_ui).
    Always produces at least MIN_FACILITIES_TO_DISPLAY facilities in text when possible.
    """
    facilities: List[Dict[str, Any]] = []

    # 1) Start with priced results (dedupe by name)
    seen = set()
    for r in priced_results:
        name = _pick_hospital_name(r)
        key = name.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        facilities.append(r)
        if len(facilities) >= MIN_FACILITIES_TO_DISPLAY:
            break

    # 2) Top up with nearby hospitals if we still need more
    for h in fallback_hospitals:
        if len(facilities) >= MIN_FACILITIES_TO_DISPLAY:
            break
        name = (h.get("hospital_name") or "").strip() or "Unknown facility"
        key = name.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        facilities.append(h)

    # If still empty, return a graceful message (should be rare if hospitals are loaded)
    if not facilities:
        return (
            "I couldn’t find hospitals near that ZIP code in my database yet. "
            "Try another nearby ZIP, or expand the search radius.",
            [],
        )

    # Prepare an estimate range if needed
    est_range = estimate_cost_range(service_query or "this service", payment_mode or "cash")

    # Build answer text
    bullets = build_service_education_bullets(service_query or "this service", payment_mode or "cash")

    lines = []
    lines.extend(bullets)
    lines.append("")
    lines.append(f"Here are nearby options for **{service_query or 'this service'}** ({payment_mode}):")
    for i, f in enumerate(facilities[:MIN_FACILITIES_TO_DISPLAY], start=1):
        name = _pick_hospital_name(f) if "hospital_name" not in f else (f.get("hospital_name") or _pick_hospital_name(f))
        addr = _pick_address(f)
        phone = _pick_phone(f) if "phone" not in f else (f.get("phone") or _pick_phone(f))
        dist = f.get("distance_miles") or f.get("distance") or f.get("miles")
        dist_txt = ""
        try:
            if dist is not None:
                dist_txt = f" ({float(dist):.1f} mi)"
        except Exception:
            pass

        price = _pick_price_fields(f)
        if price is not None:
            price_txt = _format_money(price)
            price_note = "DB price"
        else:
            price_txt = est_range
            price_note = "ESTIMATE (no DB price yet)"

        detail_bits = []
        if addr:
            detail_bits.append(addr)
        if phone:
            detail_bits.append(f"Tel: {phone}")
        if detail_bits:
            detail = " | ".join(detail_bits)
        else:
            detail = "Contact info not available in DB."

        lines.append(f"{i}) **{name}**{dist_txt}")
        lines.append(f"   - {detail}")
        lines.append(f"   - Price: **{price_txt}** ({price_note})")

    lines.append("")
    lines.append("Confirm with the facility and your insurer.")

    # Build facility payload for UI
    ui_payload = []
    for f in facilities[:MIN_FACILITIES_TO_DISPLAY]:
        ui_payload.append(
            {
                "hospital_name": f.get("hospital_name") or _pick_hospital_name(f),
                "address": _pick_address(f),
                "phone": f.get("phone") or _pick_phone(f),
                "distance_miles": f.get("distance_miles") or f.get("distance"),
                "price": _pick_price_fields(f),
                "estimated_range": None if _pick_price_fields(f) is not None else est_range,
                "price_is_estimate": _pick_price_fields(f) is None,
            }
        )

    return "\n".join(lines), ui_payload


# ----------------------------
# Universal Service Variant Gate (DB-first)
# ----------------------------

def _normalize_service_text(s: str) -> str:
    s = (s or "").lower().strip()
    # Normalize common punctuation/spaces: "xray" == "x-ray" == "x ray"
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Join common patterns
    s = s.replace("x ray", "xray")
    return s

def _tokenize_service_text(s: str) -> List[str]:
    s = _normalize_service_text(s)
    toks = [t for t in s.split(" ") if t]
    # Keep only reasonably informative tokens
    toks = [t for t in toks if len(t) >= 2]
    return toks

async def search_service_variants_by_text(conn, user_text: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Search for service variants using the stg_services_cpt table.
    
    The stg_services_cpt table has columns: cpt_code, cpt_explanation, patient_summary, category
    This matches the service_variants.csv data structure.

    Strategy:
    - Normalize text (e.g., xray/x-ray/x ray).
    - Remove pricing/stop words.
    - Use token OR-matching with a simple relevance score.
    """
    base = _normalize_service_text(user_text)
    toks = _tokenize_service_text(base)
    if not toks:
        return []

    # Keep tokens that are not generic "price" language
    stop = {
        "how","much","does","do","an","a","the","cost","costs","price","prices","pricing","estimate","estimated",
        "near","nearest","in","for","of","to","please","give","me","what","is","are",
    }
    toks = [t for t in toks if t and t not in stop]
    if not toks:
        # fall back to the normalized base as a single token (still better than nothing)
        toks = [base.strip().lower()]

    toks = toks[:6]

    # Build OR match conditions and a score
    # stg_services_cpt has: cpt_code, cpt_explanation, patient_summary, category
    where_parts = []
    score_parts = []
    params = []
    for i, tok in enumerate(toks, start=1):
        like = f"%{tok}%"
        params.append(like)
        # Use REPLACE to normalize hyphens in database text for matching
        # This ensures "x-ray" in DB matches "xray" search term
        where_parts.append(f"""(
            LOWER(REPLACE(COALESCE(cpt_explanation,''), '-', '')) LIKE ${i} 
            OR LOWER(REPLACE(COALESCE(patient_summary,''), '-', '')) LIKE ${i}
            OR LOWER(REPLACE(COALESCE(category,''), '-', '')) LIKE ${i}
            OR LOWER(COALESCE(cpt_code,'')) LIKE ${i}
        )""")
        score_parts.append(f"""(
            CASE WHEN LOWER(REPLACE(COALESCE(cpt_explanation,''), '-', '')) LIKE ${i} THEN 2 ELSE 0 END +
            CASE WHEN LOWER(REPLACE(COALESCE(category,''), '-', '')) LIKE ${i} THEN 2 ELSE 0 END +
            CASE WHEN LOWER(REPLACE(COALESCE(patient_summary,''), '-', '')) LIKE ${i} THEN 1 ELSE 0 END
        )""")

    where_sql = " OR ".join(where_parts)
    score_sql = " + ".join(score_parts)

    # Query stg_services_cpt table - this is where the service_variants.csv data lives
    sql = f"""
        SELECT 
            cpt_code,
            cpt_explanation,
            patient_summary,
            category,
            cpt_explanation as variant_name,
            ({score_sql}) AS match_score
        FROM public.stg_services_cpt
        WHERE ({where_sql})
        ORDER BY match_score DESC, category NULLS LAST, cpt_code ASC
        LIMIT {int(limit)}
    """
    
    try:
        rows = await conn.fetch(sql, *params)
        if rows:
            logger.info(f"stg_services_cpt search found {len(rows)} results for query: {user_text}")
            return [dict(r) for r in rows]
        
        logger.info(f"stg_services_cpt empty for query: {user_text}, trying services table")
        
        # Fallback to services table
        fallback_sql = f"""
            SELECT 
                code as cpt_code,
                cpt_explanation,
                patient_summary,
                category,
                service_description as variant_name,
                ({score_sql}) AS match_score
            FROM public.services
            WHERE code_type = 'CPT' AND ({where_sql})
            ORDER BY match_score DESC, category NULLS LAST, code ASC
            LIMIT {int(limit)}
        """
        rows = await conn.fetch(fallback_sql, *params)
        if rows:
            logger.info(f"services fallback found {len(rows)} results for query: {user_text}")
        return [dict(r) for r in rows]
        
    except Exception as e:
        logger.error(f"service variant search failed: {e}")
        return []

def build_variant_numbered_prompt(user_label: str, variants: List[Dict[str, Any]]) -> str:
    base = (_normalize_service_text(user_label) or "this service")
    header = base.upper() if len(base) <= 22 else "this service"
    lines: List[str] = []
    lines.append(f"Before I look up prices, which exact billed **{header}** do you mean?")
    lines.append("")
    for i, v in enumerate(variants, start=1):
        # Build a readable name from cpt_explanation or variant_name
        expl = (v.get("cpt_explanation") or "").strip()
        name = (v.get("variant_name") or "").strip()
        
        # Extract a short, meaningful name from the explanation
        if expl:
            # Take first sentence or first 80 chars
            if ". " in expl:
                display_name = expl.split(". ")[0]
            else:
                display_name = expl[:80]
            if len(display_name) > 80:
                display_name = display_name[:77] + "..."
        elif name:
            display_name = name[:80]
        else:
            display_name = "Option"
        
        cpt = (v.get("cpt_code") or "").strip()
        category = (v.get("category") or "").strip()
        summ = (v.get("patient_summary") or "").strip()
        
        # Build the line
        line = f"{i}) {display_name}"
        if cpt:
            line += f" (CPT {cpt})"
        if category:
            line += f" [{category}]"
        lines.append(line)
        
        # Add summary on next line if available and different from explanation
        if summ and summ != expl and len(summ) < 150:
            lines.append(f"   _{summ}_")
    
    lines.append("")
    lines.append("Reply with the **number only** (e.g., `1`).")
    return "\n".join(lines)


async def maybe_fill_variant_summaries_with_llm(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If patient_summary is missing, ask the LLM for 1 short patient-friendly line per option.

    We keep this optional: if the LLM errors, we fall back to cpt_explanation snippets.
    """
    missing = [i for i, v in enumerate(variants) if not (v.get("patient_summary") or "").strip()]
    if not missing:
        return variants

    # Keep prompt small and deterministic
    items = []
    for i, v in enumerate(variants):
        if i not in missing:
            continue
        items.append(
            {
                "index": i,
                "variant_name": (v.get("variant_name") or "").strip(),
                "cpt_code": (v.get("cpt_code") or "").strip(),
                "cpt_explanation": (v.get("cpt_explanation") or "").strip(),
            }
        )

    system = (
        "You write short patient-friendly explanations of billed medical services. "
        "One sentence each. Focus on purpose and what makes it different (views/with-contrast/etc). "
        "No medical advice."
    )
    user = (
        "Create a JSON array of objects with keys: index, summary. "
        "Summary must be <= 22 words.\n\n" + json.dumps(items)
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            timeout=20,
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = None
        try:
            data = json.loads(txt)
        except Exception:
            # If the model wrapped JSON in text, try to extract the first JSON array.
            m = re.search(r"\[[\s\S]*\]", txt)
            if m:
                data = json.loads(m.group(0))
        if isinstance(data, list):
            by_idx = {}
            for o in data:
                if isinstance(o, dict) and "index" in o and "summary" in o:
                    by_idx[int(o["index"])] = str(o["summary"]).strip()
            for i in missing:
                summ = by_idx.get(i)
                if summ:
                    variants[i]["patient_summary"] = summ
    except Exception:
        return variants
    return variants

def apply_variant_choice_from_candidates(merged: Dict[str, Any], message: str) -> bool:
    """Apply a numeric choice to merged state. Returns True if applied."""
    s = (message or "").strip()
    if not s.isdigit():
        return False
    idx = int(s) - 1
    candidates = merged.get("_variant_candidates") or []
    if idx < 0 or idx >= len(candidates):
        return False
    picked = candidates[idx]
    merged["code_type"] = "CPT"
    merged["code"] = picked.get("cpt_code")
    merged["service_query"] = picked.get("variant_name") or merged.get("service_query")
    merged["variant_id"] = picked.get("id")
    merged["variant_name"] = picked.get("variant_name")
    merged.pop("_variant_candidates", None)
    merged.pop("_awaiting", None)
    return True


def _is_yes(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"y", "yes", "yeah", "yep", "correct", "right"}


def _is_no(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"n", "no", "nope", "nah"}


def build_single_variant_yesno_prompt(user_label: str, v: Dict[str, Any]) -> str:
    name = (v.get("variant_name") or user_label or "this service").strip()
    cpt = (v.get("cpt_code") or "").strip()
    summ = (v.get("patient_summary") or "").strip()
    if not summ:
        expl = (v.get("cpt_explanation") or "").strip()
        summ = expl[:160].strip() + ("..." if len(expl) > 160 else "")
    parts = [
        f"Just to confirm, do you mean **{name}**" + (f" (CPT {cpt})" if cpt else "") + "?",
    ]
    if summ:
        parts.append(f"{summ}")
    parts.append("Reply **Y** or **N**.")
    return "\n".join(parts)


# ----------------------------
# Service refiners (DB-first, Python fallback)
# ----------------------------
_refiners_cache: Optional[dict] = None
_refiners_cache_loaded_at: float = 0.0


def _norm(s: str) -> str:
    return (s or "").strip().lower()


async def load_refiners_from_db(conn: asyncpg.Connection) -> dict:
    rows = await conn.fetch(
        """
        SELECT id, title, keywords, require_choice_before_pricing,
               preview_code_type, preview_code, question_text
        FROM public.service_refiner
        WHERE is_active = true
        ORDER BY id
        """
    )
    if not rows:
        return {"version": 1, "refiners": []}

    ids = [r["id"] for r in rows]
    crows = await conn.fetch(
        """
        SELECT refiner_id, choice_key, choice_label, code_type, code, sort_order
        FROM public.service_refiner_choice
        WHERE is_active = true AND refiner_id = ANY($1::text[])
        ORDER BY refiner_id, sort_order, choice_key
        """,
        ids,
    )

    choices_by_ref: Dict[str, List[dict]] = {}
    for c in crows:
        choices_by_ref.setdefault(c["refiner_id"], []).append(
            {"key": c["choice_key"], "label": c["choice_label"], "code_type": c["code_type"], "code": c["code"]}
        )

    refiners: List[dict] = []
    for r in rows:
        preview = None
        if r["preview_code_type"] and r["preview_code"]:
            preview = {"code_type": r["preview_code_type"], "code": r["preview_code"]}
        refiners.append(
            {
                "id": r["id"],
                "title": r["title"],
                "match": {"keywords": list(r["keywords"] or [])},
                "require_choice_before_pricing": bool(r["require_choice_before_pricing"]),
                "preview_code": preview,
                "question": r["question_text"],
                "choices": choices_by_ref.get(r["id"], []),
            }
        )

    return {"version": 1, "refiners": refiners}


async def get_refiners(conn: asyncpg.Connection) -> dict:
    global _refiners_cache, _refiners_cache_loaded_at
    now = time.time()
    if _refiners_cache and (now - _refiners_cache_loaded_at) < REFINERS_CACHE_TTL_SECONDS:
        return _refiners_cache

    data = None
    try:
        data = await load_refiners_from_db(conn)
    except Exception as e:
        logger.warning(f"Refiners DB load failed, using Python fallback: {e}")

    if not data or not data.get("refiners"):
        data = refiners_registry()

    _refiners_cache = data
    _refiners_cache_loaded_at = now
    return data


def match_refiner(service_query: str, refiners_doc: dict) -> Optional[dict]:
    q = _norm(service_query)
    if not q:
        return None
    for ref in refiners_doc.get("refiners", []):
        kws = [_norm(k) for k in (ref.get("match", {}).get("keywords") or [])]
        if any(k and k in q for k in kws):
            return ref
    return None


def apply_refiner_choice(message: str, merged: dict, refiner: Optional[dict]) -> dict:
    if not refiner:
        return merged
    choice_key = (message or "").strip()
    if not choice_key:
        return merged
    for ch in refiner.get("choices", []):
        if str(ch.get("key")) == choice_key:
            merged["code_type"] = ch.get("code_type")
            merged["code"] = ch.get("code")
            merged["refiner_id"] = refiner.get("id")
            merged["refiner_choice"] = choice_key
            return merged
    return merged


def maybe_apply_preview_code(merged: dict, refiner: Optional[dict]) -> dict:
    if not refiner:
        return merged
    if refiner.get("require_choice_before_pricing") is True:
        return merged
    if merged.get("code_type") and merged.get("code"):
        return merged
    preview = refiner.get("preview_code")
    if preview and preview.get("code_type") and preview.get("code"):
        merged["code_type"] = preview["code_type"]
        merged["code"] = preview["code"]
        merged["refiner_id"] = refiner.get("id")
    return merged


def needs_refinement(merged: dict, refiner: Optional[dict]) -> bool:
    if not refiner:
        return False
    if merged.get("refiner_choice"):
        return False

    code_type = merged.get("code_type")
    code = merged.get("code")
    preview = refiner.get("preview_code")

    if refiner.get("require_choice_before_pricing") is True:
        return True
    if not (code_type and code):
        return True
    if preview and code_type == preview.get("code_type") and code == preview.get("code"):
        return True
    return False


def get_refinement_prompt(refiner: dict) -> str:
    lines = [refiner.get("question", "").strip(), ""]
    for ch in refiner.get("choices", []):
        lines.append(f"{ch.get('key')}) {ch.get('label')}")
    lines.append("")
    lines.append("Reply with the number that fits best.")
    return "\n".join([l for l in lines if l])


# ----------------------------
# Main streaming endpoint
# ----------------------------
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

                # 1) Intent + state merge
                intent = await extract_intent(req.message, state)
                merged = merge_state(state, intent)

                # If this message is a NEW pricing question (service inferred) and the user did not provide a ZIP,
                # do not reuse a prior ZIP/payment from the session. Force ZIP collection first.
                if intent.get("_new_price_question"):
                    if not message_contains_zip(req.message):
                        merged.pop("zipcode", None)
                    # Always reset payment/payer/plan for a new pricing question unless explicitly stated
                    if not message_contains_payment_info(req.message):
                        merged.pop("payment_mode", None)
                        merged.pop("payer_like", None)
                        merged.pop("plan_like", None)
                        merged.pop("cash_only", None)
                    merged.pop("_awaiting", None)
                _normalize_payment_mode(merged)

                # 2) Force/override pricing intent when we are mid-flow
                intent = apply_intent_override_if_needed(intent, req.message, merged, session_id)
                mode = intent.get("mode") or "hybrid"

                # 3) OPTIONAL: reset gating fields only when this message looks like a NEW pricing question
                # (Never reset while we are awaiting ZIP/payment/payer)
                if (
                    RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION
                    and mode in ["price", "hybrid"]
                    and not (merged.get("_awaiting") in {"zip", "payment", "payer"})
                ):
                    # Only reset if the message itself contains a price keyword AND a service hint
                    msg_l = (req.message or "").lower()
                    inferred_service = infer_service_query_from_message(req.message)
                    if inferred_service and any(kw in msg_l for kw in INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS):
                        reset_gating_fields_for_new_price_question(req.message, merged)

                # 4) Load refiners + apply choice (if user replies to a refiner prompt)
                # IMPORTANT: do not let refiner numeric keys hijack the universal variant-choice flow.
                refiners_doc = await get_refiners(conn)
                refiner = match_refiner(merged.get("service_query") or "", refiners_doc)
                if merged.get("_awaiting") not in {"variant_choice", "variant_confirm_yesno", "variant_clarify"}:
                    merged = apply_refiner_choice(req.message, merged, refiner)
                # Do not apply preview codes until a CPT-backed variant is confirmed.
                if merged.get("variant_confirmed") is True or (merged.get("code_type") and merged.get("code")):
                    merged = maybe_apply_preview_code(merged, refiner)

                # ----------------------------
                # GENERAL mode
                # ----------------------------
                if mode == "general":
                    system = "You are CostSavvy.health. Answer clearly in plain language. Avoid medical advice."
                    parts: List[str] = []
                    for chunk in stream_llm_to_sse(system, req.message, parts):
                        yield chunk
                    full_answer = "".join(parts).strip() or "I couldn’t generate a response."
                    await save_message(conn, session_id, "assistant", full_answer, {"mode": "general", "intent": intent})
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, None, 0, False, full_answer)
                    yield sse({"type": "final", "used_web_search": False})
                    return

                # ----------------------------
                # CARE mode (symptom-based, find care)
                # ----------------------------
                if mode == "care":
                    # User described symptoms - help them find care, NOT price services
                    # IMPORTANT: Reset any pricing session state - this is a NEW concern
                    if intent.get("_reset_session"):
                        # Clear pricing-related state
                        merged.pop("service_query", None)
                        merged.pop("code_type", None)
                        merged.pop("code", None)
                        merged.pop("payment_mode", None)
                        merged.pop("payer_like", None)
                        merged.pop("plan_like", None)
                        merged.pop("_variant_candidates", None)
                        merged.pop("_variant_single", None)
                        merged.pop("variant_confirmed", None)
                        merged.pop("zipcode", None)  # Ask for ZIP fresh for care
                    
                    zipcode = intent.get("zipcode")  # Only use ZIP from current intent, not merged
                    
                    if not zipcode:
                        # Ask for ZIP to find nearby care options
                        merged["_awaiting"] = "care_zip"
                        merged["_symptom_description"] = intent.get("symptom_description") or req.message
                        msg = (
                            "I understand you're not feeling well. To help you find care options nearby, "
                            "what's your 5-digit ZIP code?"
                        )
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "care", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return
                    
                    # We have ZIP - find nearby hospitals/urgent care
                    try:
                        nearby = await get_nearby_hospitals(conn, zipcode, limit=5)
                    except Exception as e:
                        logger.warning(f"Care facility lookup failed: {e}")
                        nearby = []
                    
                    # Build care guidance response
                    symptom_desc = merged.get("_symptom_description") or intent.get("symptom_description") or req.message
                    
                    lines = []
                    lines.append(f"I'm sorry you're experiencing that. Here are some nearby care options:\n")
                    
                    if nearby:
                        for i, h in enumerate(nearby[:5], 1):
                            name = h.get("hospital_name") or "Healthcare Facility"
                            addr = _pick_address(h)
                            phone = h.get("phone") or ""
                            dist = h.get("distance_miles")
                            dist_txt = f" ({dist:.1f} mi)" if dist else ""
                            
                            lines.append(f"**{i}. {name}**{dist_txt}")
                            if addr:
                                lines.append(f"   {addr}")
                            if phone:
                                lines.append(f"   Tel: {phone}")
                            lines.append("")
                    else:
                        lines.append("I couldn't find facilities near that ZIP code in my database.")
                        lines.append("")
                    
                    lines.append("**When to seek care:**")
                    lines.append("- **Emergency (911)**: Chest pain, difficulty breathing, severe bleeding, stroke symptoms")
                    lines.append("- **Urgent Care**: Non-life-threatening but needs same-day attention")
                    lines.append("- **Primary Care**: Can wait 1-2 days for an appointment")
                    lines.append("")
                    lines.append("*If you're unsure, call ahead or use a nurse hotline. This is not medical advice.*")
                    
                    full_answer = "\n".join(lines)
                    yield sse({"type": "delta", "text": full_answer})
                    
                    # Clear care-specific state
                    merged.pop("_awaiting", None)
                    merged.pop("_symptom_description", None)
                    
                    await save_message(conn, session_id, "assistant", full_answer, {"mode": "care", "intent": intent})
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, None, len(nearby), False, full_answer)
                    yield sse({"type": "final", "used_web_search": False})
                    return

                # ----------------------------

                # ----------------------------
                # EDUCATION mode (health topic explanation with web search)
                # ----------------------------
                if mode == "education":
                    health_topic = intent.get("health_topic") or extract_health_topic(req.message)
                    original_question = intent.get("original_question") or req.message
                    
                    # First, check if we have info in our database
                    yield sse({"type": "status", "message": "Looking up health information..."})
                    
                    db_service_info = None
                    try:
                        db_service_info = await lookup_service_explanation(conn, health_topic)
                    except Exception as e:
                        logger.warning(f"Database lookup for health education failed: {e}")
                    
                    # Then search the web for additional authoritative information
                    search_results = []
                    used_web = False
                    if WEB_SEARCH_ENABLED and (BRAVE_API_KEY or SERPER_API_KEY):
                        try:
                            search_results = await web_search_health_info(health_topic, num_results=5)
                            used_web = len(search_results) > 0
                        except Exception as e:
                            logger.warning(f"Web search for health education failed: {e}")
                    
                    # Generate response
                    full_answer = await generate_health_education_response(
                        query=health_topic,
                        search_results=search_results,
                        original_message=original_question,
                        db_service_info=db_service_info
                    )
                    
                    yield sse({"type": "delta", "text": full_answer})
                    
                    await save_message(
                        conn, session_id, "assistant", full_answer,
                        {"mode": "education", "intent": intent, "health_topic": health_topic, "sources_count": len(search_results), "db_match": db_service_info is not None}
                    )
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, None, len(search_results), used_web, full_answer)
                    yield sse({"type": "final", "used_web_search": used_web})
                    return
                # PRICE / HYBRID mode (deterministic, DB-first)
                # ----------------------------
                if mode in ["price", "hybrid"]:
                    # Gate 0: need a service (or a code)
                    if not (merged.get("service_query") or "").strip() and not (merged.get("code_type") and merged.get("code")):
                        # Try deterministic inference for common services
                        inferred = infer_service_query_from_message(req.message)
                        if inferred:
                            merged["service_query"] = inferred

                    
                    # Gate 0b: Universal CPT-backed variant confirmation (BEFORE ZIP)
                    # Spec:
                    # - 0 matches: ask for more specificity (still before ZIP)
                    # - 1 match: ask Y/N confirmation
                    # - N matches: show numbered list, user replies with number only
                    if not (merged.get("code_type") and merged.get("code")):
                        # If we previously asked for more specificity, treat the user's reply as an updated service query.
                        if merged.get("_awaiting") == "variant_clarify":
                            merged["service_query"] = infer_service_query_from_message(req.message) or req.message
                            merged.pop("_awaiting", None)
                            merged.pop("_variant_candidates", None)
                            merged.pop("_variant_single", None)

                        # If we are waiting on a Y/N confirmation for a single match.
                        if merged.get("_awaiting") == "variant_confirm_yesno":
                            if _is_yes(req.message):
                                v = merged.get("_variant_single") or {}
                                merged["code_type"] = "CPT"
                                merged["code"] = v.get("cpt_code")
                                merged["service_query"] = v.get("variant_name") or merged.get("service_query")
                                merged["variant_id"] = v.get("id")
                                merged["variant_name"] = v.get("variant_name")
                                merged["variant_confirmed"] = True
                                merged.pop("_variant_single", None)
                                merged.pop("_awaiting", None)
                                await update_session_state(conn, session_id, merged)
                            elif _is_no(req.message):
                                # Clear and ask for a clearer description before ZIP.
                                merged.pop("_variant_single", None)
                                merged.pop("code_type", None)
                                merged.pop("code", None)
                                merged["variant_confirmed"] = False
                                merged["_awaiting"] = "variant_clarify"
                                msg = "No problem, what *exactly* is being ordered? Add details like body part, number of views, with/without contrast, or purpose."
                                yield sse({"type": "delta", "text": msg})
                                await save_message(conn, session_id, "assistant", msg, {"mode": "price"})
                                await update_session_state(conn, session_id, merged)
                                yield sse({"type": "final", "used_web_search": False})
                                return
                            else:
                                v = merged.get("_variant_single") or {}
                                msg = build_single_variant_yesno_prompt(merged.get("service_query") or req.message, v)
                                yield sse({"type": "delta", "text": msg})
                                await save_message(conn, session_id, "assistant", msg, {"mode": "price"})
                                await update_session_state(conn, session_id, merged)
                                yield sse({"type": "final", "used_web_search": False})
                                return

                        # If user just replied with a numeric choice, apply it.
                        if merged.get("_awaiting") == "variant_choice":
                            if apply_variant_choice_from_candidates(merged, req.message):
                                merged["variant_confirmed"] = True
                                await update_session_state(conn, session_id, merged)
                            else:
                                # Re-ask with the same numbered options
                                candidates = merged.get("_variant_candidates") or []
                                if candidates:
                                    msg = build_variant_numbered_prompt(merged.get("service_query") or req.message, candidates)
                                    yield sse({"type": "delta", "text": msg})
                                    await save_message(conn, session_id, "assistant", msg, {"mode": "price"})
                                    await update_session_state(conn, session_id, merged)
                                    yield sse({"type": "final", "used_web_search": False})
                                    return

                        # If we still don't have a code, search variants FIRST before any other gates.
                        # This is the MANDATORY first step per the instructions.
                        if not (merged.get("code_type") and merged.get("code")) and merged.get("_awaiting") not in {"variant_choice", "variant_confirm_yesno", "variant_clarify"}:
                            qtext = merged.get("service_query") or req.message
                            logger.info(f"Searching for service variants with query: {qtext}")
                            raw_candidates = await search_service_variants_by_text(conn, qtext, limit=8)
                            logger.info(f"Variant search returned {len(raw_candidates)} candidates")

                            if not raw_candidates:
                                merged["_awaiting"] = "variant_clarify"
                                msg = (
                                    "I can price this, but I need a bit more detail on the exact billed service. "
                                    "What exactly is being ordered?"
                                )
                                yield sse({"type": "delta", "text": msg})
                                await save_message(conn, session_id, "assistant", msg, {"mode": "price"})
                                await update_session_state(conn, session_id, merged)
                                yield sse({"type": "final", "used_web_search": False})
                                return

                            # Normalize candidates we keep in state
                            candidates = [
                                {
                                    "id": c.get("id"),
                                    "cpt_code": c.get("cpt_code"),
                                    "variant_name": c.get("variant_name") or c.get("cpt_explanation", "")[:80],
                                    "patient_summary": c.get("patient_summary"),
                                    "cpt_explanation": c.get("cpt_explanation"),
                                    "category": c.get("category"),
                                }
                                for c in raw_candidates
                            ]

                            if len(candidates) == 1:
                                # Ask yes/no confirmation
                                candidates = await maybe_fill_variant_summaries_with_llm(candidates)
                                merged["_variant_single"] = candidates[0]
                                merged["_awaiting"] = "variant_confirm_yesno"
                                msg = build_single_variant_yesno_prompt(qtext, candidates[0])
                                yield sse({"type": "delta", "text": msg})
                                await save_message(conn, session_id, "assistant", msg, {"mode": "price"})
                                await update_session_state(conn, session_id, merged)
                                yield sse({"type": "final", "used_web_search": False})
                                return

                            # Multiple matches: show numbered list (fill summaries if needed)
                            candidates = await maybe_fill_variant_summaries_with_llm(candidates)
                            merged["_variant_candidates"] = candidates
                            merged["_awaiting"] = "variant_choice"
                            msg = build_variant_numbered_prompt(qtext, candidates)
                            yield sse({"type": "delta", "text": msg})
                            await save_message(conn, session_id, "assistant", msg, {"mode": "price"})
                            await update_session_state(conn, session_id, merged)
                            await log_query(conn, session_id, req.message, {"mode": "price"}, None, 0, False, msg)
                            yield sse({"type": "final", "used_web_search": False})
                            return

                    # Gate 1: ZIP required
                    zipcode = merged.get("zipcode")
                    if not zipcode:
                        merged["_awaiting"] = "zip"
                        msg = "What’s your 5-digit ZIP code?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_zip", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return
                    if merged.get("_awaiting") == "zip":
                        merged.pop("_awaiting", None)

                    # Gate 2: payment mode required
                    payment_mode = merged.get("payment_mode")
                    if not payment_mode:
                        merged["_awaiting"] = "payment"
                        msg = "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_payment", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return
                    if merged.get("_awaiting") == "payment":
                        merged.pop("_awaiting", None)

                    _normalize_payment_mode(merged)
                    payment_mode = merged.get("payment_mode") or payment_mode

                    # Gate 3 (insurance): require payer
                    if payment_mode == "insurance" and not (merged.get("payer_like") or "").strip():
                        merged["_awaiting"] = "payer"
                        msg = "Which insurance carrier should I match prices for (e.g., Aetna, UnitedHealthcare, Blue Cross)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_payer", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return
                    if merged.get("_awaiting") == "payer" and (merged.get("payer_like") or "").strip():
                        merged.pop("_awaiting", None)

                    # Refiners: ask for choice before pricing when required
                    if refiner and refiner.get("require_choice_before_pricing") is True and not merged.get("refiner_choice"):
                        msg = get_refinement_prompt(refiner)
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_variant", "refiner_id": refiner.get("id"), "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # Resolve code if needed - BUT only if variant was confirmed
                    # Per instructions: ALWAYS require variant selection before pricing
                    if not (merged.get("code_type") and merged.get("code")):
                        # If we don't have a confirmed variant, we should NOT proceed
                        # This should have been handled by Gate 0b above
                        # If we reach here without a code, force variant selection
                        if not merged.get("variant_confirmed"):
                            qtext = merged.get("service_query") or req.message
                            raw_candidates = await search_service_variants_by_text(conn, qtext, limit=8)
                            
                            if raw_candidates:
                                candidates = [
                                    {
                                        "id": c.get("id"),
                                        "cpt_code": c.get("cpt_code"),
                                        "variant_name": c.get("variant_name"),
                                        "patient_summary": c.get("patient_summary"),
                                        "cpt_explanation": c.get("cpt_explanation"),
                                    }
                                    for c in raw_candidates
                                ]
                                candidates = await maybe_fill_variant_summaries_with_llm(candidates)
                                merged["_variant_candidates"] = candidates
                                merged["_awaiting"] = "variant_choice"
                                msg = build_variant_numbered_prompt(qtext, candidates)
                                yield sse({"type": "delta", "text": msg})
                                await save_message(conn, session_id, "assistant", msg, {"mode": "price"})
                                await update_session_state(conn, session_id, merged)
                                yield sse({"type": "final", "used_web_search": False})
                                return
                            
                            # No variants found - try resolve_service_code as fallback
                            resolved = await resolve_service_code(conn, merged)
                            if resolved:
                                merged["code_type"], merged["code"] = resolved
                        else:
                            # Variant was confirmed but code not set - try resolution
                            resolved = await resolve_service_code(conn, merged)
                            if resolved:
                                merged["code_type"], merged["code"] = resolved

                    # If still no service, ask
                    if not (merged.get("service_query") or "").strip() and not (merged.get("code_type") and merged.get("code")):
                        msg = "What service are you pricing (for example: colonoscopy, MRI brain, chest x-ray, office visit, lab test)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_service", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # If code is still missing, ask for more detail
                    if not (merged.get("code_type") and merged.get("code")):
                        msg = "I can price this, but I need a bit more detail on the exact service. What exactly is being ordered?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_service_detail", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # ---- PRICED LOOKUP (geo-first, progressive radius) ----
                    results, used_max_radius = await price_lookup_progressive(
                        conn,
                        merged["zipcode"],
                        merged["code_type"],
                        merged["code"],
                        merged.get("service_query") or "",
                        merged.get("payer_like"),
                        merged.get("plan_like"),
                        merged.get("payment_mode") or "cash",
                    )

                    # Always fetch at least 5 facilities for display
                    try:
                        nearby_hospitals = await get_nearby_hospitals(conn, merged["zipcode"], limit=MIN_FACILITIES_TO_DISPLAY)
                    except Exception as e:
                        logger.warning(f"Nearby hospitals lookup failed: {e}")
                        nearby_hospitals = []

                    facility_text, facility_payload = build_facility_block(
                        service_query=merged.get("service_query") or "this service",
                        payment_mode=merged.get("payment_mode") or "cash",
                        priced_results=results,
                        fallback_hospitals=nearby_hospitals,
                    )

                    yield sse(
                        {
                            "type": "results",
                            "results": results[:25],
                            "facilities": facility_payload,
                            "state": {
                                k: merged.get(k)
                                for k in [
                                    "zipcode",
                                    "payment_mode",
                                    "payer_like",
                                    "plan_like",
                                    "service_query",
                                    "code_type",
                                    "code",
                                    "refiner_id",
                                    "refiner_choice",
                                ]
                            },
                        }
                    )
                    yield sse({"type": "delta", "text": facility_text})

                    full_answer = facility_text
                    await save_message(
                        conn,
                        session_id,
                        "assistant",
                        full_answer,
                        {
                            "mode": mode,
                            "intent": intent,
                            "result_count": len(results),
                            "used_max_radius": used_max_radius,
                            "refiner_id": (refiner or {}).get("id"),
                        },
                    )
                    await update_session_state(conn, session_id, merged)

                    used_web = len(results) < MIN_DB_RESULTS_BEFORE_WEB
                    await log_query(conn, session_id, req.message, intent, used_max_radius, len(results), used_web, full_answer)
                    yield sse({"type": "final", "used_web_search": used_web})
                    return

                # Fallback: ask to rephrase (should be rare)
                msg = "I’m not sure what you need. Can you rephrase?"
                yield sse({"type": "delta", "text": msg})
                await save_message(conn, session_id, "assistant", msg, {"mode": "fallback", "intent": intent})
                await update_session_state(conn, session_id, merged)
                await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                yield sse({"type": "final", "used_web_search": False})

        except Exception as e:
            logger.exception("Unhandled error")
            yield sse({"type": "error", "message": f"{type(e).__name__}: {str(e)}"})
            yield sse({"type": "final", "used_web_search": False})

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@app.post("/chat")

async def chat(_req: ChatRequest, _request: Request):
    raise HTTPException(410, detail="Use /chat_stream")