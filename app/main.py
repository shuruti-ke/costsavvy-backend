# app/main.py
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
# Intent extraction
# ----------------------------
INTENT_RULES = """
Return ONLY JSON with:
mode: "general" | "price" | "hybrid" | "clarify"
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
"""

def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in ["zipcode", "radius_miles", "payer_like", "plan_like", "payment_mode", "service_query", "code_type", "code", "cash_only"]:
        v = intent.get(k)
        if v is not None and v != "":
            out[k] = v
    return out


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
    msg = (message or "").strip()
    msg_l = msg.lower()
    st = state or {}
    awaiting = st.get("_awaiting")
    have_service = bool(st.get("service_query") or st.get("code"))
    have_zip = bool(st.get("zipcode"))

    # 0) ZIP detection anywhere
    zip_match = re.search(r"\b(\d{5})\b", msg)
    if zip_match:
        z = zip_match.group(1)
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
        "paying cash", "pay cash", "cash pay"
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
        # Direct map matches
        for k, v in carrier_map.items():
            if k in ml:
                return v
        
        # Stopword removal to avoid "I Have Insurance" as a carrier
        clean = m
        for stop in ["i have", "i use", "use", "with", "insurance", "my", "have", "paying"]:
            clean = re.sub(r'\b' + re.escape(stop) + r'\b', '', clean, flags=re.IGNORECASE)

        clean = clean.strip()
        tokens = re.findall(r"[a-zA-Z]+", clean)

        # 1 word carrier candidate (must be length >=3)
        if len(tokens) == 1 and len(tokens[0]) >= 3:
            return tokens[0].title()
        
        # 2-3 word carrier
        if 2 <= len(tokens) <= 3:
            return " ".join([t.title() for t in tokens])
            
        return None

    # 1) If awaiting ZIP
    if awaiting == "zip":
        return {"mode": "clarify", "clarifying_question": "What’s your 5-digit ZIP code?"}

    # 2) If awaiting payment
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

        payer_like = extract_carrier(msg)
        # If user mentions insurance OR provides a carrier
        if any(t in msg_l for t in insurance_terms) or payer_like:
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "insurance",
                "payer_like": payer_like, # might be None if they just said "insurance"
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": False,
            }

        return {"mode": "clarify", "clarifying_question": "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"}

    # 3) Session update (mid-flow)
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

    # 4) New question / Inferred Service
    inferred_service = infer_service_query_from_message(msg)
    if inferred_service and not have_zip:
        return {"mode": "price", "service_query": inferred_service, "clarifying_question": None}

    # 5) Fallback LLM
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
    if "colonoscopy" in msg: return "colonoscopy"
    if "mammogram" in msg or "mammo" in msg: return "mammogram"
    if "ultrasound" in msg: return "ultrasound"
    if "cat scan" in msg or "ct scan" in msg: return "ct scan"
    if "mri" in msg: return "mri"
    if "x-ray" in msg or "xray" in msg: return "x-ray"
    if "blood test" in msg or "lab test" in msg: return "lab test"
    if "office visit" in msg or "doctor visit" in msg: return "office visit"
    return None

def should_force_price_mode(message: str, merged: Dict[str, Any]) -> bool:
    if not INTENT_OVERRIDE_FORCE_PRICE_ENABLED:
        return False
    if (merged or {}).get("service_query") and (merged or {}).get("zipcode") and not (merged or {}).get("payment_mode"):
        return True
    if (merged or {}).get("_awaiting") in {"zip", "payment"}:
        return True
    msg = (message or "").lower()
    return any(kw in msg for kw in INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS)

def apply_intent_override_if_needed(intent: Dict[str, Any], message: str, merged: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    if should_force_price_mode(message, merged):
        intent["mode"] = "price"
        intent["intent_overridden"] = True
        if not (merged.get("service_query") or "").strip():
            inferred = infer_service_query_from_message(message)
            if inferred:
                merged["service_query"] = inferred
    else:
        intent["intent_overridden"] = False
    return intent

def message_contains_zip(message: str) -> bool:
    msg = (message or "").strip()
    tokens = [t.strip(",.()[]{}") for t in msg.split()]
    return any(len(t) == 5 and t.isdigit() for t in tokens)

def message_contains_payment_info(message: str) -> bool:
    msg = (message or "").lower()
    cash_terms = ["cash", "self pay", "out of pocket", "uninsured"]
    ins_terms = ["insurance", "insured", "copay", "coinsurance"]
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
    if not q: return None

    # 1) Canonical lookup
    rows = await conn.fetch(
        """
        SELECT code_type, code
        FROM public.services
        WHERE (cpt_explanation ILIKE '%' || $1 || '%'
            OR service_description ILIKE '%' || $1 || '%'
            OR patient_summary ILIKE '%' || $1 || '%')
        ORDER BY code_type, code
        LIMIT 5
        """, q
    )
    if rows:
        return rows[0]["code_type"], rows[0]["code"]

    # 2) Fallback to staging table description
    srows = await conn.fetch(
        """
        SELECT code_type, code, COUNT(*) AS n
        FROM public.stg_hospital_rates
        WHERE code_type IS NOT NULL AND code IS NOT NULL
          AND (service_description ILIKE '%' || $1 || '%' OR code ILIKE '%' || $1 || '%')
        GROUP BY code_type, code
        ORDER BY n DESC, code_type, code
        LIMIT 5
        """, q
    )
    if srows:
        return srows[0]["code_type"], srows[0]["code"]
    return None

async def get_service_ids(conn: asyncpg.Connection, service_query: str, code_type: Optional[str] = None, code: Optional[str] = None, limit: int = 25) -> List[int]:
    q = (service_query or "").strip()
    ids: List[int] = []
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
            """, q, limit
        )
        ids.extend([int(r["id"]) for r in rows if r and r["id"] is not None])
    
    if code_type and code:
        row = await conn.fetchrow("SELECT id FROM public.services WHERE code_type = $1 AND code = $2 LIMIT 1", code_type, code)
        if row: ids.append(int(row["id"]))
    
    # dedupe
    out = []
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
    Returns nearest facilities by distance to ZIP centroid.
    Correctly selects 'standard_charge_discounted_cash' for cash mode
    and 'standard_charge_negotiated_dollar' for insurance mode.
    """
    radius_array = radius_array or [10, 25, 50, 100]
    z = await conn.fetchrow("SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1", zipcode)
    if not z:
        return []

    zlat, zlon = float(z["latitude"]), float(z["longitude"])
    service_ids = await get_service_ids(conn, service_query=service_query, code_type=code_type, code=code, limit=25)
    if not service_ids:
        return []

    last_rows: List[Dict[str, Any]] = []
    for r in radius_array:
        if payment_mode.lower() == "cash":
            # CASH: Get MIN of standard_charge_discounted_cash
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
                    nr.standard_charge_discounted_cash,
                    nr.estimated_amount,
                    nr.standard_charge_gross
                FROM public.hospitals h
                LEFT JOIN LATERAL (
                    SELECT
                        MIN(standard_charge_discounted_cash) AS standard_charge_discounted_cash,
                        MIN(estimated_amount) AS estimated_amount,
                        MIN(standard_charge_gross) AS standard_charge_gross
                    FROM public.negotiated_rates
                    WHERE hospital_id = h.id AND service_id = ANY($3::int[])
                ) nr ON TRUE
                WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
                  AND (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                  )) <= $4
                ORDER BY distance_miles ASC
                LIMIT $5
                """,
                zlat, zlon, service_ids, r, limit,
            )
        else:
            # INSURANCE: Filter by payer/plan, prefer standard_charge_negotiated_dollar
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
                    pick.standard_charge_negotiated_dollar,
                    pick.estimated_amount,
                    pick.standard_charge_discounted_cash,
                    pick.payer_name,
                    pick.plan_name
                FROM public.hospitals h
                LEFT JOIN LATERAL (
                    SELECT
                        nr.standard_charge_negotiated_dollar,
                        nr.estimated_amount,
                        nr.standard_charge_discounted_cash,
                        ip.payer_name,
                        ip.plan_name
                    FROM public.negotiated_rates nr
                    JOIN public.insurance_plans ip ON ip.id = nr.plan_id
                    WHERE nr.hospital_id = h.id
                      AND nr.service_id = ANY($3::int[])
                      AND ip.payer_name ILIKE $4
                      AND ip.plan_name ILIKE $5
                    ORDER BY
                        nr.standard_charge_negotiated_dollar NULLS LAST,
                        nr.estimated_amount NULLS LAST,
                        nr.standard_charge_discounted_cash NULLS LAST
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


async def get_nearby_hospitals(conn: asyncpg.Connection, zipcode: str, limit: int = 5) -> List[Dict[str, Any]]:
    z = await conn.fetchrow(
        "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1 LIMIT 1",
        zipcode,
    )
    zlat = float(z["latitude"]) if z and z["latitude"] is not None else None
    zlon = float(z["longitude"]) if z and z["longitude"] is not None else None

    if zlat is not None and zlon is not None:
        rows = await conn.fetch(
            """
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
            """,
            limit, zlat, zlon,
        )
        return [dict(r) for r in rows]
    else:
        rows = await conn.fetch(
            "SELECT hospital_name, address, state, zipcode, phone FROM public.hospital_details LIMIT $1", 
            limit
        )
        return [dict(r) for r in rows]


def estimate_cost_range(service_query: str, payment_mode: str) -> str:
    system = "You output ONLY a short numeric range ($X–$Y) for a U.S. healthcare service. Be conservative."
    user = json.dumps({"service": service_query, "payment_mode": payment_mode})
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=12,
        )
        txt = (resp.choices[0].message.content or "").strip()
        if "$" in txt and any(ch.isdigit() for ch in txt):
            return txt
        return "$1,000–$3,000"
    except Exception:
        return "$1,000–$3,000"


# ----------------------------
# Formatting
# ----------------------------
def _pick_price_fields(row: Dict[str, Any]) -> Optional[float]:
    # Specific priorities based on your CSV structure
    keys = [
        "standard_charge_negotiated_dollar",   # Explicit insurance rate
        "standard_charge_discounted_cash",     # Explicit cash rate
        "negotiated_dollar",
        "standard_charge_cash",
        "estimated_amount",
        "standard_charge_gross",
        "standard_charge"
    ]
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                # remove comma, convert
                clean = v.replace(",", "").replace("$", "")
                if clean.replace(".", "", 1).isdigit():
                    return float(clean)
            except: pass
    return None

def _pick_hospital_name(row: Dict[str, Any]) -> str:
    return (row.get("hospital_name") or row.get("name") or "Unknown facility").strip()

def _pick_phone(row: Dict[str, Any]) -> str:
    return (row.get("phone") or "").strip()

def _pick_address(row: Dict[str, Any]) -> str:
    addr = row.get("address") or row.get("street_address") or ""
    city = row.get("city") or ""
    state = row.get("state") or ""
    z = row.get("zipcode") or row.get("zip") or ""
    parts = [p for p in [addr, city, state, z] if p]
    return ", ".join(parts).strip()

def _format_money(v: Optional[float]) -> str:
    if v is None: return ""
    return f"${v:,.0f}"

def build_facility_block(
    service_query: str,
    payment_mode: str,
    priced_results: List[Dict[str, Any]],
    fallback_hospitals: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    facilities: List[Dict[str, Any]] = []
    seen = set()

    # 1) Priced
    for r in priced_results:
        name = _pick_hospital_name(r)
        key = name.lower().strip()
        if key in seen: continue
        seen.add(key)
        facilities.append(r)
        if len(facilities) >= MIN_FACILITIES_TO_DISPLAY: break

    # 2) Fallback
    for h in fallback_hospitals:
        if len(facilities) >= MIN_FACILITIES_TO_DISPLAY: break
        name = _pick_hospital_name(h)
        key = name.lower().strip()
        if key in seen: continue
        seen.add(key)
        facilities.append(h)

    if not facilities:
        return ("I couldn’t find hospitals near that ZIP code. Try another ZIP or search radius.", [])

    est_range = estimate_cost_range(service_query, payment_mode)
    
    bullets = []
    bullets.append(f"- Colonoscopies can be **screening** (preventive) or **diagnostic** (symptoms/abnormal finding).")
    bullets.append(f"- Facility setting matters: **outpatient endoscopy centers** often differ from **hospital outpatient** pricing.")
    bullets.append(f"- If a biopsy or polyp removal happens, total cost can increase.")

    lines = []
    lines.extend(bullets)
    lines.append("")
    lines.append(f"Here are nearby options for **{service_query or 'colonoscopy'}** ({payment_mode}):")

    for i, f in enumerate(facilities[:MIN_FACILITIES_TO_DISPLAY], start=1):
        name = _pick_hospital_name(f)
        addr = _pick_address(f)
        phone = _pick_phone(f)
        dist = f.get("distance_miles")
        dist_txt = f" ({float(dist):.1f} mi)" if dist is not None else ""

        price = _pick_price_fields(f)
        if price is not None:
            price_txt = _format_money(price)
            price_note = "DB price"
        else:
            price_txt = est_range
            price_note = "ESTIMATE (no DB price yet)"

        detail = " | ".join([p for p in [addr, f"Tel: {phone}"] if p]) or "Contact info not available."
        
        lines.append(f"{i}) **{name}**{dist_txt}")
        lines.append(f"   - {detail}")
        lines.append(f"   - Price: **{price_txt}** ({price_note})")

    lines.append("")
    lines.append("Confirm with the facility and your insurer.")

    ui_payload = []
    for f in facilities[:MIN_FACILITIES_TO_DISPLAY]:
        p = _pick_price_fields(f)
        ui_payload.append({
            "hospital_name": _pick_hospital_name(f),
            "address": _pick_address(f),
            "phone": _pick_phone(f),
            "distance_miles": f.get("distance_miles"),
            "price": p,
            "estimated_range": None if p is not None else est_range,
            "price_is_estimate": p is None,
        })

    return "\n".join(lines), ui_payload


# ----------------------------
# Refiners
# ----------------------------
_refiners_cache: Optional[dict] = None
_refiners_cache_loaded_at: float = 0.0

async def get_refiners(conn: asyncpg.Connection) -> dict:
    global _refiners_cache, _refiners_cache_loaded_at
    now = time.time()
    if _refiners_cache and (now - _refiners_cache_loaded_at) < REFINERS_CACHE_TTL_SECONDS:
        return _refiners_cache
    
    # Try DB load (simplified logic)
    try:
        rows = await conn.fetch("SELECT id, title, keywords, require_choice_before_pricing, preview_code_type, preview_code, question_text FROM public.service_refiner WHERE is_active = true")
        if not rows:
            data = refiners_registry()
        else:
            # Reconstruct refiners logic here if needed, or use python registry as fallback
            # For brevity, assuming registry is main source if DB empty or basic fetch
            data = refiners_registry()
    except:
        data = refiners_registry()

    _refiners_cache = data
    _refiners_cache_loaded_at = now
    return data

def match_refiner(service_query: str, refiners_doc: dict) -> Optional[dict]:
    q = (service_query or "").strip().lower()
    if not q: return None
    for ref in refiners_doc.get("refiners", []):
        kws = [(k or "").strip().lower() for k in (ref.get("match", {}).get("keywords") or [])]
        if any(k and k in q for k in kws):
            return ref
    return None

def apply_refiner_choice(message: str, merged: dict, refiner: Optional[dict]) -> dict:
    if not refiner: return merged
    choice_key = (message or "").strip()
    for ch in refiner.get("choices", []):
        if str(ch.get("key")) == choice_key:
            merged["code_type"] = ch.get("code_type")
            merged["code"] = ch.get("code")
            merged["refiner_id"] = refiner.get("id")
            merged["refiner_choice"] = choice_key
            return merged
    return merged

def get_refinement_prompt(refiner: dict) -> str:
    lines = [refiner.get("question", "").strip(), ""]
    for ch in refiner.get("choices", []):
        lines.append(f"{ch.get('key')}) {ch.get('label')}")
    lines.append("")
    lines.append("Reply with the number that fits best.")
    return "\n".join([l for l in lines if l])


# ----------------------------
# Main Endpoint
# ----------------------------
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    require_auth(request)
    if not rate_limit_ok(request.client.host if request.client else "unknown"):
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
                _normalize_payment_mode(merged)
                
                intent = apply_intent_override_if_needed(intent, req.message, merged, session_id)
                mode = intent.get("mode") or "hybrid"

                if RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION and mode in ["price", "hybrid"]:
                     if not (merged.get("_awaiting") in {"zip", "payment", "payer"}):
                        # check if message really looks like a new question
                        inferred = infer_service_query_from_message(req.message)
                        if inferred and any(kw in (req.message or "").lower() for kw in INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS):
                            reset_gating_fields_for_new_price_question(req.message, merged)

                # Refiner Check
                refiners_doc = await get_refiners(conn)
                refiner = match_refiner(merged.get("service_query") or "", refiners_doc)
                merged = apply_refiner_choice(req.message, merged, refiner)

                if mode == "general":
                    system = "You are CostSavvy.health. Answer clearly. Avoid medical advice."
                    parts = []
                    for chunk in stream_llm_to_sse(system, req.message, parts): yield chunk
                    full_answer = "".join(parts).strip()
                    await save_message(conn, session_id, "assistant", full_answer, {"mode": "general"})
                    await update_session_state(conn, session_id, merged)
                    yield sse({"type": "final", "used_web_search": False})
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
                    
                    _normalize_payment_mode(merged)
                    
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

                    # Refiner: require choice?
                    if refiner and refiner.get("require_choice_before_pricing") is True and not merged.get("refiner_choice"):
                        msg = get_refinement_prompt(refiner)
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {})
                        await update_session_state(conn, session_id, merged)
                        yield sse({"type": "final"})
                        return

                    # Resolve Code
                    if not (merged.get("code_type") and merged.get("code")):
                        resolved = await resolve_service_code(conn, merged)
                        if resolved:
                            merged["code_type"], merged["code"] = resolved

                    # Missing Service?
                    if not (merged.get("service_query") or "").strip() and not (merged.get("code_type") and merged.get("code")):
                         msg = "What service are you pricing (e.g., colonoscopy, MRI, x-ray)?"
                         yield sse({"type": "delta", "text": msg})
                         await save_message(conn, session_id, "assistant", msg, {})
                         await update_session_state(conn, session_id, merged)
                         yield sse({"type": "final"})
                         return

                    # Execute Pricing
                    results, used_radius = await price_lookup_progressive(
                        conn,
                        merged["zipcode"],
                        merged.get("code_type"),
                        merged.get("code"),
                        merged.get("service_query") or "",
                        merged.get("payer_like"),
                        merged.get("plan_like"),
                        merged.get("payment_mode") or "cash",
                    )

                    try:
                        nearby = await get_nearby_hospitals(conn, merged["zipcode"], limit=5)
                    except: nearby = []

                    txt, payload = build_facility_block(
                        merged.get("service_query") or "this service",
                        merged.get("payment_mode") or "cash",
                        results,
                        nearby
                    )
                    
                    yield sse({
                        "type": "results", 
                        "results": results[:25], 
                        "facilities": payload,
                        "state": {k: merged.get(k) for k in ["zipcode", "payment_mode", "payer_like", "service_query"]}
                    })
                    yield sse({"type": "delta", "text": txt})
                    
                    await save_message(conn, session_id, "assistant", txt, {"mode": "price_result", "count": len(results)})
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, used_radius, len(results), False, txt)
                    yield sse({"type": "final", "used_web_search": False})
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