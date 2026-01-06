# app/main.py
import os
import json
import re
import uuid
import time
import logging
from typing import Optional, Any, Dict, List, Tuple
from decimal import Decimal
from datetime import datetime, date
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
MIN_FACILITIES_TO_DISPLAY = int(os.getenv("MIN_FACILITIES_TO_DISPLAY", "5"))
REFINERS_CACHE_TTL_SECONDS = int(os.getenv("REFINERS_CACHE_TTL_SECONDS", "300"))

INTENT_OVERRIDE_FORCE_PRICE_ENABLED = os.getenv("INTENT_OVERRIDE_FORCE_PRICE_ENABLED", "true").lower() in ("1", "true", "yes", "y", "on")
INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS = [
    s.strip().lower()
    for s in os.getenv("INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS", "cost,price,how much,pricing,estimate,rate,charge,fee").split(",")
    if s.strip()
]
RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION = os.getenv("RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION", "true").lower() in ("1", "true", "yes", "y", "on")
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
    return f"data: {json.dumps(obj, default=_json_default)}\n\n"

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
        session_id, json.dumps({}),
    )
    return session_id, {}

async def save_message(conn: asyncpg.Connection, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
    await conn.execute(
        "INSERT INTO public.chat_message (session_id, role, content, metadata) VALUES ($1,$2,$3,$4::jsonb)",
        session_id, role, content, json.dumps(metadata or {}),
    )

async def update_session_state(conn: asyncpg.Connection, session_id: str, state: Dict[str, Any]):
    await conn.execute(
        "UPDATE public.chat_session SET session_state = $2::jsonb, last_seen = now() WHERE id = $1",
        session_id, json.dumps(state),
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
        session_id, question, json.dumps(intent), used_radius, result_count, used_web, answer_text,
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
Notes:
- If user says "cash price", "self-pay", "out of pocket" => payment_mode="cash" and cash_only=true.
- If user mentions insurance or a carrier name => payment_mode="insurance".
"""

def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in [
        "zipcode", "radius_miles", "payer_like", "plan_like",
        "payment_mode", "service_query", "code_type", "code",
        "variant_cpt_code", "variant_name", "cash_only",
    ]:
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
    if _is_zip_only_message(message) and (state or {}).get("service_query"):
        return {
            "mode": "price",
            "zipcode": message.strip(),
            "service_query": state.get("service_query"),
            "code_type": state.get("code_type"),
            "code": state.get("code"),
            "cash_only": state.get("cash_only", False),
        }
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You extract intent for healthcare Q&A and price lookup. Be conservative."},
                {"role": "system", "content": INTENT_RULES},
                {"role": "user", "content": json.dumps({"message": message, "session_state": state})},
            ],
            timeout=10,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Intent extraction failed: {e}")
        return {"mode": "clarify", "clarifying_question": "What 5-digit ZIP code should I search near?"}

# ----------------------------
# Intent override + deterministic service inference
# ----------------------------
def infer_service_query_from_message(message: str) -> Optional[str]:
    msg = (message or "").lower()
    if "colonoscopy" in msg: return "colonoscopy"
    if "mammogram" in msg or "mammo" in msg: return "mammogram"
    if "ultrasound" in msg: return "ultrasound"
    if "ct scan" in msg or "cat scan" in msg: return "ct scan"
    if "mri" in msg: return "mri"
    if "x-ray" in msg or "xray" in msg: return "x-ray"
    if "lab test" in msg or "blood test" in msg: return "lab test"
    if "office visit" in msg: return "office visit"
    return None

def should_force_price_mode(message: str, merged: Dict[str, Any]) -> bool:
    if not INTENT_OVERRIDE_FORCE_PRICE_ENABLED:
        return False
    msg = (message or "").lower()
    return any(kw in msg for kw in INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS)

def apply_intent_override_if_needed(intent: Dict[str, Any], message: str, merged: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    if should_force_price_mode(message, merged):
        prev = intent.get("mode")
        if prev != "price":
            logger.info("Intent override applied: forcing mode=price", extra={"session_id": session_id})
            intent["mode"] = "price"
            if not (merged.get("service_query") or "").strip():
                inferred = infer_service_query_from_message(message)
                if inferred:
                    merged["service_query"] = inferred
    return intent

def message_contains_zip(message: str) -> bool:
    msg = (message or "").strip()
    tokens = [t.strip(",.()[]{}") for t in msg.split()]
    return any(len(t) == 5 and t.isdigit() for t in tokens)

def message_contains_payment_info(message: str) -> bool:
    msg = (message or "").lower()
    cash_terms = ["cash", "self pay", "self-pay", "out of pocket", "uninsured", "no insurance"]
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
# Service variants (CPT subtypes)
# ----------------------------
# ----------------------------
# Service query normalization (typos / aliases)
# ----------------------------
SERVICE_TYPO_MAP = {
    # common misspellings
    "colonscopy": "colonoscopy",
    "colonsocpy": "colonoscopy",
    "colonoscopyy": "colonoscopy",
    "colonosopy": "colonoscopy",
    "colonosocpy": "colonoscopy",
    "colonscopy": "colonoscopy",
}

def normalize_service_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return SERVICE_TYPO_MAP.get(q, q)

async def fetch_service_variants(conn: asyncpg.Connection, service_query: str, limit: int = 10) -> List[Dict[str, Any]]:
    q = (service_query or "").strip()
    if not q:
        return []
    q_l = q.lower()
    norm_map = {"colonscopy": "colonoscopy", "colonscoppy": "colonoscopy", "colonocopy": "colonoscopy"}
    q_l = norm_map.get(q_l, q_l)
    pat = f"%{q_l}%"
    rows = await conn.fetch(
        """
        SELECT parent_service, cpt_code, variant_name, patient_summary, is_preventive
        FROM public.service_variants
        WHERE LOWER(variant_name) ILIKE $1
        OR LOWER(parent_service) ILIKE $1
        ORDER BY parent_service, cpt_code
        LIMIT $2
        """, parent, limit,
    )
    return [dict(r) for r in rows]

def _variant_question(service_query: str, variants: List[Dict[str, Any]]) -> str:
    svc = (service_query or "this service").strip()
    lines = [f"{svc.title()} can be billed in different ways, and the price can vary by type."]
    lines.append("Here are the common options:")
    lines.append("")
    for idx, v in enumerate(variants, start=1):
        name = (v.get("variant_name") or "").strip() or f"Option {idx}"
        summary = (v.get("patient_summary") or "").strip()
        if summary and len(summary) > 240:
            summary = summary[:237].rstrip() + "..."
        if summary:
            lines.append(f"{idx}) **{name}** — {summary}")
        else:
            lines.append(f"{idx}) **{name}**")
    lines.append("")
    lines.append("Which option best matches what you need? Reply with the number (or the CPT code).")
    return "\n".join(lines)

def apply_variant_choice(message: str, merged: Dict[str, Any]) -> Dict[str, Any]:
    if (merged or {}).get("_awaiting") != "variant":
        return merged
    opts = merged.get("_variant_options") or []
    msg = (message or "").strip()
    # Direct CPT code
    if re.fullmatch(r"\d{5}", msg):
        for o in opts:
            if str(o.get("cpt_code")) == msg:
                merged.update({
                    "variant_cpt_code": msg,
                    "variant_name": o.get("variant_name"),
                    "code_type": "CPT",
                    "code": msg,
                })
                merged.pop("_awaiting", None)
                merged.pop("_variant_options", None)
                return merged
    # Numeric choice
    if msg.isdigit():
        idx = int(msg)
        if 1 <= idx <= len(opts):
            chosen = opts[idx - 1]
            cpt = str(chosen.get("cpt_code") or "").strip()
            if cpt:
                merged.update({
                    "variant_cpt_code": cpt,
                    "variant_name": chosen.get("variant_name"),
                    "code_type": "CPT",
                    "code": cpt,
                })
                merged.pop("_awaiting", None)
                merged.pop("_variant_options", None)
                return merged
    return merged

# ----------------------------
# DB: resolve code + price lookup
# ----------------------------
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
        q,
    )
    if not rows:
        return None
    return rows[0]["code_type"], rows[0]["code"]

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
        zipcode, code_type, code, payer_like, plan_like, radius_array,
    )
    return [dict(r) for r in rows]

async def price_lookup_progressive(
    conn: asyncpg.Connection,
    zipcode: str,
    code_type: str,
    code: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    last_results: List[Dict[str, Any]] = []
    used_max_radius: Optional[int] = None
    for r in PRICE_RADIUS_ATTEMPTS:
        radius_array = [x for x in PRICE_RADIUS_ATTEMPTS if x <= r]
        try:
            res = await price_lookup_v3(conn, zipcode, code_type, code, payer_like, plan_like, radius_array=radius_array)
        except Exception as e:
            logger.warning(f"Price lookup failed at radius {r}: {e}")
            res = []
        if res:
            last_results = res
            used_max_radius = r
            if len(res) >= MIN_FACILITIES_TO_DISPLAY:
                return res, r
    return last_results, used_max_radius

# ----------------------------
# Nearby hospitals
# ----------------------------
async def get_nearby_hospitals(conn: asyncpg.Connection, zipcode: str, limit: int = 5) -> List[Dict[str, Any]]:
    z = await conn.fetchrow(
        "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1 LIMIT 1", zipcode,
    )
    zlat = float(z["latitude"]) if z and z["latitude"] is not None else None
    zlon = float(z["longitude"]) if z and z["longitude"] is not None else None
    if zlat is not None and zlon is not None:
        q = """
        SELECT h.hospital_name AS hospital_name,
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
        return [dict(r) for r in rows]
    q2 = """
    SELECT h.hospital_name AS hospital_name,
        h.address AS address,
        h.state AS state,
        h.zipcode AS zipcode,
        h.phone AS phone,
        NULL::float AS distance_miles
    FROM public.hospital_details h
    ORDER BY h.state, h.zipcode, h.hospital_name
    LIMIT $1
    """
    rows = await conn.fetch(q2, limit)
    return [dict(r) for r in rows]

# ----------------------------
# Estimated range
# ----------------------------
def estimate_cost_range(service_query: str, payment_mode: str) -> str:
    system = "You output ONLY a short numeric range. Format: '$X–$Y'. No extra text."
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
# Facility formatting — ✅ Schema-Aligned
# ----------------------------
def _pick_price_fields(row: Dict[str, Any]) -> Optional[float]:
    # ✅ Use your real column names from hospital_details & negotiated_rates
    for k in [
        "standard_charge_discounted_cash",  # cash/self-pay
        "standard_charge_cash",             # negotiated_rates
        "standard_charge_negotiated_dollar", # insured
        "negotiated_dollar",               # alias
        "estimated_amount",
        "standard_charge_gross"
    ]:
        v = row.get(k)
        if isinstance(v, (int, float, Decimal)):
            try:
                return float(v)
            except:
                continue
    return None

def _pick_hospital_name(row: Dict[str, Any]) -> str:
    return (row.get("hospital_name") or row.get("name") or "Unknown facility").strip()

def _pick_phone(row: Dict[str, Any]) -> str:
    return (row.get("phone") or "").strip()

def _pick_address(row: Dict[str, Any]) -> str:
    addr = row.get("address") or ""
    city = "" or ""
    state = row.get("state") or ""
    z = row.get("zipcode") or ""
    parts = [p for p in [addr, city, state, z] if p]
    return ", ".join(parts).strip()

def _format_money(v: Optional[float]) -> str:
    if v is None:
        return ""
    try:
        return f"${v:,.0f}"
    except Exception:
        return f"${v}"

def build_facility_block(
    service_query: str,
    payment_mode: str,
    priced_results: List[Dict[str, Any]],
    fallback_hospitals: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    facilities: List[Dict[str, Any]] = []
    seen = set()
    for r in priced_results:
        name = _pick_hospital_name(r).lower()
        if name in seen:
            continue
        seen.add(name)
        facilities.append(r)
        if len(facilities) >= MIN_FACILITIES_TO_DISPLAY:
            break
    for h in fallback_hospitals:
        if len(facilities) >= MIN_FACILITIES_TO_DISPLAY:
            break
        name = (h.get("hospital_name") or "").strip().lower()
        if name in seen:
            continue
        seen.add(name)
        facilities.append(h)

    if not facilities:
        return ("I don’t yet have facility records for that area.", [])

    est_range = estimate_cost_range(service_query or "this service", payment_mode or "cash")

    # ✅ Generic bullets — no colonoscopy hardcode
    svc_title = (service_query or "this procedure").title()
    bullets = []
    bullets.append(f"- {svc_title} can be **screening/preventive** or **diagnostic/therapeutic**, affecting cost.")
    bullets.append(f"- Facility setting matters: **outpatient centers** often differ from **hospital outpatient** pricing.")
    bullets.append(f"- Additional services (e.g., biopsy, removal) may increase the total.")

    lines = []
    lines.extend(bullets)
    lines.append("")
    lines.append(f"Here are nearby options for **{service_query or 'this service'}** ({payment_mode}):")

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
        detail_bits = [p for p in [addr, f"Tel: {phone}" if phone else None] if p]
        detail = " | ".join(detail_bits) if detail_bits else "Contact info not available."
        lines.append(f"{i}) **{name}**{dist_txt}")
        lines.append(f"   - {detail}")
        lines.append(f"   - Price: **{price_txt}** ({price_note})")
    lines.append("")
    lines.append("Confirm with the facility and your insurer.")

    ui_payload = []
    for f in facilities[:MIN_FACILITIES_TO_DISPLAY]:
        ui_payload.append({
            "hospital_name": _pick_hospital_name(f),
            "address": _pick_address(f),
            "phone": _pick_phone(f),
            "distance_miles": f.get("distance_miles"),
            "price": _pick_price_fields(f),
            "estimated_range": None if _pick_price_fields(f) is not None else est_range,
            "price_is_estimate": _pick_price_fields(f) is None,
        })
    return "\n".join(lines), ui_payload

# ----------------------------
# Service refiners
# ----------------------------
_refiners_cache: Optional[dict] = None
_refiners_cache_loaded_at: float = 0.0

def _norm(s: str) -> str:
    return (s or "").strip().lower()

async def load_refiners_from_db(conn: asyncpg.Connection) -> dict:
    rows = await conn.fetch("""
        SELECT id, title, keywords, require_choice_before_pricing,
               preview_code_type, preview_code, question_text
        FROM public.service_refiner
        WHERE is_active = true
        ORDER BY id
    """)
    if not rows:
        return {"version": 1, "refiners": []}
    ids = [r["id"] for r in rows]
    crows = await conn.fetch("""
        SELECT refiner_id, choice_key, choice_label, code_type, code, sort_order
        FROM public.service_refiner_choice
        WHERE is_active = true AND refiner_id = ANY($1::text[])
        ORDER BY refiner_id, sort_order, choice_key
    """, ids)
    choices_by_ref: Dict[str, List[dict]] = {}
    for c in crows:
        choices_by_ref.setdefault(c["refiner_id"], []).append({
            "key": c["choice_key"], "label": c["choice_label"],
            "code_type": c["code_type"], "code": c["code"]
        })
    refiners = []
    for r in rows:
        preview = None
        if r["preview_code_type"] and r["preview_code"]:
            preview = {"code_type": r["preview_code_type"], "code": r["preview_code"]}
        refiners.append({
            "id": r["id"],
            "title": r["title"],
            "match": {"keywords": list(r["keywords"] or [])},
            "require_choice_before_pricing": bool(r["require_choice_before_pricing"]),
            "preview_code": preview,
            "question": r["question_text"],
            "choices": choices_by_ref.get(r["id"], []),
        })
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
            merged.update({
                "code_type": ch.get("code_type"),
                "code": ch.get("code"),
                "refiner_id": refiner.get("id"),
                "refiner_choice": choice_key,
            })
            return merged
    return merged

def maybe_apply_preview_code(merged: dict, refiner: Optional[dict]) -> dict:
    if not refiner or refiner.get("require_choice_before_pricing") is True:
        return merged
    if merged.get("code_type") and merged.get("code"):
        return merged
    preview = refiner.get("preview_code")
    if preview and preview.get("code_type") and preview.get("code"):
        merged.update({
            "code_type": preview["code_type"],
            "code": preview["code"],
            "refiner_id": refiner.get("id"),
        })
    return merged

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

                intent = await extract_intent(req.message, state)
                merged = merge_state(state, intent)
                _normalize_payment_mode(merged)
                merged = apply_variant_choice(req.message, merged)
                intent = apply_intent_override_if_needed(intent, req.message, merged, session_id)
                mode = intent.get("mode") or "hybrid"

                if RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION and mode in ["price", "hybrid"] and should_force_price_mode(req.message, merged):
                    reset_gating_fields_for_new_price_question(req.message, merged)

                refiners_doc = await get_refiners(conn)
                refiner = match_refiner(merged.get("service_query") or "", refiners_doc)
                merged = apply_refiner_choice(req.message, merged, refiner)
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
                # PRICE / HYBRID mode
                # ----------------------------
                if mode in ["price", "hybrid"]:
                    # Gate 1: ZIP
                    zipcode = merged.get("zipcode")
                    if not zipcode:
                        msg = "What’s your 5-digit ZIP code?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_zip", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # Gate 2: payment mode
                    payment_mode = merged.get("payment_mode")
                    if not payment_mode:
                        msg = "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_payment", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # Gate 3: insurance → payer
                    if payment_mode == "insurance" and not (merged.get("payer_like") or "").strip():
                        msg = "Which insurance carrier should I match prices for (e.g., Aetna, UnitedHealthcare, Blue Cross)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_payer", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # 👇 CRITICAL: Save state BEFORE variant prompt
                    variants = []
                    try:
                        variants = await fetch_service_variants(conn, merged.get("service_query") or "", limit=10)
                    except Exception as e:
                        logger.warning(f"Service variants lookup failed: {e}")

                    sq = (merged.get("service_query") or "").strip().lower()
                    sq = {"colonscopy": "colonoscopy", "colonscoppy": "colonoscopy", "colonocopy": "colonoscopy"}.get(sq, sq)
                    if sq:
                        variants = [v for v in variants if sq in (str(v.get("variant_name") or "").lower())]

                    if len(variants) >= 2:
                        merged["_awaiting"] = "variant"
                        merged["_variant_options"] = [
                            {"cpt_code": str(v.get("cpt_code") or "").strip(), "variant_name": (v.get("variant_name") or "").strip()}
                            for v in variants[:10] if v.get("cpt_code")
                        ]
                        # ✅ Save state BEFORE yielding prompt
                        await update_session_state(conn, session_id, merged)

                        msg = _variant_question(merged.get("service_query") or "this service", variants[:10])
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_variant", "intent": intent})
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # Refiners
                    if refiner and refiner.get("require_choice_before_pricing") is True and not merged.get("refiner_choice"):
                        msg = get_refinement_prompt(refiner)
                        await update_session_state(conn, session_id, merged)
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_variant", "refiner_id": refiner.get("id"), "intent": intent})
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # Resolve code
                    if not (merged.get("code_type") and merged.get("code")):
                        resolved = await resolve_service_code(conn, merged)
                        if resolved:
                            merged["code_type"], merged["code"] = resolved

                    if not (merged.get("service_query") or "").strip() and not (merged.get("code_type") and merged.get("code")):
                        msg = "What service are you pricing (e.g., MRI brain, chest x-ray, office visit)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_service", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    if not (merged.get("code_type") and merged.get("code")):
                        msg = "I need more detail on the exact service. What exactly is being ordered?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_service_detail", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # --- PRICED LOOKUP ---
                    results, used_max_radius = await price_lookup_progressive(
                        conn, merged["zipcode"], merged["code_type"], merged["code"],
                        merged.get("payer_like"), merged.get("plan_like"),
                    )
                    nearby_hospitals = []
                    try:
                        nearby_hospitals = await get_nearby_hospitals(conn, merged["zipcode"], limit=MIN_FACILITIES_TO_DISPLAY)
                    except Exception as e:
                        logger.warning(f"Nearby hospitals lookup failed: {e}")

                    facility_text, facility_payload = build_facility_block(
                        service_query=merged.get("service_query") or "this service",
                        payment_mode=merged.get("payment_mode") or "cash",
                        priced_results=results,
                        fallback_hospitals=nearby_hospitals,
                    )

                    yield sse({
                        "type": "results",
                        "results": results[:25],
                        "facilities": facility_payload,
                        "state": {k: merged.get(k) for k in [
                            "zipcode", "payment_mode", "payer_like", "plan_like",
                            "service_query", "code_type", "code", "refiner_id", "refiner_choice"
                        ]},
                    })

                    yield sse({"type": "delta", "text": facility_text})
                    full_answer = facility_text
                    await save_message(conn, session_id, "assistant", full_answer, {
                        "mode": mode, "intent": intent, "result_count": len(results),
                        "used_max_radius": used_max_radius, "refiner_id": (refiner or {}).get("id"),
                    })
                    await update_session_state(conn, session_id, merged)
                    used_web = len(results) < MIN_DB_RESULTS_BEFORE_WEB
                    await log_query(conn, session_id, req.message, intent, used_max_radius, len(results), used_web, full_answer)
                    yield sse({"type": "final", "used_web_search": used_web})
                    return

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