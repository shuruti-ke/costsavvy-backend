# app/main.py

import os
import sys
import re
import json
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import databases
import sqlalchemy
from sqlalchemy import text

# -------- CONFIG & LOGGER --------
logger = logging.getLogger("uvicorn.error")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db")
database = databases.Database(DATABASE_URL)

# -------- MODELS --------
class ChatMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# -------- UTILS --------

def sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def is_zip_code(s: str) -> bool:
    return bool(re.fullmatch(r"\d{5}", s.strip()))

def infer_parent_service_key(query: str) -> Optional[str]:
    """Map free-text query to known parent_service keys (e.g., 'mri', 'ct', 'xray')."""
    query = query.lower()
    # Prioritize longer matches to avoid false positives (e.g., "ultrasound" not "sound")
    if "mri" in query or "magnetic resonance" in query:
        return "mri"
    if "ct scan" in query or "cat scan" in query or "computed tomography" in query:
        return "ct"
    if "x-ray" in query or "xray" in query:
        return "xray"
    if "ultrasound" in query or "sonogram" in query:
        return "ultrasound"
    if "mammogram" in query:
        return "mammogram"
    return None

def apply_service_variant_choice(user_input: str, state: Dict[str, Any], variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Applies variant choice (e.g., '2') to session state."""
    try:
        choice_idx = int(clean_text(user_input)) - 1
        if 0 <= choice_idx < len(variants):
            v = variants[choice_idx]
            state["variant_id"] = v["id"]
            state["code_type"] = "cpt"
            state["code"] = v["cpt_code"]
            state["variant_name"] = v["variant_name"]
            state["patient_summary"] = v["patient_summary"]
            state["is_preventive"] = v["is_preventive"]
            state["parent_service"] = v["parent_service"]
            # Clear awaiting flags
            state.pop("_awaiting", None)
            state.pop("_variant_choices", None)
            logger.info(f"Variant selected: {v['variant_name']} (CPT {v['cpt_code']})")
            return state
    except (ValueError, IndexError, KeyError):
        pass
    # Invalid choice — leave state unchanged (will reprompt)
    return state

def build_service_variant_prompt(label: str, variants: List[Dict[str, Any]]) -> str:
    """Ask user to choose a CPT-backed variant BEFORE collecting ZIP.

    Works for ANY service type (imaging, procedures, labs, visits, etc.).
    """
    lines: List[str] = []
    base = infer_parent_service_key(label) or (label or "this service")
    base = base.strip()
    # Keep the header short and readable, even if label is a full question.
    header = base.upper() if len(base) <= 18 else "this service"
    lines.append(f"Before I look up prices, which exact billed **{header}** do you mean?")
    lines.append("")
    lines.append("Small details can change the code and price (body part, with/without contrast, number of views, screening vs diagnostic, etc.).")
    lines.append("")
    for i, v in enumerate(variants, start=1):
        name = (v.get("variant_name") or "Variant").strip()
        summary = (v.get("patient_summary") or "").strip()
        cpt = (v.get("cpt_code") or "").strip()
        parts = [f"{i}) **{name}**"]
        if cpt:
            parts.append(f"(CPT {cpt})")
        if summary:
            parts.append(f"— {summary}")
        lines.append(" ".join(parts))
    lines.append("")
    lines.append("👉 Reply with the **number** (e.g., `2`) that matches your exam.")
    return "\n".join(lines)

# -------- DB HELPERS --------

async def get_service_variants_for_parent(conn, parent_service: str) -> List[Dict[str, Any]]:
    """Fetch variants from DB. Fallback to empty list on error."""
    try:
        query = text("""
            SELECT id, parent_service, cpt_code, variant_name, patient_summary, is_preventive
            FROM service_variants
            WHERE LOWER(parent_service) = LOWER(:ps)
            ORDER BY variant_name
        """)
        rows = await conn.fetch_all(query, {"ps": parent_service})
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching variants for {parent_service}: {e}")
        return []


async def get_service_variants_by_text(conn, user_text: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Universal variant lookup (NOT limited to imaging).

    Searches service_variants.variant_name and service_variants.cpt_explanation using:
      - ILIKE '%user_text%'
      - punctuation-insensitive match (xray/x-ray/x ray)

    Returns a ranked list with columns needed downstream.
    """
    q = (user_text or "").strip()
    if not q:
        return []

    q_norm = re.sub(r"[^a-z0-9]+", "", q.lower())
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", q.lower()) if len(t) >= 3][:6]

    try:
        query = text("""
            WITH cand AS (
              SELECT
                sv.id,
                sv.parent_service,
                sv.cpt_code,
                sv.variant_name,
                sv.cpt_explanation,
                sv.patient_summary,
                sv.is_preventive,
                (
                  CASE WHEN lower(sv.variant_name) ILIKE '%' || lower(:q) || '%' THEN 8 ELSE 0 END +
                  CASE WHEN lower(coalesce(sv.cpt_explanation,'')) ILIKE '%' || lower(:q) || '%' THEN 5 ELSE 0 END +
                  CASE WHEN regexp_replace(lower(sv.variant_name), '[^a-z0-9]+', '', 'g') LIKE '%' || :qn || '%' THEN 6 ELSE 0 END +
                  CASE WHEN regexp_replace(lower(coalesce(sv.cpt_explanation,'')), '[^a-z0-9]+', '', 'g') LIKE '%' || :qn || '%' THEN 4 ELSE 0 END +
                  (
                    SELECT count(*)
                    FROM unnest(:tokens::text[]) t
                    WHERE lower(sv.variant_name) LIKE '%' || t || '%'
                       OR lower(coalesce(sv.cpt_explanation,'')) LIKE '%' || t || '%'
                  )
                )::int AS score
              FROM service_variants sv
            )
            SELECT *
            FROM cand
            WHERE score > 0
            ORDER BY score DESC, is_preventive DESC NULLS LAST, variant_name ASC, id ASC
            LIMIT :lim
        """)
        rows = await conn.fetch_all(query, {"q": q, "qn": q_norm, "tokens": tokens or [""], "lim": limit})
        # If tokens were empty, unnest([""]) adds noise, strip by re-filtering score>0 already handles most.
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error searching service_variants by text '{q}': {e}")
        return []


def _llm_make_patient_variant_summaries(variants: List[Dict[str, Any]], user_text: str) -> Dict[str, str]:
    """Return {CPT: summary} using the LLM (no web).

    Used only when multiple matches exist and patient_summary is missing.
    """
    system = (
        "You write short, patient-friendly explanations for medical billing service variants. "
        "Avoid medical advice. Focus on purpose, preparation, and key differences. "
        "Return JSON mapping CPT code to a 1-2 sentence summary."
    )
    payload = {
        "user_query": user_text,
        "variants": [
            {
                "cpt_code": v.get("cpt_code"),
                "variant_name": v.get("variant_name"),
                "cpt_explanation": v.get("cpt_explanation"),
            }
            for v in variants
        ],
    }
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
            timeout=12,
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = json.loads(txt) if txt.startswith("{") else {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
    except Exception:
        pass
    return {}

async def price_lookup_staging_by_cpt_with_variants(conn, cpt_code: str, zip_code: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Join stg_hospital_rates + service_variants for richer output."""
    try:
        query = text("""
            SELECT
                h.hospital_name,
                h.city,
                h.state,
                h.zip_code,
                h.distance_miles,
                h.standard_charge,
                h.cash_price,
                h.payer_name,
                h.payer_type,
                v.variant_name,
                v.patient_summary
            FROM stg_hospital_rates h
            LEFT JOIN service_variants v ON h.cpt_code = v.cpt_code
            WHERE h.cpt_code = :cpt
              AND h.zip_code = :zip
            ORDER BY h.distance_miles ASC NULLS LAST, 
                     COALESCE(h.cash_price, h.standard_charge) ASC NULLS LAST
            LIMIT :limit
        """)
        rows = await conn.fetch_all(query, {"cpt": cpt_code, "zip": zip_code, "limit": limit})
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Pricing query failed: {e}")
        return []

# -------- LIFECYCLE --------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- ENDPOINTS --------
@app.post("/chat_stream")
async def chat_stream(req: ChatMessageRequest):
    async def event_gen():
        session_id = req.session_id or str(uuid.uuid4())
        async with database.transaction() as conn:
            # Load or init session state
            query = text("SELECT state FROM sessions WHERE id = :id")
            row = await conn.fetch_one(query, {"id": session_id})
            merged = json.loads(row["state"]) if row else {}

            # Save user message
            await save_message(conn, session_id, "user", req.message, {})

            # High-level intent: pricing?
            intent = "pricing" if re.search(r"(price|cost|how much|charge|pay|afford)", req.message.lower()) else "general"

            # ----------------------------
            # UNIVERSAL SERVICE VARIANT SELECTION (BEFORE ZIP OR PAYMENT)
            # ----------------------------
            # Goal: confirm an exact CPT-backed variant first (for ANY service type),
            # then collect ZIP/payment and run the DB pricing lookup.

            # 1) If we're awaiting a numbered selection, try to apply it.
            if merged.get("_awaiting") == "service_variant" and isinstance(merged.get("_variant_choices"), list):
                merged = apply_service_variant_choice(req.message, merged, merged.get("_variant_choices") or [])
                if merged.get("variant_id") and (merged.get("code_type") and merged.get("code")):
                    merged.pop("_variant_choices", None)

            # 2) If we're awaiting a yes/no confirmation for a single match.
            if merged.get("_awaiting") == "variant_confirm" and isinstance(merged.get("_variant_confirm_choice"), dict):
                ans = (req.message or "").strip().lower()
                if ans in {"yes", "y", "yeah", "yep", "correct", "right", "ok", "okay"}:
                    choice = merged.get("_variant_confirm_choice") or {}
                    merged["code_type"] = "CPT"
                    merged["code"] = str(choice.get("cpt_code") or "").strip()
                    merged["variant_id"] = choice.get("id")
                    merged["variant_name"] = choice.get("variant_name")
                    merged["service_query"] = choice.get("variant_name") or merged.get("service_query")
                    merged.pop("_awaiting", None)
                    merged.pop("_variant_confirm_choice", None)
                elif ans in {"no", "n", "nope", "incorrect", "wrong"}:
                    merged.pop("_variant_confirm_choice", None)
                    merged.pop("_awaiting", None)
                else:
                    msg = "Please reply **Yes** or **No** so I can match the right billed service."
                    yield sse({"type": "delta", "text": msg})
                    await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_variant_confirm"})
                    await update_session_state(conn, session_id, merged)
                    yield sse({"type": "final", "used_web_search": False})
                    return

            # 3) If we still don't have a CPT-backed variant, search service_variants by user text.
            if not (merged.get("code_type") and merged.get("code")):
                query_text = (merged.get("service_query") or "").strip() or (req.message or "").strip()
                candidates = await get_service_variants_by_text(conn, query_text, limit=12)

                # De-dup by (cpt_code, variant_name)
                seen = set()
                uniq: List[Dict[str, Any]] = []
                for c in candidates:
                    key = (str(c.get("cpt_code") or ""), str(c.get("variant_name") or ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(c)

                if len(uniq) == 1:
                    one = uniq[0]
                    merged["_awaiting"] = "variant_confirm"
                    merged["_variant_confirm_choice"] = {
                        "id": one.get("id"),
                        "cpt_code": one.get("cpt_code"),
                        "variant_name": one.get("variant_name"),
                        "patient_summary": one.get("patient_summary"),
                        "cpt_explanation": one.get("cpt_explanation"),
                    }
                    friendly = (one.get("patient_summary") or "").strip()
                    if not friendly:
                        # fall back to a short trimmed explanation
                        friendly = (one.get("cpt_explanation") or "").strip()
                        friendly = (friendly[:160] + "…") if len(friendly) > 160 else friendly
                    label = (one.get("variant_name") or query_text).strip()
                    msg = f"Just to confirm, you mean **{label}**?" + (f"\n- {friendly}" if friendly else "") + "\n\nReply **Yes** or **No**."
                    yield sse({"type": "delta", "text": msg})
                    await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_variant_confirm", "intent": intent})
                    await update_session_state(conn, session_id, merged)
                    yield sse({"type": "final", "used_web_search": False})
                    return

                if len(uniq) >= 2:
                    # Enrich summaries when missing
                    missing = [u for u in uniq if not (u.get("patient_summary") or "").strip()]
                    if missing:
                        llm_summaries = _llm_make_patient_variant_summaries(uniq, query_text)
                        for u in uniq:
                            if not (u.get("patient_summary") or "").strip():
                                u["patient_summary"] = llm_summaries.get(str(u.get("cpt_code") or ""), "") or u.get("patient_summary")

                    lite = [
                        {
                            "id": u.get("id"),
                            "parent_service": u.get("parent_service"),
                            "cpt_code": u.get("cpt_code"),
                            "variant_name": u.get("variant_name"),
                            "patient_summary": u.get("patient_summary"),
                            "is_preventive": u.get("is_preventive"),
                        }
                        for u in uniq
                    ]
                    merged["_variant_choices"] = lite
                    merged["_awaiting"] = "service_variant"
                    msg = build_service_variant_prompt(query_text, lite)
                    yield sse({"type": "delta", "text": msg})
                    await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_service_variant", "intent": intent, "variant_count": len(lite)})
                    await update_session_state(conn, session_id, merged)
                    yield sse({"type": "final", "used_web_search": False})
                    return

                # No DB matches: ask for specificity BEFORE ZIP.
                msg = (
                    "Before I look up prices, what exact service is being ordered? "
                    "Please include any key details like body part, with/without contrast, number of views, "
                    "or screening vs diagnostic if relevant."
                )
                yield sse({"type": "delta", "text": msg})
                await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_service_specifics", "intent": intent})
                await update_session_state(conn, session_id, merged)
                yield sse({"type": "final", "used_web_search": False})
                return

            # ----------------------------
            # ZIP CODE GATE
            # ----------------------------
            if not merged.get("zip_code"):
                if is_zip_code(req.message):
                    merged["zip_code"] = clean_text(req.message)
                else:
                    # Prompt for ZIP
                    msg = "What’s your 5-digit ZIP code? (e.g., 90210)"
                    yield sse({"type": "delta", "text": msg})
                    await save_message(conn, session_id, "assistant", msg, {"mode": "awaiting_zip"})
                    merged["_awaiting"] = "zip_code"
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                    yield sse({"type": "final", "used_web_search": False})
                    return

            # ----------------------------
            # PAYMENT MODE GATE (after ZIP, per instructions)
            # ----------------------------
            if not merged.get("payment_mode"):
                m = (req.message or "").lower()
                if any(k in m for k in ["cash", "self pay", "self-pay", "selfpay"]):
                    merged["payment_mode"] = "cash"
                elif "insurance" in m or "aetna" in m or "uhc" in m or "united" in m or "blue cross" in m or "bcbs" in m:
                    merged["payment_mode"] = "insurance"
                    # Best-effort carrier capture (keep raw text, DB query uses ILIKE patterns)
                    merged["payer_like"] = (req.message or "").strip()
                else:
                    msg = "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"
                    yield sse({"type": "delta", "text": msg})
                    await save_message(conn, session_id, "assistant", msg, {"mode": "awaiting_payment"})
                    merged["_awaiting"] = "payment_mode"
                    await update_session_state(conn, session_id, merged)
                    yield sse({"type": "final", "used_web_search": False})
                    return

            if merged.get("_awaiting") == "payment_mode" and merged.get("payment_mode"):
                merged.pop("_awaiting", None)

            # ----------------------------
            # FINAL PRICING LOOKUP
            # ----------------------------
            code_type = merged.get("code_type")
            code = merged.get("code")
            zip_code = merged.get("zip_code")

            if not (code_type and code and zip_code):
                # Fallback: try to resolve service from query
                ## NOTE: No external resolver available in this deployable build.
                # If we cannot resolve a CPT/code deterministically, we will ask the user for more detail.
                code_type, code = None, None
                if False:
                    pass
                else:
                    msg = "I couldn’t identify the service code. Try being more specific (e.g., 'MRI of the knee without contrast')."
                    yield sse({"type": "delta", "text": msg})
                    await save_message(conn, session_id, "assistant", msg, {"error": "no_code"})
                    await update_session_state(conn, session_id, merged)
                    yield sse({"type": "final", "used_web_search": False})
                    return

            # Do pricing lookup
            results = await price_lookup_staging_by_cpt_with_variants(
                conn,
                zipcode=zip_code,
                cpt_code=code,
                payment_mode=merged.get("payment_mode") or "cash",
                payer_like=merged.get("payer_like"),
                plan_like=merged.get("plan_like"),
                limit=10,
            )

            if not results:
                # Fallback to simpler query
                try:
                    query = text("""
                        SELECT hospital_name, city, state, zip_code, distance_miles,
                               standard_charge, cash_price
                        FROM stg_hospital_rates
                        WHERE cpt_code = :cpt AND zip_code = :zip
                        ORDER BY distance_miles ASC, cash_price ASC
                        LIMIT 10
                    """)
                    rows = await conn.fetch_all(query, {"cpt": code, "zip": zip_code})
                    results = [dict(row) for row in rows]
                except Exception as e:
                    logger.error(f"Fallback pricing failed: {e}")

            if results:
                # Build response
                top = results[:5] if len(results) >= 5 else results
                lines = [f"Here are the closest facilities for **{merged.get('service_query', code)}** near {zip_code}:"]
                lines.append("")
                for i, r in enumerate(top, 1):
                    name = r.get("hospital_name") or "Unnamed Facility"
                    city = r.get("city") or ""
                    state = r.get("state") or ""
                    loc = f"{city}, {state}".strip(", ")
                    if loc == ",": loc = ""

                    cash = r.get("cash_price")
                    std = r.get("standard_charge")
                    price = None
                    if cash is not None and cash > 0:
                        price = f"${cash:,.0f} (cash)"
                    elif std is not None and std > 0:
                        price = f"${std:,.0f} (standard)"
                    else:
                        price = "Price unavailable"

                    dist = r.get("distance_miles")
                    dist_str = f"{dist:.1f} mi away" if dist is not None else "Location unknown"

                    variant = r.get("variant_name") or ""
                    if variant:
                        lines.append(f"{i}) **{name}** — {variant}")
                    else:
                        lines.append(f"{i}) **{name}**")
                    lines.append(f"   • {loc} • {dist_str}")
                    lines.append(f"   • {price}")
                    lines.append("")

                msg = "\n".join(lines)
                yield sse({"type": "delta", "text": msg})
                await save_message(conn, session_id, "assistant", msg, {
                    "mode": "pricing_result",
                    "intent": intent,
                    "code": code,
                    "zip": zip_code,
                    "count": len(results),
                })
            else:
                msg = f"No pricing found for this service (CPT {code}) near {zip_code}. Try a nearby ZIP or check spelling."
                yield sse({"type": "delta", "text": msg})
                await save_message(conn, session_id, "assistant", msg, {"error": "no_results"})

            # Finalize
            await update_session_state(conn, session_id, merged)
            await log_query(conn, session_id, req.message, intent, code, len(results) if results else 0, False, msg)
            yield sse({"type": "final", "used_web_search": False})


    # -------- DB WRAPPERS (stub implementations) --------

    async def save_message(conn, session_id: str, role: str, content: str, meta: Dict):
        ts = datetime.utcnow().isoformat()
        stmt = text("""
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (:sid, :role, :content, :meta, :ts)
            ON CONFLICT DO NOTHING
        """)
        await conn.execute(stmt, {
            "sid": session_id,
            "role": role,
            "content": content,
            "meta": json.dumps(meta),
            "ts": ts,
        })

    async def update_session_state(conn, session_id: str, state: Dict):
        ts = datetime.utcnow().isoformat()
        stmt = text("""
            INSERT INTO sessions (id, state, updated_at)
            VALUES (:id, :state, :ts)
            ON CONFLICT (id) DO UPDATE SET state = :state, updated_at = :ts
        """)
        await conn.execute(stmt, {
            "id": session_id,
            "state": json.dumps(state),
            "ts": ts,
        })

    async def log_query(conn, session_id: str, query: str, intent: str, code: Optional[str], result_count: int, used_web: bool, response: str):
        stmt = text("""
            INSERT INTO query_log (session_id, query, intent, code, result_count, used_web_search, response, created_at)
            VALUES (:sid, :q, :intent, :code, :cnt, :web, :resp, :ts)
        """)
        await conn.execute(stmt, {
            "sid": session_id,
            "q": query,
            "intent": intent,
            "code": code,
            "cnt": result_count,
            "web": used_web,
            "resp": response[:1000],
            "ts": datetime.utcnow().isoformat(),
        })

    return StreamingResponse(event_gen(), media_type="text/event-stream")
