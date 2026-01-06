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

def build_service_variant_prompt(parent_service: str, variants: List[Dict[str, Any]]) -> str:
    """Ask user to choose a variant BEFORE collecting ZIP.
    Emphasize body part, contrast, modality, and clinical purpose."""
    lines: List[str] = []
    lines.append(f"Before I look up prices, which specific type of **{parent_service.upper()}** do you need?")
    lines.append("")
    lines.append("The exact exam matters for accurate pricing (e.g., body part, contrast use, clinical purpose).")
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
            # SERVICE VARIANT SELECTION (BEFORE ZIP OR PAYMENT)
            # ----------------------------
            parent_service = infer_parent_service_key((merged.get("service_query") or "").strip())
            if not parent_service:
                # Try current query if service_query not set
                parent_service = infer_parent_service_key(req.message)

            if parent_service:
                # If already awaiting variant, apply choice
                if merged.get("_awaiting") == "service_variant" and isinstance(merged.get("_variant_choices"), list):
                    merged = apply_service_variant_choice(req.message, merged, merged.get("_variant_choices") or [])
                    # If valid, update service_query
                    if merged.get("variant_name"):
                        merged["service_query"] = f"{parent_service.upper()}: {merged.get('variant_name')}"
                        merged.pop("_awaiting", None)
                        merged.pop("_variant_choices", None)

                # Proceed only if no variant/CPT yet
                if not (merged.get("variant_id") or (merged.get("code_type") and merged.get("code"))):
                    variants = await get_service_variants_for_parent(conn, parent_service)
                    lite = []
                    for v in variants:
                        lite.append({
                            "id": v.get("id"),
                            "parent_service": v.get("parent_service"),
                            "cpt_code": v.get("cpt_code"),
                            "variant_name": v.get("variant_name"),
                            "patient_summary": v.get("patient_summary"),
                            "is_preventive": v.get("is_preventive"),
                        })

                    if not lite:
                        pass  # fall back to generic resolution later
                    elif len(lite) == 1:
                        merged = apply_service_variant_choice("1", merged, lite)
                        if merged.get("variant_name"):
                            merged["service_query"] = f"{parent_service.upper()}: {merged.get('variant_name')}"
                    elif len(lite) >= 2:
                        merged["_variant_choices"] = lite
                        merged["_awaiting"] = "service_variant"
                        msg = build_service_variant_prompt(parent_service, lite)
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {
                            "mode": "clarify_service_variant",
                            "intent": intent,
                            "parent_service": parent_service,
                            "variant_count": len(lite),
                        })
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
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
            results = await price_lookup_staging_by_cpt_with_variants(conn, code, zip_code, limit=10)

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
        }

    return StreamingResponse(event_gen(), media_type="text/event-stream")
)