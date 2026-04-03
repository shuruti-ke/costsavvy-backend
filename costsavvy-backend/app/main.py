# app/main.py
import os
import json
import re
import uuid
import time
import logging
import asyncio
import httpx
import statistics
from typing import Optional, Any, Dict, List, Tuple

from apscheduler.schedulers.asyncio import AsyncIOScheduler

import asyncpg
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, OpenAIError

# Python fallback refiners (file: service_refiners.py)
try:
    from app.service_refiners import refiners_registry
except ImportError:
    from service_refiners import refiners_registry

# ----------------------------
# Config
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("costsavvy")
# Per-request HTTP logs (Brave/OpenAI) are very noisy at INFO
logging.getLogger("httpx").setLevel(logging.WARNING)

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

APP_API_KEY = os.getenv("APP_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MIN_DB_RESULTS_BEFORE_WEB = int(os.getenv("MIN_DB_RESULTS_BEFORE_WEB", "3"))

# Always try to show at least this many facilities in the answer
MIN_FACILITIES_TO_DISPLAY = int(os.getenv("MIN_FACILITIES_TO_DISPLAY", "5"))
# Fetch extra nearby rows so priced results can inherit lat/lon and the map can show 5 pins vs ZIP
NEARBY_COORD_LOOKUP_LIMIT = int(
    os.getenv("NEARBY_COORD_LOOKUP_LIMIT", str(max(30, MIN_FACILITIES_TO_DISPLAY * 6)))
)

# Web search configuration for health education
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes", "y", "on")
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "15"))
BRAVE_API_KEY = (
    os.getenv("BRAVE_API_KEY", "").strip()
    or os.getenv("Brave_API_Key", "").strip()
    or os.getenv("BRAVE_KEY", "").strip()
)
TAVILY_API_KEY = (
    os.getenv("TAVILY_API_KEY", "").strip()
    or os.getenv("Tavily_API_Key", "").strip()
    or os.getenv("TAVILY_KEY", "").strip()
)

# Facility price web search (Brave first, then Tavily + LLM parse); shown before DB when any web price parses
WEB_PRICE_SEARCH_ENABLED = os.getenv("WEB_PRICE_SEARCH_ENABLED", "true").lower() in (
    "1", "true", "yes", "y", "on"
)

# Nearest ZIP hospitals → web price first; DB (negotiated_rates + staging) merged for reference + discrepancy checks
PRICE_WEB_FIRST = os.getenv("PRICE_WEB_FIRST", "true").lower() in ("1", "true", "yes", "y", "on")

# Alert when median web price is far above/below median DB reference (e.g. web ~$1.5k vs DB ~$200)
PRICE_DISCREPANCY_RATIO = float(os.getenv("PRICE_DISCREPANCY_RATIO", "2.5"))
PRICE_DISCREPANCY_MIN_WEB = float(os.getenv("PRICE_DISCREPANCY_MIN_WEB", "500"))
PRICE_DISCREPANCY_MIN_ABS_GAP = float(os.getenv("PRICE_DISCREPANCY_MIN_ABS_GAP", "400"))

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
ADMIN_DISCREPANCY_WEBHOOK_URL = os.getenv("ADMIN_DISCREPANCY_WEBHOOK_URL", "").strip()

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

ENRICHMENT_SCHEDULE_HOUR = int(os.getenv("ENRICHMENT_SCHEDULE_HOUR", "2"))   # 2 AM by default
ENRICHMENT_BATCH_SIZE = int(os.getenv("ENRICHMENT_BATCH_SIZE", "20"))
TRANSPARENCY_FILE_SIZE_LIMIT_MB = int(os.getenv("TRANSPARENCY_FILE_SIZE_LIMIT_MB", "100"))

# ----------------------------
# App
# ----------------------------
app = FastAPI()

# CORS middleware - allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

client = OpenAI(api_key=OPENAI_API_KEY)
pool: asyncpg.Pool | None = None

# Mount static files if directory exists
import pathlib
static_dir = pathlib.Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    logger.warning("Static directory not found - static files will not be served")


# ----------------------------
# Models
# ----------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    # Quick Search: ignore stored session_state so any option change gets a clean pricing context
    fresh_context: bool = False


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
# Nightly hospital enrichment
# ----------------------------

_scheduler: Optional[AsyncIOScheduler] = None


async def _ensure_enrichment_table(conn: asyncpg.Connection) -> None:
    """Create the pending enrichment queue table if it doesn't exist."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS public.pending_hospital_enrichment (
            id              SERIAL PRIMARY KEY,
            hospital_name   TEXT NOT NULL,
            address         TEXT,
            zipcode         TEXT,
            discovered_zip  TEXT,
            source          TEXT DEFAULT 'web_search',
            status          TEXT DEFAULT 'pending',
            hospital_id     INT,
            error_message   TEXT,
            discovered_at   TIMESTAMPTZ DEFAULT NOW(),
            processed_at    TIMESTAMPTZ,
            UNIQUE (hospital_name, COALESCE(zipcode, ''))
        )
    """)


async def queue_hospitals_for_enrichment(
    conn: asyncpg.Connection,
    hospitals: List[Dict[str, Any]],
    discovered_zip: str,
) -> int:
    """
    Queue newly web-discovered hospitals for nightly enrichment.
    Returns count of newly queued rows.
    """
    queued = 0
    for h in hospitals:
        name = (h.get("hospital_name") or "").strip()
        if not name or len(name) < 4:
            continue
        try:
            await conn.execute(
                """
                INSERT INTO public.pending_hospital_enrichment
                    (hospital_name, address, zipcode, discovered_zip, source, status)
                VALUES ($1, $2, $3, $4, 'web_search', 'pending')
                ON CONFLICT (hospital_name, COALESCE(zipcode, '')) DO NOTHING
                """,
                name,
                (h.get("address") or "").strip() or None,
                (h.get("zipcode") or "").strip() or None,
                discovered_zip or None,
            )
            queued += 1
        except Exception as e:
            logger.warning("Failed to queue hospital %s for enrichment: %s", name, e)
    return queued


async def _geocode_nominatim(address: str) -> Optional[tuple]:
    """
    Free geocoding via OpenStreetMap Nominatim. Returns (lat, lon) or None.
    Rate-limited to 1 req/s by Nominatim policy — caller must throttle.
    """
    if not address:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            resp = await hc.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": address, "format": "json", "limit": 1},
                headers={"User-Agent": "CostSavvy-HealthPriceApp/1.0 (health price transparency)"},
            )
            if resp.status_code == 200:
                results = resp.json()
                if results:
                    return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as e:
        logger.warning("Nominatim geocode failed for '%s': %s", address, e)
    return None


async def _enrich_hospital_details_from_web(
    hospital_name: str, address: str, zipcode: str
) -> Dict[str, Any]:
    """
    Use web search to fill in missing hospital details (phone, full address, state).
    Returns a dict with whatever fields could be extracted.
    """
    q = f'"{hospital_name}" hospital address phone number {zipcode or address or ""}'
    snippets: List[str] = []

    if BRAVE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as hc:
                resp = await hc.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": BRAVE_API_KEY},
                    params={"q": q, "count": 5},
                )
                if resp.status_code == 200:
                    for item in resp.json().get("web", {}).get("results", [])[:5]:
                        t = item.get("title", "")
                        s = item.get("description", "")
                        if t or s:
                            snippets.append(f"{t}: {s}")
        except Exception as e:
            logger.warning("Brave hospital detail search failed: %s", e)

    if not snippets and TAVILY_API_KEY:
        raw = await _tavily_search(q, num_results=5)
        snippets = [f"{r.get('title','')}: {r.get('snippet','')}" for r in raw if r.get('title') or r.get('snippet')]

    if not snippets:
        return {}

    system = (
        "Extract hospital contact details from web snippets. "
        'Return JSON only: {"address": "full street address", "city": "city", "state": "2-letter state", '
        '"zipcode": "5-digit zip", "phone": "phone number"}. '
        "Only include fields you can confirm from the snippets. Return {} if nothing found."
    )
    user = f"Hospital: {hospital_name}\nSnippets:\n" + "\n".join(snippets[:5])
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            timeout=15,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.warning("LLM hospital detail extraction failed: %s", e)
        return {}


async def _find_price_transparency_url(hospital_name: str, state: str) -> Optional[str]:
    """
    Search for the hospital's CMS-mandated machine-readable price transparency file URL.
    """
    q = f'"{hospital_name}" {state or ""} hospital price transparency machine readable file JSON CSV 2024 2025'
    snippets: List[Dict[str, Any]] = []

    if BRAVE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as hc:
                resp = await hc.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": BRAVE_API_KEY},
                    params={"q": q, "count": 8},
                )
                if resp.status_code == 200:
                    snippets = resp.json().get("web", {}).get("results", [])[:8]
        except Exception as e:
            logger.warning("Brave transparency URL search failed: %s", e)

    if not snippets and TAVILY_API_KEY:
        raw = await _tavily_search(q, num_results=8)
        snippets = [{"url": r.get("url",""), "title": r.get("title",""), "description": r.get("snippet","")} for r in raw]

    if not snippets:
        return None

    # Look for direct file URLs in results
    file_extensions = (".json", ".csv", ".xlsx", ".xls", ".zip")
    for item in snippets:
        url = (item.get("url") or "").lower()
        if any(url.endswith(ext) for ext in file_extensions):
            return item["url"]

    # Ask LLM to extract a file URL from snippets
    body = "\n".join(
        f"- URL: {s.get('url','')} | {s.get('title','')}: {(s.get('description') or s.get('snippet',''))[:300]}"
        for s in snippets
    )
    system = (
        "You extract the direct URL of a hospital's CMS price transparency machine-readable file. "
        "Look for URLs ending in .json, .csv, .xlsx, or .zip that are hospital chargemaster/price files. "
        'Return JSON: {"url": "direct file URL or null"}. '
        "Return null if no direct file URL is found."
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Hospital: {hospital_name}\nSearch results:\n{body}"},
            ],
            response_format={"type": "json_object"},
            timeout=15,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        url = data.get("url")
        return url if url and url != "null" and url.startswith("http") else None
    except Exception as e:
        logger.warning("LLM transparency URL extraction failed: %s", e)
        return None


def _parse_transparency_csv_rows(
    content: bytes, hospital_id: int, hospital_name: str
) -> List[Dict[str, Any]]:
    """
    Parse CMS-standard hospital price transparency CSV.
    Extracts rows with CPT codes and prices.
    Returns list of dicts ready for stg_hospital_rates insert.
    """
    import io
    import csv

    rows: List[Dict[str, Any]] = []
    try:
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]

        # Map common column name variants
        def find_col(candidates):
            for c in candidates:
                for h in headers:
                    if c in h:
                        return h
            return None

        col_code = find_col(["cpt", "hcpcs", "code", "procedure_code", "billing_code"])
        col_desc = find_col(["description", "service", "item", "procedure"])
        col_cash = find_col(["cash", "discounted_cash", "self_pay", "selfpay"])
        col_gross = find_col(["gross", "chargemaster", "list_price", "standard_charge"])
        col_payer = find_col(["payer", "insurer", "insurance"])
        col_neg = find_col(["negotiated", "contracted", "allowed"])

        if not col_code:
            return rows

        for row in reader:
            code = (row.get(col_code) or "").strip()
            if not code or not any(c.isdigit() for c in code):
                continue
            # Only CPT codes (5-digit numeric or alphanumeric starting with digit)
            if len(code) > 10:
                continue

            def safe_float(val):
                if not val:
                    return None
                try:
                    return float(str(val).replace("$", "").replace(",", "").strip())
                except Exception:
                    return None

            cash = safe_float(row.get(col_cash)) if col_cash else None
            gross = safe_float(row.get(col_gross)) if col_gross else None
            neg = safe_float(row.get(col_neg)) if col_neg else None
            desc = (row.get(col_desc) or "")[:500] if col_desc else ""
            payer = (row.get(col_payer) or "")[:200] if col_payer else None

            if cash is None and gross is None and neg is None:
                continue

            rows.append({
                "hospital_id": hospital_id,
                "hospital_name": hospital_name,
                "code": code,
                "code_type": "CPT",
                "service_description": desc,
                "standard_charge_discounted_cash": cash,
                "standard_charge_gross": gross,
                "standard_charge_negotiated_dollar": neg,
                "payer_name": payer,
            })

            if len(rows) >= 50000:  # cap per file
                break
    except Exception as e:
        logger.warning("CSV transparency parse error: %s", e)
    return rows


async def _download_and_parse_transparency_file(
    url: str, hospital_id: int, hospital_name: str
) -> List[Dict[str, Any]]:
    """
    Download a price transparency file (CSV or JSON) with size and timeout limits.
    Returns parsed rows for stg_hospital_rates.
    """
    size_limit = TRANSPARENCY_FILE_SIZE_LIMIT_MB * 1024 * 1024
    rows: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as hc:
            async with hc.stream("GET", url) as resp:
                if resp.status_code != 200:
                    logger.warning("Transparency file download failed: HTTP %s for %s", resp.status_code, url)
                    return []
                content_type = resp.headers.get("content-type", "").lower()
                chunks: List[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > size_limit:
                        logger.info("Transparency file for %s exceeds %dMB limit, truncating", hospital_name, TRANSPARENCY_FILE_SIZE_LIMIT_MB)
                        chunks.append(chunk)
                        break
                    chunks.append(chunk)
                content = b"".join(chunks)

        url_lower = url.lower()
        if url_lower.endswith(".csv") or "csv" in content_type:
            rows = _parse_transparency_csv_rows(content, hospital_id, hospital_name)
        elif url_lower.endswith(".json") or "json" in content_type:
            try:
                data = json.loads(content.decode("utf-8", errors="replace"))
                # CMS standard JSON schema has a "standard_charge_information" array
                charges = []
                if isinstance(data, dict):
                    charges = data.get("standard_charge_information", []) or data.get("items", []) or []
                elif isinstance(data, list):
                    charges = data

                for item in charges[:50000]:
                    if not isinstance(item, dict):
                        continue
                    code = str(item.get("code") or item.get("cpt_code") or item.get("billing_code") or "").strip()
                    if not code or len(code) > 10:
                        continue
                    desc = str(item.get("description") or item.get("service_description") or "")[:500]

                    def _sf(k):
                        v = item.get(k)
                        try:
                            return float(v) if v is not None else None
                        except Exception:
                            return None

                    cash = _sf("discounted_cash") or _sf("cash_price") or _sf("self_pay_price")
                    gross = _sf("gross_charge") or _sf("standard_charge_gross")
                    neg = _sf("negotiated_dollar") or _sf("negotiated_rate")
                    if cash is None and gross is None and neg is None:
                        continue
                    rows.append({
                        "hospital_id": hospital_id,
                        "hospital_name": hospital_name,
                        "code": code,
                        "code_type": "CPT",
                        "service_description": desc,
                        "standard_charge_discounted_cash": cash,
                        "standard_charge_gross": gross,
                        "standard_charge_negotiated_dollar": neg,
                        "payer_name": None,
                    })
            except Exception as e:
                logger.warning("JSON transparency parse error for %s: %s", hospital_name, e)

        logger.info("Parsed %d price rows from transparency file for %s", len(rows), hospital_name)
    except Exception as e:
        logger.warning("Transparency file download/parse failed for %s (%s): %s", hospital_name, url, e)
    return rows


async def _bulk_insert_stg_rates(conn: asyncpg.Connection, rows: List[Dict[str, Any]]) -> int:
    """Bulk upsert parsed transparency prices into stg_hospital_rates."""
    if not rows:
        return 0
    inserted = 0
    # Insert in batches of 500
    for i in range(0, len(rows), 500):
        batch = rows[i:i+500]
        try:
            await conn.executemany(
                """
                INSERT INTO public.stg_hospital_rates
                    (hospital_id, hospital_name, code, code_type, service_description,
                     standard_charge_discounted_cash, standard_charge_gross,
                     standard_charge_negotiated_dollar, payer_name)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT DO NOTHING
                """,
                [
                    (
                        r["hospital_id"], r["hospital_name"], r["code"], r["code_type"],
                        r.get("service_description"), r.get("standard_charge_discounted_cash"),
                        r.get("standard_charge_gross"), r.get("standard_charge_negotiated_dollar"),
                        r.get("payer_name"),
                    )
                    for r in batch
                ],
            )
            inserted += len(batch)
        except Exception as e:
            logger.warning("Batch stg_rates insert failed: %s", e)
    return inserted


async def run_nightly_hospital_enrichment() -> Dict[str, Any]:
    """
    Nightly job: process pending_hospital_enrichment queue.
    For each pending hospital:
      1. Enrich address/phone via web search
      2. Geocode via Nominatim
      3. Upsert into hospital_details
      4. Find + parse CMS price transparency file
      5. Bulk insert prices into stg_hospital_rates
    """
    if not pool:
        logger.warning("Enrichment job: DB pool not ready, skipping")
        return {"skipped": True}

    summary = {"processed": 0, "enriched": 0, "geocoded": 0, "prices_loaded": 0, "errors": 0}

    async with pool.acquire() as conn:
        await _ensure_enrichment_table(conn)

        pending = await conn.fetch(
            """
            SELECT id, hospital_name, address, zipcode, discovered_zip
            FROM public.pending_hospital_enrichment
            WHERE status = 'pending'
            ORDER BY discovered_at ASC
            LIMIT $1
            """,
            ENRICHMENT_BATCH_SIZE,
        )

        if not pending:
            logger.info("Nightly enrichment: no pending hospitals")
            return summary

        logger.info("Nightly enrichment: processing %d hospitals", len(pending))

        for row in pending:
            row_id = row["id"]
            name = row["hospital_name"]
            address = row["address"] or ""
            zipcode = row["zipcode"] or row["discovered_zip"] or ""

            try:
                await conn.execute(
                    "UPDATE public.pending_hospital_enrichment SET status='processing' WHERE id=$1", row_id
                )

                # Step 1: enrich details from web
                details = await _enrich_hospital_details_from_web(name, address, zipcode)
                full_address = details.get("address") or address
                city = details.get("city") or ""
                state = details.get("state") or ""
                zip_enriched = details.get("zipcode") or zipcode
                phone = details.get("phone")

                # Build geocodable address string
                geo_str = ", ".join(filter(None, [full_address, city, state, zip_enriched]))
                if not geo_str:
                    geo_str = f"{name} hospital"

                # Step 2: geocode
                import asyncio as _asyncio
                await _asyncio.sleep(1.1)  # Nominatim 1 req/s limit
                coords = await _geocode_nominatim(geo_str)
                lat = coords[0] if coords else None
                lon = coords[1] if coords else None
                if lat:
                    summary["geocoded"] += 1

                # Step 3: upsert into hospital_details
                hospital_id = await conn.fetchval(
                    """
                    INSERT INTO public.hospital_details
                        (hospital_name, address, state, zipcode, phone, latitude, longitude)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (hospital_name, COALESCE(zipcode, ''))
                    DO UPDATE SET
                        address    = COALESCE(EXCLUDED.address, hospital_details.address),
                        state      = COALESCE(EXCLUDED.state, hospital_details.state),
                        zipcode    = COALESCE(EXCLUDED.zipcode, hospital_details.zipcode),
                        phone      = COALESCE(EXCLUDED.phone, hospital_details.phone),
                        latitude   = COALESCE(EXCLUDED.latitude, hospital_details.latitude),
                        longitude  = COALESCE(EXCLUDED.longitude, hospital_details.longitude)
                    RETURNING hospital_id
                    """,
                    name,
                    full_address or None,
                    state or None,
                    zip_enriched or None,
                    phone or None,
                    lat,
                    lon,
                )
                summary["enriched"] += 1

                # Step 4: find price transparency file
                prices_loaded = 0
                if hospital_id and WEB_SEARCH_ENABLED and (BRAVE_API_KEY or TAVILY_API_KEY):
                    transparency_url = await _find_price_transparency_url(name, state)
                    if transparency_url:
                        price_rows = await _download_and_parse_transparency_file(
                            transparency_url, hospital_id, name
                        )
                        prices_loaded = await _bulk_insert_stg_rates(conn, price_rows)
                        summary["prices_loaded"] += prices_loaded

                await conn.execute(
                    """
                    UPDATE public.pending_hospital_enrichment
                    SET status='done', hospital_id=$2, processed_at=NOW(), error_message=NULL
                    WHERE id=$1
                    """,
                    row_id, hospital_id,
                )
                summary["processed"] += 1
                logger.info(
                    "Enriched hospital '%s': geocoded=%s, prices=%d",
                    name, coords is not None, prices_loaded,
                )

            except Exception as e:
                summary["errors"] += 1
                logger.error("Enrichment failed for '%s': %s", name, e)
                try:
                    await conn.execute(
                        "UPDATE public.pending_hospital_enrichment SET status='failed', error_message=$2 WHERE id=$1",
                        row_id, str(e)[:500],
                    )
                except Exception:
                    pass

    logger.info("Nightly enrichment complete: %s", summary)
    return summary

# ----------------------------
# Startup / Shutdown
# ----------------------------
@app.on_event("startup")
async def startup():
    global pool, _scheduler
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        await _ensure_enrichment_table(conn)
    # Schedule nightly hospital enrichment
    _scheduler = AsyncIOScheduler(timezone="UTC")
    _scheduler.add_job(
        run_nightly_hospital_enrichment,
        "cron",
        hour=ENRICHMENT_SCHEDULE_HOUR,
        minute=0,
        id="nightly_hospital_enrichment",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Nightly hospital enrichment scheduler started (runs at %02d:00 UTC)", ENRICHMENT_SCHEDULE_HOUR)


@app.on_event("shutdown")
async def shutdown():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    if pool:
        await pool.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/admin/run-enrichment")
async def admin_run_enrichment(x_admin_key: str = Header(default="")):
    """Manually trigger the nightly hospital enrichment job."""
    api_key = os.getenv("APP_API_KEY", "")
    if api_key and x_admin_key != api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    result = await run_nightly_hospital_enrichment()
    return {"status": "ok", "result": result}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    """Browsers request /favicon.ico by default; serve branded SVG (widely supported)."""
    p = pathlib.Path("static/favicon.svg")
    if p.is_file():
        return FileResponse(p, media_type="image/svg+xml")
    return Response(status_code=204)


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
async def _tavily_search(query: str, num_results: int = 8) -> List[Dict[str, Any]]:
    """Tavily search API; returns same shape as Brave snippets for downstream LLM use."""
    if not TAVILY_API_KEY:
        return []
    n = max(1, min(num_results, 20))
    try:
        async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as http:
            resp = await http.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": n,
                    "include_answer": False,
                },
            )
            if resp.status_code != 200:
                logger.warning(f"Tavily search HTTP {resp.status_code}")
                return []
            data = resp.json()
            out: List[Dict[str, Any]] = []
            for item in (data.get("results") or [])[:n]:
                out.append(
                    {
                        "title": (item.get("title") or "")[:500],
                        "snippet": (item.get("content") or item.get("raw_content") or "")[:2000],
                        "url": item.get("url") or "",
                        "source": "tavily",
                    }
                )
            return out
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


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
    
    if not results and TAVILY_API_KEY:
        tq = f"{query} health medical"
        results.extend(await _tavily_search(tq, num_results=num_results))

    return results


def _normalize_hospital_key(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


async def _web_search_hospital_pricing_snippets(
    hospital_name: str,
    location_hint: str,
    service_query: str,
    code: str,
    payment_mode: str,
    num_results: int = 8,
) -> List[Dict[str, Any]]:
    """General web search tuned for facility-level price signals (not health-education sites)."""
    if not WEB_SEARCH_ENABLED or not WEB_PRICE_SEARCH_ENABLED:
        return []
    parts = [f'"{hospital_name}"', (service_query or "").strip() or "medical procedure"]
    if code:
        parts.append(str(code))
    if location_hint:
        parts.append(location_hint)
    pm = (payment_mode or "cash").lower()
    if pm.startswith("insur"):
        parts.append("total hospital charges facility fee price estimate insurance")
    else:
        parts.append("cash price self-pay hospital chargemaster")
    query = " ".join(p for p in parts if p)

    out: List[Dict[str, Any]] = []

    if BRAVE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as http:
                resp = await http.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": BRAVE_API_KEY},
                    params={"q": query, "count": num_results},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("web", {}).get("results", [])[:num_results]:
                        out.append(
                            {
                                "title": item.get("title", ""),
                                "snippet": item.get("description", ""),
                                "url": item.get("url", ""),
                                "source": "brave",
                            }
                        )
        except Exception as e:
            logger.warning(f"Brave facility price search failed: {e}")

    if not out and TAVILY_API_KEY:
        out.extend(await _tavily_search(query, num_results=num_results))

    return out


def _parse_facility_web_prices_batch_llm(
    hospital_snippets: List[Tuple[str, List[Dict[str, Any]]]],
    service_query: str,
    payment_mode: str,
) -> Dict[str, float]:
    """One structured pass over all snippets; returns normalized-hospital-key -> USD price."""
    if not hospital_snippets:
        return {}
    blocks = []
    for hosp, results in hospital_snippets:
        body = "\n".join(
            f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in (results or [])[:10]
        )
        blocks.append(f'### "{hosp}"\n{body}')
    user = (
        f"Service: {service_query}\nPayment context: {payment_mode}\n\n" + "\n\n".join(blocks)
    )
    n = len(hospital_snippets)
    system = (
        "You read web search snippets about U.S. hospital pricing. There is one ### hospital section in order; "
        f"the user message has exactly {n} sections. "
        "For each section, extract ONE numeric USD price only if snippets clearly cite an amount for that specific hospital "
        "(or its published price list / chargemaster / CDM for this service). "
        "Do not guess from unrelated facilities or generic national averages unless clearly tied to that hospital. "
        f'Return JSON only: an array of exactly {n} objects '
        '[{"hospital":"<exact heading name>","price_usd":number|null}, ...] '
        "in the SAME order as the ### sections (first object = first hospital). Use null when unclear."
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=45,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw)
        arr = json.loads(raw)
    except Exception as e:
        logger.warning(f"Batch web price parse failed: {e}")
        return {}

    by_key: Dict[str, float] = {}
    if not isinstance(arr, list):
        return {}
    for i, entry in enumerate(arr):
        if not isinstance(entry, dict):
            continue
        p = entry.get("price_usd")
        if not isinstance(p, (int, float)) or p <= 0:
            continue
        if i < len(hospital_snippets):
            anchor = (hospital_snippets[i][0] or "").strip()
            if anchor:
                by_key[_normalize_hospital_key(anchor)] = float(p)
                continue
        name = (entry.get("hospital") or "").strip()
        if name:
            by_key[_normalize_hospital_key(name)] = float(p)
    return by_key


def _web_facility_catalog_from_targets(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One entry per probed hospital that got a web-extracted price (for UI + API)."""
    out: List[Dict[str, Any]] = []
    for t in targets:
        wp = t.get("web_candidate_price")
        if wp is None:
            continue
        try:
            wf = float(wp)
        except (TypeError, ValueError):
            continue
        if wf <= 0:
            continue
        lat, lon = t.get("latitude"), t.get("longitude")
        try:
            lat_f = float(lat) if lat is not None else None
        except (TypeError, ValueError):
            lat_f = None
        try:
            lon_f = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            lon_f = None
        out.append(
            {
                "hospital_name": _pick_hospital_name(t),
                "address": _pick_address(t),
                "distance_miles": t.get("distance_miles") or t.get("distance"),
                "web_price": wf,
                "latitude": lat_f,
                "longitude": lon_f,
            }
        )
    return out


async def _web_search_facilities_near_zip(zipcode: str, num_results: int = 14) -> List[Dict[str, Any]]:
    """Broad web search for hospitals serving a ZIP (when local DB directory has no rows)."""
    z = (zipcode or "").strip()
    if not z:
        return []
    # Use multiple targeted queries for better regional coverage
    q = f'hospitals medical centers near ZIP {z} acute care emergency health system surgery'
    out: List[Dict[str, Any]] = []
    if BRAVE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as http:
                resp = await http.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": BRAVE_API_KEY},
                    params={"q": q, "count": max(1, min(num_results, 20))},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("web", {}).get("results", [])[:num_results]:
                        out.append(
                            {
                                "title": item.get("title", "")[:500],
                                "snippet": item.get("description", "")[:2000],
                                "url": item.get("url", ""),
                                "source": "brave",
                            }
                        )
        except Exception as e:
            logger.warning("Brave ZIP facility directory search failed: %s", e)
    if not out and TAVILY_API_KEY:
        out.extend(await _tavily_search(q, num_results=num_results))
    # If still sparse, try a second broader query (catches regional health systems)
    if len(out) < 5:
        q2 = f'major hospital health system serving {z} ZIP code United States'
        if BRAVE_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as http:
                    resp = await http.get(
                        'https://api.search.brave.com/res/v1/web/search',
                        headers={'X-Subscription-Token': BRAVE_API_KEY},
                        params={'q': q2, 'count': max(1, min(num_results, 20))},
                    )
                    if resp.status_code == 200:
                        for item in resp.json().get('web', {}).get('results', [])[:num_results]:
                            out.append({
                                'title': item.get('title', '')[:500],
                                'snippet': item.get('description', '')[:2000],
                                'url': item.get('url', ''),
                                'source': 'brave',
                            })
            except Exception as e:
                logger.warning('Brave broad facility search failed: %s', e)
        elif TAVILY_API_KEY:
            out.extend(await _tavily_search(q2, num_results=num_results))
    return out


def _fallback_hospital_names_from_zip_snippets(
    snippets: List[Dict[str, Any]], max_h: int
) -> List[Dict[str, Any]]:
    """Heuristic names from result titles when LLM parse is unavailable."""
    out: List[Dict[str, Any]] = []
    for s in snippets or []:
        t = (s.get("title") or "").strip()
        if not t or len(t) < 8:
            continue
        tl = t.lower()
        if not any(
            k in tl
            for k in (
                "hospital",
                "medical center",
                "health system",
                "clinic",
                "upmc",
                "mayo",
                "cleveland clinic",
            )
        ):
            continue
        name = t.split(" - ")[0].split(" | ")[0].split(" – ")[0].strip()[:200]
        if len(name) < 5:
            continue
        out.append({"hospital_name": name, "address": ""})
        if len(out) >= max_h:
            break
    return out


def _parse_hospital_list_near_zip_llm(
    zipcode: str, snippets: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    if not snippets:
        return []
    body = "\n".join(
        f"- {s.get('title', '')}: {(s.get('snippet') or '')[:450]}" for s in snippets[:20]
    )
    system = (
        "You extract real U.S. hospital or major medical center names that plausibly serve patients near the ZIP, "
        "using only the web snippets (do not invent). "
        f'Return JSON only: {{"hospitals":[{{"hospital_name":"string","address":"string"}}]}} '
        f"with at most {limit} distinct facilities; omit chains of only primary-care clinics if acute hospitals exist."
    )
    user = f"ZIP: {zipcode}\n\nSnippets:\n{body}"
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            timeout=30,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw)
        data = json.loads(raw)
        arr = data.get("hospitals") if isinstance(data, dict) else None
        if not isinstance(arr, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            name = (item.get("hospital_name") or item.get("name") or "").strip()
            addr = (item.get("address") or "").strip()
            if len(name) < 4:
                continue
            out.append({"hospital_name": name, "address": addr})
            if len(out) >= limit:
                break
        return out
    except Exception as e:
        logger.warning("LLM hospital list parse near ZIP failed: %s", e)
        return []


async def synthetic_nearby_hospitals_from_web(zipcode: str) -> List[Dict[str, Any]]:
    """
    Build a facility list from web search when public.hospital_details has nothing near this ZIP.
    Rows match the shape expected by attach_web_prices_to_facility_results / build_facility_block.
    """
    z = (zipcode or "").strip()
    if len(z) != 5 or not z.isdigit():
        return []
    if not WEB_SEARCH_ENABLED or not (BRAVE_API_KEY or TAVILY_API_KEY):
        return []

    snips = await _web_search_facilities_near_zip(z)
    lim = max(MIN_FACILITIES_TO_DISPLAY, 5)
    rows = _parse_hospital_list_near_zip_llm(z, snips, limit=lim)
    if not rows:
        rows = _fallback_hospital_names_from_zip_snippets(snips, lim)
    out: List[Dict[str, Any]] = []
    for r in rows:
        name = (r.get("hospital_name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "hospital_id": None,
                "hospital_name": name,
                "address": (r.get("address") or "").strip(),
                "state": None,
                "zipcode": None,
                "phone": None,
                "latitude": None,
                "longitude": None,
                "distance_miles": None,
                "_web_discovered": True,
            }
        )
        if len(out) >= NEARBY_COORD_LOOKUP_LIMIT:
            break
    if out:
        logger.info("Web-synthesized %d hospitals near ZIP %s (no DB directory match)", len(out), z)
    return out


async def attach_web_prices_to_facility_results(
    results: List[Dict[str, Any]],
    service_query: str,
    code: str,
    payment_mode: str,
    fill_hospitals: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, List[float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Always (when enabled + API keys) runs web searches for up to MIN_FACILITIES_TO_DISPLAY
    hospitals to confirm DB pricing — priced rows first, then nearby facilities to reach 5.
    Sets web_candidate_price on rows when extraction succeeds.
    Returns (attempted, all_web_prices_from_targets, web_facility_catalog, probed_row_refs).
    """
    if not WEB_PRICE_SEARCH_ENABLED or not WEB_SEARCH_ENABLED:
        return False, [], [], []
    if not (BRAVE_API_KEY or TAVILY_API_KEY):
        return False, [], [], []

    targets: List[Dict[str, Any]] = []
    seen = set()
    limit = MIN_FACILITIES_TO_DISPLAY

    for r in results:
        name = _pick_hospital_name(r)
        key = name.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        targets.append(r)
        if len(targets) >= limit:
            break

    if fill_hospitals and len(targets) < limit:
        for h in fill_hospitals:
            if len(targets) >= limit:
                break
            name = (h.get("hospital_name") or "").strip()
            key = name.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            targets.append(h)

    if not targets:
        return False, [], [], []

    sem = asyncio.Semaphore(4)

    async def fetch_one(row: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        name = _pick_hospital_name(row)
        loc = _pick_address(row) or ""
        async with sem:
            snips = await _web_search_hospital_pricing_snippets(
                name, loc, service_query, code, payment_mode
            )
        return name, snips

    pairs = await asyncio.gather(*[fetch_one(r) for r in targets])
    key_to_price = _parse_facility_web_prices_batch_llm(pairs, service_query, payment_mode)
    missing_pairs = [
        (hn, sn)
        for hn, sn in pairs
        if _normalize_hospital_key(hn) not in key_to_price
    ]
    if missing_pairs:
        key_to_price.update(
            _parse_facility_web_prices_batch_llm(missing_pairs, service_query, payment_mode)
        )

    for r in targets:
        k = _normalize_hospital_key(_pick_hospital_name(r))
        p = key_to_price.get(k)
        if p is None and key_to_price:
            for kk, pv in key_to_price.items():
                if len(k) >= 6 and (k in kk or kk in k):
                    p = pv
                    break
        if p is not None:
            r["web_candidate_price"] = p

    web_vals_all = [
        float(t["web_candidate_price"])
        for t in targets
        if t.get("web_candidate_price") is not None
    ]
    catalog = _web_facility_catalog_from_targets(targets)
    return True, web_vals_all, catalog, targets


_DB_MERGE_KEYS = (
    "best_price",
    "standard_charge_cash",
    "negotiated_dollar",
    "estimated_amount",
    "standard_charge_gross",
    "payer_name",
    "plan_name",
    "code",
    "code_type",
)


def merge_db_price_fields_into_rows(
    target_rows: List[Dict[str, Any]], db_rows: List[Dict[str, Any]]
) -> None:
    """Copy negotiated_rates-shaped fields from progressive DB lookup onto probed rows (reference only)."""
    if not target_rows or not db_rows:
        return
    by_id: Dict[int, Dict[str, Any]] = {}
    by_key: Dict[str, Dict[str, Any]] = {}
    for r in db_rows:
        hid = r.get("hospital_id")
        try:
            if hid is not None:
                by_id[int(hid)] = r
        except (TypeError, ValueError):
            pass
        k = _normalize_hospital_key(_pick_hospital_name(r))
        if k and k not in by_key:
            by_key[k] = r
    for row in target_rows:
        src = None
        hid = row.get("hospital_id")
        try:
            if hid is not None:
                src = by_id.get(int(hid))
        except (TypeError, ValueError):
            pass
        if src is None:
            src = by_key.get(_normalize_hospital_key(_pick_hospital_name(row)))
        if not src:
            continue
        for k in _DB_MERGE_KEYS:
            if row.get(k) is None and src.get(k) is not None:
                row[k] = src[k]


def analyze_web_vs_db_discrepancy(
    rows: List[Dict[str, Any]], payment_mode: str
) -> Optional[Dict[str, Any]]:
    """
    Flag when web-extracted prices cluster high but DB reference is much lower (or the reverse).
    """
    web_vals: List[float] = []
    db_vals: List[float] = []
    per_hospital: List[Dict[str, Any]] = []
    for r in rows:
        w = r.get("web_candidate_price")
        d = r.get("db_reference_price")
        wf = float(w) if isinstance(w, (int, float)) and float(w) > 0 else None
        df = float(d) if isinstance(d, (int, float)) and float(d) > 0 else None
        if wf is not None:
            web_vals.append(wf)
        if df is not None:
            db_vals.append(df)
        if wf is not None or df is not None:
            per_hospital.append(
                {
                    "hospital_name": _pick_hospital_name(r),
                    "web_price": wf,
                    "db_reference_price": df,
                }
            )
    if len(web_vals) < 2 or not db_vals:
        return None
    mw = float(statistics.median(web_vals))
    md = float(statistics.median(db_vals))
    if mw < PRICE_DISCREPANCY_MIN_WEB:
        return None
    ratio_high = mw / md if md > 0 else None
    ratio_low = md / mw if mw > 0 else None
    triggered = False
    reason = ""
    if ratio_high is not None and ratio_high >= PRICE_DISCREPANCY_RATIO and (mw - md) >= PRICE_DISCREPANCY_MIN_ABS_GAP:
        triggered = True
        reason = f"median_web {mw:.0f} is ~{ratio_high:.1f}x median_db {md:.0f}"
    elif ratio_low is not None and ratio_low >= PRICE_DISCREPANCY_RATIO and (md - mw) >= PRICE_DISCREPANCY_MIN_ABS_GAP:
        triggered = True
        reason = f"median_db {md:.0f} is ~{ratio_low:.1f}x median_web {mw:.0f}"
    if not triggered:
        wspread = max(web_vals) - min(web_vals)
        if (
            len(web_vals) >= 3
            and mw > 0
            and wspread <= 0.4 * mw
            and ratio_high is not None
            and ratio_high >= 2.0
            and (mw - md) >= PRICE_DISCREPANCY_MIN_ABS_GAP
        ):
            triggered = True
            reason = f"tight_web_cluster (spread {wspread:.0f}) vs median_db {md:.0f} (~{ratio_high:.1f}x)"
    if not triggered:
        return None
    return {
        "reason": reason,
        "median_web": mw,
        "median_db": md,
        "web_prices": web_vals,
        "db_prices": db_vals,
        "hospitals": per_hospital,
        "payment_mode": payment_mode,
    }


async def record_price_discrepancy_alert(
    conn: Optional[asyncpg.Connection],
    session_id: Optional[str],
    zipcode: str,
    code: Optional[str],
    payment_mode: str,
    detail: Dict[str, Any],
) -> None:
    """
    Super-admin signals: structured log, optional webhook POST, optional DB row.
    Create table (once): see POST /admin/price_discrepancies docs or run:
      CREATE TABLE IF NOT EXISTS public.admin_price_discrepancy_alerts (
        id bigserial PRIMARY KEY,
        created_at timestamptz DEFAULT now(),
        session_id text,
        zipcode text,
        code text,
        payment_mode text,
        detail_json jsonb
      );
    """
    payload = {
        "type": "price_discrepancy",
        "zipcode": zipcode,
        "code": code,
        "payment_mode": payment_mode,
        "session_id": session_id,
        "detail": detail,
    }
    logger.warning("PRICE_DISCREPANCY_ALERT %s", json.dumps(payload, default=str)[:4000])
    if ADMIN_DISCREPANCY_WEBHOOK_URL:
        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                await http.post(ADMIN_DISCREPANCY_WEBHOOK_URL, json=payload)
        except Exception as e:
            logger.warning(f"Discrepancy webhook post failed: {e}")
    if conn is not None:
        try:
            await conn.execute(
                """
                INSERT INTO public.admin_price_discrepancy_alerts
                  (session_id, zipcode, code, payment_mode, detail_json)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                session_id,
                zipcode,
                code or "",
                payment_mode,
                json.dumps(detail, default=str),
            )
        except Exception as e:
            logger.info("admin_price_discrepancy_alerts insert skipped: %s", e)


def _row_effective_price(
    row: Dict[str, Any],
    payment_mode: str = "cash",
    prefer_web_sourced_prices: bool = False,
) -> Optional[float]:
    """
    Prefer this facility's web-extracted price whenever present; otherwise DB/staging transparency price.
    prefer_web_sourced_prices is kept for call-site compatibility but does not gate per-row web display.
    """
    wp = row.get("web_candidate_price")
    web_pf = float(wp) if isinstance(wp, (int, float)) and wp and float(wp) > 0 else None
    if web_pf is not None:
        return web_pf
    return _pick_transparency_price(row, payment_mode)


def _row_price_display_note(
    row: Dict[str, Any],
    payment_mode: str = "cash",
    prefer_web_sourced_prices: bool = False,
) -> str:
    wp = row.get("web_candidate_price")
    web_pf = float(wp) if isinstance(wp, (int, float)) and wp and float(wp) > 0 else None
    if web_pf is not None:
        return "Web search price"
    if _pick_transparency_price(row, payment_mode) is not None:
        pm = (payment_mode or "").lower()
        if pm.startswith("insur"):
            try:
                g = row.get("standard_charge_gross")
                if g is not None and float(g) == float(_pick_transparency_price(row, payment_mode) or 0):
                    return "Listed hospital charge (DB)"
            except (TypeError, ValueError):
                pass
        return "DB price"
    return "ESTIMATE (no DB price yet)"


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


async def lookup_procedure_display_for_cpt(conn: asyncpg.Connection, cpt_code: str) -> Optional[str]:
    """Patient-facing procedure name for results heading when the user searched by CPT only."""
    code = (cpt_code or "").strip()
    if not code:
        return None
    try:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(
                NULLIF(TRIM(sv.variant_name), ''),
                NULLIF(TRIM(s.service_description), ''),
                NULLIF(TRIM(s.cpt_explanation), '')
            ) AS lbl
            FROM public.service_variants sv
            LEFT JOIN public.services s ON s.code = sv.cpt_code AND s.code_type = 'CPT'
            WHERE sv.cpt_code = $1
            ORDER BY sv.id ASC
            LIMIT 1
            """,
            code,
        )
        if row and row["lbl"]:
            return str(row["lbl"]).strip()
        row2 = await conn.fetchrow(
            """
            SELECT COALESCE(
                NULLIF(TRIM(service_description), ''),
                NULLIF(TRIM(cpt_explanation), '')
            ) AS lbl
            FROM public.services
            WHERE code_type = 'CPT' AND code = $1
            LIMIT 1
            """,
            code,
        )
        if row2 and row2["lbl"]:
            return str(row2["lbl"]).strip()
    except Exception as e:
        logger.warning(f"CPT procedure label lookup failed: {e}")
    return None


def _extract_cpt_procedure_name_from_web_snippets(cpt_code: str, results: List[Dict[str, Any]]) -> Optional[str]:
    """Use snippets from authoritative coding / Medicare sources; no name without snippet support."""
    code = (cpt_code or "").strip()
    if not code or not results:
        return None
    lines = []
    for r in results[:8]:
        url = ((r.get("url") or "")[:180]).strip()
        title = (r.get("title") or "")[:400]
        snip = (r.get("snippet") or "")[:1200]
        lines.append(f"URL: {url}\nTitle: {title}\nSnippet: {snip}")
    ctx = "\n\n".join(lines)
    system = (
        "You read web search snippets that may quote CPT (Current Procedural Terminology) descriptors. "
        "CPT is AMA-copyrighted; only use wording clearly tied to the requested code in the snippets. "
        f"For CPT code {code}, return a concise procedure title ONLY if the snippets explicitly describe "
        "what that code represents. Do not invent or copy unrelated procedures. "
        'Return JSON only: {"procedure_name": string|null} — procedure_name: max 18 words, Title Case, '
        "must not contain the CPT numeric code."
    )
    user = f"CPT code: {code}\n\n--- Snippets ---\n{ctx}"
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=25,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw)
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        name = (obj.get("procedure_name") or "").strip()
        if not name or name.lower() == "null":
            return None
        if code in name:
            return None
        return name[:240]
    except Exception as e:
        logger.warning(f"CPT web snippet parse failed: {e}")
        return None


async def lookup_cpt_descriptor_via_web(cpt_code: str) -> Optional[str]:
    """
    When services/service_variants have no row, resolve a display name from the web.
    Search is biased toward CMS/Medicare and established coding-reference domains.
    """
    code = (cpt_code or "").strip()
    if not code:
        return None
    if not WEB_SEARCH_ENABLED or not (BRAVE_API_KEY or TAVILY_API_KEY):
        return None

    query = (
        f"CPT code {code} procedure descriptor "
        "site:cms.gov OR site:medicare.gov OR site:aapc.com OR site:findacode.com OR site:ama-assn.org"
    )
    results: List[Dict[str, Any]] = []
    if BRAVE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as http:
                resp = await http.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": BRAVE_API_KEY},
                    params={"q": query, "count": 8},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("web", {}).get("results", [])[:8]:
                        results.append(
                            {
                                "title": item.get("title", ""),
                                "snippet": item.get("description", ""),
                                "url": item.get("url", ""),
                                "source": "brave",
                            }
                        )
        except Exception as e:
            logger.warning(f"Brave CPT descriptor search failed: {e}")

    if not results and TAVILY_API_KEY:
        results = await _tavily_search(query, num_results=8)

    if not results:
        return None
    return _extract_cpt_procedure_name_from_web_snippets(code, results)


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


def extract_explicit_cpt_code(message: str) -> Optional[str]:
    """Match phrases like 'CPT code 70551' (avoids treating bare 5-digit numbers as CPT)."""
    m = re.search(r"(?i)\bCPT\s*(?:code)?\s*(\d{5})\b", message or "")
    return m.group(1).strip() if m else None


def extract_zip_from_price_message(message: str) -> Optional[str]:
    """Prefer explicit ZIP phrases from Quick Search templates (e.g. 'My ZIP is 06119')."""
    m = message or ""
    mm = re.search(r"(?i)\b(?:my\s+)?zip\s*(?:code)?\s*(?:is)?[:\s]+(\d{5})\b", m)
    if mm:
        return mm.group(1)
    mm = re.search(r"(?i)\bnear\s+(?:zip\s*)?(\d{5})\b", m)
    return mm.group(1) if mm else None


def resolve_zip_digit_groups(message: str) -> Optional[str]:
    """
    Pick a likely ZIP from the message. When both CPT and ZIP appear, do not use the CPT
    digits as the ZIP (first bare \\d{5} in the string is often CPT, not location).
    """
    z = extract_zip_from_price_message(message)
    if z:
        return z
    cpt = extract_explicit_cpt_code(message)
    blocks = re.findall(r"\b(\d{5})\b", message or "")
    if not blocks:
        return None
    if len(blocks) == 1 and cpt and blocks[0] == cpt:
        return None
    if cpt and len(blocks) > 1:
        for b in blocks:
            if b != cpt:
                return b
        return None
    return blocks[0]


# Carrier tokens for extract_carrier_from_text (same semantics as extract_intent inner helper).
_CARRIER_KEYWORDS: Dict[str, str] = {
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


def extract_carrier_from_text(message: str) -> Optional[str]:
    """Resolve a payer/carrier name from free text (Quick Search + chat templates)."""
    m = (message or "").strip()
    if not m:
        return None
    ml = m.lower()
    for k, v in _CARRIER_KEYWORDS.items():
        if k in ml:
            return v
    clean = m
    for stop in ["i have", "i use", "use", "with", "insurance", "my", "have", "paying"]:
        clean = re.sub(r"\b" + re.escape(stop) + r"\b", "", clean, flags=re.IGNORECASE)
    clean = clean.strip()
    tokens = re.findall(r"[a-zA-Z]+", clean)
    if len(tokens) == 1 and len(tokens[0]) >= 3:
        return tokens[0].title()
    if 2 <= len(tokens) <= 3:
        return " ".join([t.title() for t in tokens])
    return None


def apply_payment_hints_from_message(message: str, merged: Dict[str, Any], intent: Dict[str, Any]) -> None:
    """
    Fill payment_mode / payer_like when the user stated them in the same message as CPT/ZIP
    (extract_intent early exits often copy stale None from session — Gate 2 then asks for payment in a loop).
    """
    mode = (intent.get("mode") or "").lower()
    if mode in ("education", "care", "general"):
        return
    if not (
        merged.get("zipcode")
        or merged.get("code")
        or (merged.get("service_query") or "").strip()
    ):
        return

    msg = (message or "").strip()
    if not msg:
        return
    msg_l = msg.lower()
    cash_terms = {
        "cash",
        "self pay",
        "self-pay",
        "selfpay",
        "out of pocket",
        "oop",
        "paying cash",
        "pay cash",
        "cash pay",
        "self pay patient",
        "selfpay patient",
    }
    insurance_terms = {"insurance", "insured", "use insurance", "with insurance"}
    carrier = extract_carrier_from_text(msg)
    has_ins = any(t in msg_l for t in insurance_terms)
    has_cash = any(t in msg_l for t in cash_terms)

    if has_ins or carrier:
        merged["payment_mode"] = "insurance"
        merged["cash_only"] = False
        if carrier:
            merged["payer_like"] = carrier
        merged.pop("_awaiting", None)
        return
    if has_cash:
        merged["payment_mode"] = "cash"
        merged["cash_only"] = True
        merged["payer_like"] = None
        merged["plan_like"] = None
        merged.pop("_awaiting", None)


def apply_zip_hint_from_message(message: str, merged: Dict[str, Any]) -> None:
    """Force ZIP from the current user message so a changed ZIP overrides stale session state."""
    z = resolve_zip_digit_groups(message or "")
    if z:
        merged["zipcode"] = z


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

    # 0) User explicitly named a different CPT than the session — new procedure (Quick Search / chat).
    explicit_cpt = extract_explicit_cpt_code(msg)
    prev_code = (st.get("code") or "").strip()
    if explicit_cpt and explicit_cpt != prev_code:
        z = resolve_zip_digit_groups(msg) or st.get("zipcode")
        return {
            "mode": "price",
            "code_type": "CPT",
            "code": explicit_cpt,
            "zipcode": z,
            "payment_mode": st.get("payment_mode"),
            "payer_like": st.get("payer_like"),
            "plan_like": st.get("plan_like"),
            "cash_only": st.get("cash_only"),
            "clarifying_question": None,
            "_explicit_cpt_update": True,
        }

    # 0b) ZIP detection — prefer 'My ZIP is #####'; never treat CPT 5-digit codes as the ZIP.
    z = resolve_zip_digit_groups(msg)
    if z:
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

    # 4) If the user is asking a new price question, mark it as such so we reset session state.
    # This ensures variant selection happens even if we already have a ZIP from a previous query.
    # Also extract ZIP and payment info from the same message if present.
    inferred_service = infer_service_query_from_message(msg)
    if inferred_service:
        # Extract ZIP from this message if present (avoid using CPT digits as ZIP)
        extracted_zip = resolve_zip_digit_groups(msg)
        
        # Extract payment mode from this message if present
        extracted_payment = None
        extracted_payer = None
        if any(t in msg_l for t in cash_terms):
            extracted_payment = "cash"
        elif any(t in msg_l for t in insurance_terms):
            extracted_payment = "insurance"
            extracted_payer = extract_carrier(msg)
        
        # Always mark as new price question to trigger session reset and variant selection
        return {
            "mode": "price", 
            "service_query": inferred_service, 
            "zipcode": extracted_zip,
            "payment_mode": extracted_payment,
            "payer_like": extracted_payer,
            "clarifying_question": None, 
            "_new_price_question": True
        }

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
                    h.latitude,
                    h.longitude,
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
                    h.latitude,
                    h.longitude,
                    (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                    )) AS distance_miles,
                    pick.best_price,
                    pick.negotiated_dollar,
                    pick.estimated_amount,
                    pick.standard_charge_cash,
                    pick.standard_charge_gross,
                    pick.payer_name,
                    pick.plan_name
                FROM public.hospitals h
                JOIN LATERAL (
                    SELECT
                        COALESCE(nr.negotiated_dollar, nr.estimated_amount, nr.standard_charge_cash) AS best_price,
                        nr.negotiated_dollar,
                        nr.estimated_amount,
                        nr.standard_charge_cash,
                        nr.standard_charge_gross,
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
    # Extend to 300 miles for sparse regions (e.g. Nevada, rural states)
    radius_array = [10, 25, 50, 100, 200, 300]

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
def _pick_best_staging_rate_row(
    recs: List[Dict[str, Any]], payment_mode: str
) -> Optional[Dict[str, Any]]:
    """Choose one stg_hospital_rates row per hospital for cash vs insurance display logic."""
    if not recs:
        return None
    pm = (payment_mode or "cash").lower()

    def score(rec: Dict[str, Any]) -> Tuple[int, float]:
        sc = rec.get("standard_charge_discounted_cash")
        nd = rec.get("standard_charge_negotiated_dollar")
        ea = rec.get("estimated_amount")
        gr = rec.get("standard_charge_gross")
        if pm.startswith("insur"):
            # Prefer listed gross (transparency), then negotiated, then estimates / cash columns
            if gr is not None and float(gr) >= 400:
                return (0, -float(gr))
            if nd is not None:
                return (1, -float(nd))
            if ea is not None:
                return (2, -float(ea))
            if sc is not None:
                return (3, -float(sc))
            return (9, 0.0)
        # cash / self-pay
        if sc is not None:
            return (0, float(sc))
        if ea is not None:
            return (1, float(ea))
        if gr is not None:
            return (2, float(gr))
        if nd is not None:
            return (3, float(nd))
        return (9, 0.0)

    def usable(rec: Dict[str, Any]) -> bool:
        for k in (
            "standard_charge_discounted_cash",
            "standard_charge_negotiated_dollar",
            "estimated_amount",
            "standard_charge_gross",
        ):
            v = rec.get(k)
            if v is None:
                continue
            try:
                if float(v) > 0:
                    return True
            except (TypeError, ValueError):
                pass
        return False

    filtered = [r for r in recs if usable(r)]
    if not filtered:
        return None
    try:
        return min(filtered, key=lambda r: score(r))
    except (TypeError, ValueError):
        return None


def _apply_staging_rate_to_row(row: Dict[str, Any], staging: Dict[str, Any]) -> None:
    """Copy staging price fields onto a facility row in negotiated_rates-compatible shape."""
    sc = staging.get("standard_charge_discounted_cash")
    nd = staging.get("standard_charge_negotiated_dollar")
    ea = staging.get("estimated_amount")
    gr = staging.get("standard_charge_gross")
    row["standard_charge_cash"] = sc
    row["negotiated_dollar"] = nd
    row["estimated_amount"] = ea
    row["standard_charge_gross"] = gr
    row["payer_name"] = staging.get("payer_name") or row.get("payer_name")
    row["plan_name"] = staging.get("plan_name") or row.get("plan_name")
    try:
        bp = None
        for v in (nd, ea, sc):
            if isinstance(v, (int, float)) and float(v) > 0:
                bp = float(v)
                break
        if bp is None and isinstance(gr, (int, float)) and float(gr) > 0:
            bp = float(gr)
        if bp is not None:
            row["best_price"] = bp
    except (TypeError, ValueError):
        pass


async def enrich_facility_rows_from_staging(
    conn: asyncpg.Connection,
    rows: List[Dict[str, Any]],
    code_type: Optional[str],
    code: Optional[str],
    payment_mode: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
) -> None:
    """
    Fill missing transparency prices from public.stg_hospital_rates (CPT + hospital).
    Negotiated_rates is often sparse; staging MRF data frequently has rows negotiated_rates does not.
    """
    c = (code or "").strip()
    if not c or not rows:
        return
    ct = (code_type or "").strip() or None
    payer_f = (payer_like or "").strip() or None
    plan_f = (plan_like or "").strip() or None

    need_idx: List[int] = []
    for i, row in enumerate(rows):
        if _pick_transparency_price(row, payment_mode) is not None:
            continue
        need_idx.append(i)
    if not need_idx:
        return

    by_hid: Dict[int, List[int]] = {}
    name_by_idx: Dict[int, str] = {}

    for i in need_idx:
        row = rows[i]
        hid = row.get("hospital_id")
        try:
            if hid is not None:
                ih = int(hid)
                by_hid.setdefault(ih, []).append(i)
                continue
        except (TypeError, ValueError):
            pass
        nm = _pick_hospital_name(row)
        if nm:
            name_by_idx[i] = nm.strip().lower()

    if by_hid:
        hid_list = list(by_hid.keys())
        stg_rows = await conn.fetch(
            """
            SELECT
              r.hospital_id,
              r.hospital_name,
              r.standard_charge_discounted_cash,
              r.standard_charge_negotiated_dollar,
              r.estimated_amount,
              r.standard_charge_gross,
              r.payer_name,
              r.plan_name
            FROM public.stg_hospital_rates r
            WHERE r.hospital_id = ANY($1::int[])
              AND r.code = $2
              AND ($3::text IS NULL OR r.code_type IS NULL OR r.code_type = $3)
              AND ($4::text IS NULL OR r.payer_name ILIKE '%' || $4 || '%')
              AND ($5::text IS NULL OR r.plan_name ILIKE '%' || $5 || '%')
              AND (
                r.standard_charge_discounted_cash IS NOT NULL
                OR r.standard_charge_negotiated_dollar IS NOT NULL
                OR r.estimated_amount IS NOT NULL
                OR r.standard_charge_gross IS NOT NULL
              )
            """,
            hid_list,
            c,
            ct,
            payer_f,
            plan_f,
        )
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for r in stg_rows:
            hid = int(r["hospital_id"])
            grouped.setdefault(hid, []).append(dict(r))
        for hid, idxs in by_hid.items():
            best = _pick_best_staging_rate_row(grouped.get(hid, []), payment_mode)
            if not best:
                continue
            for ix in idxs:
                _apply_staging_rate_to_row(rows[ix], best)

    if name_by_idx:
        uniq_names = sorted({v for v in name_by_idx.values() if v})
        if uniq_names:
            stg_n = await conn.fetch(
                """
                SELECT
                  LOWER(TRIM(r.hospital_name)) AS hn,
                  r.standard_charge_discounted_cash,
                  r.standard_charge_negotiated_dollar,
                  r.estimated_amount,
                  r.standard_charge_gross,
                  r.payer_name,
                  r.plan_name
                FROM public.stg_hospital_rates r
                WHERE r.code = $1
                  AND ($2::text IS NULL OR r.code_type IS NULL OR r.code_type = $2)
                  AND ($3::text IS NULL OR r.payer_name ILIKE '%' || $3 || '%')
                  AND ($4::text IS NULL OR r.plan_name ILIKE '%' || $4 || '%')
                  AND LOWER(TRIM(r.hospital_name)) = ANY($5::text[])
                  AND (
                    r.standard_charge_discounted_cash IS NOT NULL
                    OR r.standard_charge_negotiated_dollar IS NOT NULL
                    OR r.estimated_amount IS NOT NULL
                    OR r.standard_charge_gross IS NOT NULL
                  )
                """,
                c,
                ct,
                payer_f,
                plan_f,
                uniq_names,
            )
            by_name: Dict[str, List[Dict[str, Any]]] = {}
            for r in stg_n:
                key = (r["hn"] or "").strip().lower()
                if not key:
                    continue
                by_name.setdefault(key, []).append(dict(r))
            for ix, hn in name_by_idx.items():
                if _pick_transparency_price(rows[ix], payment_mode) is not None:
                    continue
                best = _pick_best_staging_rate_row(by_name.get(hn, []), payment_mode)
                if best:
                    best.pop("hn", None)
                    _apply_staging_rate_to_row(rows[ix], best)


async def get_nearby_hospitals(conn: asyncpg.Connection, zipcode: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Returns nearby hospitals with name/address/state/zipcode/phone (+ distance if possible).
    Uses public.hospital_details for facility directory fields:
      name, address, state, zipcode, phone, latitude, longitude

    Important: we only consider facilities within a reasonable driving distance of the user's ZIP.
    Without a distance cap, the query ordered *all* rows globally by distance — if the table is
    sparse in the user's region (e.g. PA) but dense elsewhere (CT), the "nearest" five could be
    hundreds of miles away in another state.
    """
    z = await conn.fetchrow(
        "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1 LIMIT 1",
        zipcode,
    )
    zlat = float(z["latitude"]) if z and z["latitude"] is not None else None
    zlon = float(z["longitude"]) if z and z["longitude"] is not None else None

    dist_sql = """(
          3959 * acos(
              cos(radians($2)) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians($3)) +
              sin(radians($2)) * sin(radians(h.latitude))
          )
        )"""

    if zlat is not None and zlon is not None:
        # Expand radius until we find facilities; cap each step so we prefer truly local rows.
        # Stop at 300 mi: beyond that, "nearby" is misleading (e.g. CT vs PA); prefer empty + web enrichment.
        # Extend to 500 miles so rural/sparse states (Nevada, Wyoming, etc.) always get results
        radius_steps = (50, 100, 200, 300, 500)
        for max_miles in radius_steps:
            q = f"""
            SELECT
              h.hospital_id AS hospital_id,
              h.hospital_name AS hospital_name,
              h.address AS address,
              h.state AS state,
              h.zipcode AS zipcode,
              h.phone AS phone,
              h.latitude AS latitude,
              h.longitude AS longitude,
              {dist_sql} AS distance_miles
            FROM public.hospital_details h
            WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
              AND {dist_sql} <= $4
            ORDER BY distance_miles ASC
            LIMIT $1
            """
            rows = await conn.fetch(q, limit, zlat, zlon, float(max_miles))
            if rows:
                return [dict(r) for r in rows]

            q_fallback = f"""
            SELECT
              h.id AS hospital_id,
              h.name AS hospital_name,
              h.address AS address,
              h.state AS state,
              h.zipcode AS zipcode,
              h.phone AS phone,
              h.latitude AS latitude,
              h.longitude AS longitude,
              {dist_sql} AS distance_miles
            FROM public.hospitals h
            WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
              AND {dist_sql} <= $4
            ORDER BY distance_miles ASC
            LIMIT $1
            """
            rows = await conn.fetch(q_fallback, limit, zlat, zlon, float(max_miles))
            if rows:
                return [dict(r) for r in rows]

        return []

    # If we can't compute distance, still return facilities (join both tables)
    q2 = """
    SELECT
      COALESCE(hd.hospital_id, h.id) AS hospital_id,
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
def _heuristic_facility_charge_range(service_query: str, code: Optional[str], payment_mode: str) -> Optional[str]:
    """Rule-based ranges so we never show '$100–$300' for hospital MRI-class services."""
    u = f"{service_query or ''} {code or ''}".lower()
    ins = (payment_mode or "").lower().startswith("insur")
    if any(x in u for x in ("70551", "70552", "70553", "70554")) or ("mri" in u and "brain" in u):
        return "$1,000–$2,800" if ins else "$900–$2,500"
    if "mri" in u:
        return "$900–$3,200"
    if any(x in u for x in ("74176", "74177", "74178", "74150", "74160", "74170")):
        return "$800–$2,400"
    if "ct " in u or "ct scan" in u or "computed tomography" in u:
        return "$600–$2,000"
    if "colonoscopy" in u:
        return "$2,000–$6,500"
    return None


def estimate_cost_range(service_query: str, payment_mode: str, code: Optional[str] = None) -> str:
    """
    Returns a short '$X–$Y' range, explicitly an estimate.
    This is used ONLY when DB pricing is missing.
    """
    heur = _heuristic_facility_charge_range(service_query, code, payment_mode)
    if heur:
        return heur
    system = (
        "You output ONLY a short numeric range for typical U.S. HOSPITAL OUTPATIENT facility charges "
        "for the service (not $10 copays; the full negotiated/list facility side).\n"
        "Format: '$X–$Y'. No extra text.\n"
        "Imaging in hospitals is usually many hundreds to a few thousand dollars before patient OOP."
    )
    user = json.dumps({"service": service_query, "payment_mode": payment_mode, "cpt_or_code": code})
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


def _pick_transparency_price(row: Dict[str, Any], payment_mode: str) -> Optional[float]:
    """
    For insurance, prefer listed hospital/standard charge when present so we compare to
    facility totals (e.g. ~$1,500 MRI) rather than only plan-allowed amounts (~$200).
    """
    pm = (payment_mode or "cash").lower()
    if pm.startswith("insur"):
        g = row.get("standard_charge_gross")
        try:
            if g is not None:
                gf = float(g)
                if gf >= 400:
                    return gf
        except (TypeError, ValueError):
            pass
    return _pick_price_fields(row)


def _pick_hospital_name(row: Dict[str, Any]) -> str:
    s = (row.get("hospital_name") or row.get("name") or "Unknown facility").strip()
    s = re.sub(r"\++", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_phone(row: Dict[str, Any]) -> str:
    return (row.get("phone") or "").strip()


def _pick_address(row: Dict[str, Any]) -> str:
    """One-line address for display and web search. Avoid duplicating city/state/ZIP when street already has them."""
    addr = (row.get("address") or row.get("street_address") or "").strip()
    city = (row.get("city") or "").strip()
    state = (row.get("state") or "").strip()
    z = (str(row.get("zipcode") or row.get("zip") or "")).strip()
    addr_l = addr.lower()

    # Street line often already includes ", City, ST 12345" — do not append state/zip again
    if addr and (
        re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\s*$", addr, re.I)
        or re.search(r",\s*[^,]+\s*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\s*$", addr, re.I)
    ):
        s = addr
    elif addr and z and z in addr.replace(" ", ""):
        s = addr
    else:
        parts = [addr] if addr else []
        if city and city.lower() not in addr_l:
            parts.append(city)
        if state and state.lower() not in addr_l and (len(state) != 2 or f", {state.upper()}" not in addr.upper()):
            parts.append(state)
        if z and z not in addr:
            parts.append(z)
        s = ", ".join(p for p in parts if p).strip()

    s = s.replace("\xc2\xa0", " ").replace("\xa0", " ")
    s = re.sub(r"Â+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ", ", s)
    return s


def _facility_stable_key(row: Dict[str, Any]) -> str:
    hid = row.get("hospital_id")
    try:
        if hid is not None:
            return f"id:{int(hid)}"
    except (TypeError, ValueError):
        pass
    z = (str(row.get("zipcode") or row.get("zip") or "")).strip()
    return f"n:{_normalize_hospital_key(_pick_hospital_name(row))}|{z}"


def _public_facility_website_or_maps(row: Dict[str, Any]) -> str:
    """Prefer an explicit website column when present; else Google Maps search (always usable)."""
    import urllib.parse

    for k in ("website", "website_url", "hospital_website", "url"):
        w = (row.get(k) or "").strip()
        if not w or "@" in w:
            continue
        if w.startswith("http://") or w.startswith("https://"):
            return w
        if "." in w and " " not in w:
            return "https://" + w.lstrip("/")
    q = f"{_pick_hospital_name(row)} {_pick_address(row)}".strip()
    return "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote(q)


def _extract_lat_lon(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat, lon = row.get("latitude"), row.get("longitude")
    try:
        if lat is not None and lon is not None:
            la, lo = float(lat), float(lon)
            if -90 <= la <= 90 and -180 <= lo <= 180:
                return la, lo
    except (TypeError, ValueError):
        pass
    return None, None


def _coords_by_hospital_key_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Map normalized hospital name -> (lat, lon) from the first row that has coordinates."""
    idx: Dict[str, Tuple[float, float]] = {}
    for row in rows:
        la, lo = _extract_lat_lon(row)
        if la is None:
            continue
        k = _normalize_hospital_key(_pick_hospital_name(row))
        if k and k not in idx:
            idx[k] = (la, lo)
    return idx


def _enrich_facilities_with_coordinates(
    facilities: List[Dict[str, Any]],
    priced_results: List[Dict[str, Any]],
    fallback_hospitals: List[Dict[str, Any]],
) -> None:
    """Copy lat/lon onto facility rows when the same hospital appears with geometry elsewhere."""
    idx = _coords_by_hospital_key_from_rows(list(priced_results) + list(fallback_hospitals))
    for f in facilities:
        if _extract_lat_lon(f)[0] is not None:
            continue
        k = _normalize_hospital_key(_pick_hospital_name(f))
        pair = idx.get(k)
        if pair:
            f["latitude"], f["longitude"] = pair[0], pair[1]


def _backfill_facilities_missing_coordinates(
    facilities: List[Dict[str, Any]],
    fallback_hospitals: List[Dict[str, Any]],
) -> None:
    """
    Replace facilities still lacking coordinates with the nearest fallback rows that have coords,
    so the map can show five hospitals relative to the ZIP (unique names only).
    """
    changed = True
    safety = 0
    while changed and safety < 12:
        safety += 1
        changed = False
        for i in range(len(facilities)):
            if _extract_lat_lon(facilities[i])[0] is not None:
                continue
            other_names = {
                _pick_hospital_name(facilities[j]).lower().strip()
                for j in range(len(facilities))
                if j != i
            }
            for h in fallback_hospitals:
                if _extract_lat_lon(h)[0] is None:
                    continue
                hn = (h.get("hospital_name") or "").strip().lower()
                if not hn or hn in other_names:
                    continue
                facilities[i] = dict(h)
                changed = True
                break


def _generate_google_maps_url(
    user_lat: float, 
    user_lon: float, 
    facilities: List[Dict[str, Any]],
    user_zipcode: str = ""
) -> str:
    """
    Generate a Google Maps URL showing the user's location and nearby hospitals.
    Uses the directions/search format to show multiple locations.
    """
    import urllib.parse
    
    valid_facilities = [f for f in facilities if _extract_lat_lon(f)[0] is not None]

    if not valid_facilities:
        return f"https://www.google.com/maps?q={user_lat},{user_lon}&z=12"

    if len(valid_facilities) == 1:
        f = valid_facilities[0]
        dest_lat, dest_lon = _extract_lat_lon(f)
        return f"https://www.google.com/maps/dir/{user_lat},{user_lon}/{dest_lat},{dest_lon}"

    markers = []
    for f in valid_facilities[:MIN_FACILITIES_TO_DISPLAY]:
        la, lo = _extract_lat_lon(f)
        if la is not None:
            markers.append(f"{la},{lo}")
    
    # Use Google Maps with multiple destinations (waypoints)
    # Format: origin -> waypoint1 -> waypoint2 -> ... -> final destination
    origin = f"{user_lat},{user_lon}"
    waypoints = "/".join(markers[:-1]) if len(markers) > 1 else ""
    destination = markers[-1] if markers else origin
    
    if waypoints:
        return f"https://www.google.com/maps/dir/{origin}/{waypoints}/{destination}"
    else:
        return f"https://www.google.com/maps/dir/{origin}/{destination}"


def _generate_static_map_url(
    user_lat: float,
    user_lon: float,
    facilities: List[Dict[str, Any]],
    api_key: str = ""
) -> Optional[str]:
    """
    Generate a Google Static Maps URL for embedding.
    Requires a Google Maps API key.
    Returns None if no API key is configured.
    """
    if not api_key:
        return None
    
    import urllib.parse
    
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    # User location marker (blue)
    markers = [f"color:blue|label:U|{user_lat},{user_lon}"]
    
    # Hospital markers (red, numbered)
    for i, f in enumerate(facilities[:MIN_FACILITIES_TO_DISPLAY], start=1):
        la, lo = _extract_lat_lon(f)
        if la is not None:
            markers.append(f"color:red|label:{i}|{la},{lo}")
    
    params = {
        "size": "600x400",
        "maptype": "roadmap",
        "key": api_key,
    }
    
    # Build URL with multiple markers params
    url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    for m in markers:
        url += f"&markers={urllib.parse.quote(m)}"
    
    return url


def _generate_map_data(
    user_lat: float,
    user_lon: float,
    user_zipcode: str,
    facilities: List[Dict[str, Any]],
    payment_mode: str = "cash",
    prefer_web_sourced_prices: bool = False,
    default_estimate_range: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate map data including URLs and coordinates for the frontend.
    """
    # Generate interactive map URL
    map_url = _generate_google_maps_url(user_lat, user_lon, facilities, user_zipcode)
    
    # Prepare facility coordinates for frontend map rendering
    facility_markers = []
    for i, f in enumerate(facilities[:MIN_FACILITIES_TO_DISPLAY], start=1):
        la, lo = _extract_lat_lon(f)
        if la is not None:
            eff = _row_effective_price(f, payment_mode, prefer_web_sourced_prices)
            facility_markers.append({
                "index": i,
                "list_index": i,
                "facility_key": _facility_stable_key(f),
                "name": _pick_hospital_name(f),
                "latitude": la,
                "longitude": lo,
                "address": _pick_address(f),
                "price": eff,
                "estimated_range": (default_estimate_range or "") if eff is None else None,
                "web_price": (
                    float(f["web_candidate_price"])
                    if f.get("web_candidate_price") is not None
                    and isinstance(f.get("web_candidate_price"), (int, float))
                    and float(f["web_candidate_price"]) > 0
                    else None
                ),
                "distance_miles": f.get("distance_miles"),
                "website_url": _public_facility_website_or_maps(f),
            })
    
    return {
        "user_location": {
            "latitude": user_lat,
            "longitude": user_lon,
            "zipcode": user_zipcode,
        },
        "facilities": facility_markers,
        "google_maps_url": map_url,
        "center": {
            "latitude": user_lat,
            "longitude": user_lon,
        },
        "zoom": 11,  # Good zoom level for ~25 mile radius
    }


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
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    user_zipcode: str = "",
    web_facility_catalog: Optional[List[Dict[str, Any]]] = None,
    procedure_code: Optional[str] = None,
    prefer_web_sourced_prices: bool = False,
    procedure_display_name: Optional[str] = None,
    db_price_discrepancy: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (text_answer, facilities_payload_for_ui, map_data).
    Always produces at least MIN_FACILITIES_TO_DISPLAY facilities in text when possible.
    map_data contains coordinates and Google Maps URL for displaying a map.
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

    # Ensure lat/lon for map pins: match priced rows to nearby directory, then swap in coords-only rows
    if facilities:
        _enrich_facilities_with_coordinates(facilities, priced_results, fallback_hospitals)
        _backfill_facilities_missing_coordinates(facilities, fallback_hospitals)

    # If still empty after DB + web top-up, explain options (directory gap vs missing web keys)
    if not facilities:
        return (
            "I couldn’t assemble a facility list for that ZIP from the data available on this run. "
            "If **Brave** or **Tavily** search is configured, try again—results can use web-discovered hospitals "
            "even when our local facility table has no rows near that ZIP. You can also try a nearby ZIP in the same metro.",
            [],
            None,
        )

    # Prepare an estimate range if needed
    est_range = estimate_cost_range(
        service_query or "this service", payment_mode or "cash", procedure_code
    )

    # Build answer text
    bullets = build_service_education_bullets(service_query or "this service", payment_mode or "cash")
    if (payment_mode or "").lower().startswith("insur"):
        bullets.append(
            "- With insurance, transparency files often include **listed hospital charges** (similar to “total fees”) "
            "as well as **plan-specific amounts**; your **copay** can be much lower than the facility total."
        )

    lines = []
    lines.extend(bullets)
    lines.append("")
    if db_price_discrepancy:
        mw = db_price_discrepancy.get("median_web", 0)
        md = db_price_discrepancy.get("median_db", 0)
        lines.append(
            f"> ⚠️ **Note:** Our database shows a median price of "
            f"**{_format_money(md)}**, but recent web research across "
            f"these facilities shows a median of **{_format_money(mw)}**. "
            "The database entry may be outdated — prices below reflect current web research where available."
        )
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

        price = _row_effective_price(f, payment_mode or "cash", prefer_web_sourced_prices)
        if price is not None:
            price_txt = _format_money(price)
            price_note = _row_price_display_note(f, payment_mode or "cash", prefer_web_sourced_prices)
        else:
            price_txt = est_range
            price_note = _row_price_display_note(f, payment_mode or "cash", prefer_web_sourced_prices)

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

    if web_facility_catalog:
        lines.append("")
        lines.append("**Web search — price found (per hospital):**")
        for j, w in enumerate(web_facility_catalog, start=1):
            wname = (w.get("hospital_name") or "Unknown facility").strip()
            wpx = w.get("web_price")
            if isinstance(wpx, (int, float)) and wpx > 0:
                lines.append(f"{j}) **{wname}** — **{_format_money(float(wpx))}** (from web)")

    lines.append("")
    lines.append("Confirm with the facility and your insurer.")
    
    # Add map link if we have user coordinates
    if user_lat is not None and user_lon is not None and facilities:
        map_url = _generate_google_maps_url(user_lat, user_lon, facilities, user_zipcode)
        lines.append("")
        lines.append(f"📍 [View all locations on map]({map_url})")

    # Build facility payload for UI
    ui_payload = []
    for f in facilities[:MIN_FACILITIES_TO_DISPLAY]:
        eff = _row_effective_price(f, payment_mode or "cash", prefer_web_sourced_prices)
        wraw = f.get("web_candidate_price")
        web_pf = (
            float(wraw)
            if isinstance(wraw, (int, float)) and wraw and float(wraw) > 0
            else None
        )
        plat, plon = _extract_lat_lon(f)
        if web_pf is not None:
            psrc = "web_search"
        elif eff is not None:
            psrc = "db"
        else:
            psrc = None
        ui_payload.append(
            {
                "hospital_name": f.get("hospital_name") or _pick_hospital_name(f),
                "address": _pick_address(f),
                "phone": f.get("phone") or _pick_phone(f),
                "distance_miles": f.get("distance_miles") or f.get("distance"),
                "price": eff,
                "web_price": web_pf,
                "estimated_range": None if eff is not None else est_range,
                "price_is_estimate": eff is None,
                "price_source": psrc,
                "latitude": plat,
                "longitude": plon,
                "facility_key": _facility_stable_key(f),
                "website_url": _public_facility_website_or_maps(f),
            }
        )

    # Generate map data for frontend
    map_data = None
    if user_lat is not None and user_lon is not None:
        map_data = _generate_map_data(
            user_lat,
            user_lon,
            user_zipcode,
            facilities,
            payment_mode or "cash",
            prefer_web_sourced_prices,
            default_estimate_range=est_range,
        )
        if map_data is not None:
            if procedure_display_name:
                map_data["procedure_display_name"] = procedure_display_name.strip()
            if procedure_code:
                map_data["procedure_code"] = str(procedure_code).strip()

    return "\n".join(lines), ui_payload, map_data


# ----------------------------
# Universal Service Variant Gate (DB-first)
# ----------------------------

def _normalize_service_text(s: str) -> str:
    """Normalize text for search matching.
    
    Removes all non-alphanumeric characters and normalizes whitespace.
    This ensures "X-ray", "x-ray", "xray", "X ray" all become "xray".
    Works for ANY term, not just hardcoded ones.
    """
    s = (s or "").lower().strip()
    # Remove ALL non-alphanumeric characters (hyphens, slashes, etc.)
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Remove spaces between single letters (e.g., "c t scan" -> "ct scan", "m r i" -> "mri")
    # This handles cases like "X ray" -> "xray" generically
    s = re.sub(r'\b([a-z])\s+(?=[a-z]\b)', r'\1', s)
    return s

def _normalize_for_sql(column: str) -> str:
    """Generate SQL expression to normalize a column value for matching.
    
    Removes hyphens, slashes, and other punctuation from DB values
    so they can match normalized user input.
    """
    # Remove common punctuation that might appear in medical terms
    return f"LOWER(REGEXP_REPLACE(COALESCE({column}, ''), '[^a-zA-Z0-9 ]', '', 'g'))"

def _tokenize_service_text(s: str) -> List[str]:
    s = _normalize_service_text(s)
    toks = [t for t in s.split(" ") if t]
    # Keep only reasonably informative tokens
    toks = [t for t in toks if len(t) >= 2]
    return toks

async def search_service_variants_by_text(conn, user_text: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Search service_variants for any service type using variant_name and patient_summary.
    Also joins with services table to get cpt_explanation for better matching.

    Strategy:
    - Normalize ALL text generically (remove punctuation, normalize spaces).
    - Use same normalization on DB values via SQL REGEXP_REPLACE.
    - No hardcoded special cases - works for any medical term.
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
    
    logger.info(f"Searching service_variants with tokens: {toks} from query: {user_text}")

    # Build OR match conditions and a score (count of matched tokens)
    # Use REGEXP_REPLACE to normalize DB values the same way we normalize user input
    # This handles ALL punctuation generically (hyphens, slashes, etc.)
    where_parts = []
    score_parts = []
    params = []
    
    # SQL normalization: remove all non-alphanumeric except spaces, then lowercase
    norm_sv_variant = "LOWER(REGEXP_REPLACE(COALESCE(sv.variant_name, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
    norm_sv_summary = "LOWER(REGEXP_REPLACE(COALESCE(sv.patient_summary, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
    norm_sv_parent = "LOWER(REGEXP_REPLACE(COALESCE(sv.parent_service, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
    norm_s_expl = "LOWER(REGEXP_REPLACE(COALESCE(s.cpt_explanation, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
    norm_s_desc = "LOWER(REGEXP_REPLACE(COALESCE(s.service_description, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
    
    for i, tok in enumerate(toks, start=1):
        like = f"%{tok}%"
        params.append(like)
        where_parts.append(f"""(
            {norm_sv_variant} LIKE ${i} 
            OR {norm_sv_summary} LIKE ${i}
            OR {norm_sv_parent} LIKE ${i}
            OR {norm_s_expl} LIKE ${i}
            OR {norm_s_desc} LIKE ${i}
        )""")
        score_parts.append(f"""(
            CASE WHEN {norm_sv_variant} LIKE ${i} THEN 2 ELSE 0 END +
            CASE WHEN {norm_sv_summary} LIKE ${i} THEN 1 ELSE 0 END +
            CASE WHEN {norm_sv_parent} LIKE ${i} THEN 2 ELSE 0 END +
            CASE WHEN {norm_s_expl} LIKE ${i} THEN 1 ELSE 0 END
        )""")

    where_sql = " OR ".join(where_parts)
    score_sql = " + ".join(score_parts)

    sql = f"""
        SELECT sv.id, sv.parent_service, sv.cpt_code, sv.variant_name, sv.patient_summary, sv.is_preventive,
               s.cpt_explanation, s.service_description,
               ({score_sql}) AS match_score
        FROM public.service_variants sv
        LEFT JOIN public.services s ON s.code = sv.cpt_code AND s.code_type = 'CPT'
        WHERE ({where_sql})
        ORDER BY match_score DESC, sv.parent_service NULLS LAST, sv.variant_name NULLS LAST, sv.id ASC
        LIMIT {int(limit)}
    """
    try:
        rows = await conn.fetch(sql, *params)
        if rows:
            logger.info(f"service_variants found {len(rows)} results for: {user_text}")
            return [dict(r) for r in rows]
        
        logger.info(f"service_variants empty, trying services table fallback for: {user_text}")
        
        # Fallback: if service_variants is empty or has no matches, search services table directly
        norm_s_desc_f = "LOWER(REGEXP_REPLACE(COALESCE(s.service_description, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
        norm_s_summary_f = "LOWER(REGEXP_REPLACE(COALESCE(s.patient_summary, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
        norm_s_expl_f = "LOWER(REGEXP_REPLACE(COALESCE(s.cpt_explanation, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
        norm_s_cat_f = "LOWER(REGEXP_REPLACE(COALESCE(s.category, ''), '[^a-zA-Z0-9 ]', '', 'g'))"
        
        where_parts_s = []
        score_parts_s = []
        for i, tok in enumerate(toks, start=1):
            where_parts_s.append(f"""(
                {norm_s_desc_f} LIKE ${i} 
                OR {norm_s_summary_f} LIKE ${i}
                OR {norm_s_expl_f} LIKE ${i}
                OR {norm_s_cat_f} LIKE ${i}
            )""")
            score_parts_s.append(f"""(
                CASE WHEN {norm_s_desc_f} LIKE ${i} THEN 2 ELSE 0 END +
                CASE WHEN {norm_s_expl_f} LIKE ${i} THEN 1 ELSE 0 END +
                CASE WHEN {norm_s_cat_f} LIKE ${i} THEN 2 ELSE 0 END
            )""")
        
        where_sql_s = " OR ".join(where_parts_s)
        score_sql_s = " + ".join(score_parts_s)
        
        fallback_sql = f"""
            SELECT s.id, s.category as parent_service, s.code as cpt_code, 
                   s.service_description as variant_name, s.patient_summary, 
                   false as is_preventive, s.cpt_explanation, s.service_description,
                   ({score_sql_s}) AS match_score
            FROM public.services s
            WHERE s.code_type = 'CPT' AND ({where_sql_s})
            ORDER BY match_score DESC, s.category NULLS LAST, s.service_description NULLS LAST
            LIMIT {int(limit)}
        """
        rows = await conn.fetch(fallback_sql, *params)
        if rows:
            logger.info(f"services fallback found {len(rows)} results for: {user_text}")
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"service_variants search failed: {e}")
        return []

def build_variant_numbered_prompt(user_label: str, variants: List[Dict[str, Any]]) -> str:
    base = (_normalize_service_text(user_label) or "this service")
    header = base.upper() if len(base) <= 22 else "this service"
    lines: List[str] = []
    lines.append(f"Before I look up prices, which exact billed **{header}** do you mean?")
    lines.append("")
    for i, v in enumerate(variants, start=1):
        name = (v.get("variant_name") or "Option").strip()
        cpt = (v.get("cpt_code") or "").strip()
        # Support both parent_service (from service_variants) and category (from services fallback)
        category = (v.get("parent_service") or v.get("category") or "").strip()
        
        # Build the line with name, CPT, and category
        line = f"{i}) {name}"
        if cpt:
            line += f" (CPT {cpt})"
        if category:
            line += f" [{category}]"
        lines.append(line)
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
                if req.fresh_context:
                    state = {}
                yield sse({"type": "session", "session_id": session_id})
                await save_message(conn, session_id, "user", req.message)

                # 1) Intent + state merge
                intent = await extract_intent(req.message, state)
                merged = merge_state(state, intent)

                if intent.get("_explicit_cpt_update"):
                    merged.pop("service_query", None)
                    merged.pop("_awaiting", None)
                    for k in (
                        "variant_id",
                        "variant_name",
                        "variant_confirmed",
                        "_variant_candidates",
                        "_variant_single",
                    ):
                        merged.pop(k, None)

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
                    # CRITICAL: Reset code fields to force variant selection for new service
                    merged.pop("code_type", None)
                    merged.pop("code", None)
                    merged.pop("variant_confirmed", None)
                    merged.pop("variant_id", None)
                    merged.pop("variant_name", None)
                    merged.pop("_variant_candidates", None)
                    merged.pop("_variant_single", None)
                    logger.info(f"Reset code fields for new price question: {intent.get('service_query')}")
                apply_zip_hint_from_message(req.message, merged)
                apply_payment_hints_from_message(req.message, merged, intent)
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
                    if WEB_SEARCH_ENABLED and (BRAVE_API_KEY or TAVILY_API_KEY):
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

                        # If we still don't have a code and we're not already awaiting a gate, search variants.
                        if not (merged.get("code_type") and merged.get("code")) and merged.get("_awaiting") not in {"zip", "payment", "payer", "variant_choice", "variant_confirm_yesno", "variant_clarify"}:
                            qtext = merged.get("service_query") or req.message
                            raw_candidates = await search_service_variants_by_text(conn, qtext, limit=8)

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
                                    "variant_name": c.get("variant_name"),
                                    "patient_summary": c.get("patient_summary"),
                                    "cpt_explanation": c.get("cpt_explanation"),
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

                    pay_mode = merged.get("payment_mode") or "cash"

                    try:
                        nearby_hospitals = await get_nearby_hospitals(
                            conn, merged["zipcode"], limit=NEARBY_COORD_LOOKUP_LIMIT
                        )
                    except Exception as e:
                        logger.warning(f"Nearby hospitals lookup failed: {e}")
                        nearby_hospitals = []

                    # Always run web hospital discovery regardless of DB results.
                    # Web results represent current, real-world facilities; DB may be stale or regionally sparse.
                    # We merge both: DB rows first (they have coordinates for map), web fills gaps and confirms.
                    if WEB_SEARCH_ENABLED and (BRAVE_API_KEY or TAVILY_API_KEY):
                        try:
                            web_discovered = await synthetic_nearby_hospitals_from_web(
                                str(merged.get("zipcode") or "")
                            )
                            if web_discovered:
                                existing_keys = {_normalize_hospital_key(h.get('hospital_name', '')) for h in nearby_hospitals}
                                web_only: List[Dict[str, Any]] = []
                                for wh in web_discovered:
                                    wkey = _normalize_hospital_key(wh.get('hospital_name', ''))
                                    if not wkey:
                                        continue
                                    if wkey not in existing_keys:
                                        # Net-new facility from web — append after DB rows
                                        web_only.append(wh)
                                        existing_keys.add(wkey)
                                    # (DB row already present — web search confirmed it exists; keep DB version for coords)
                                nearby_hospitals = nearby_hospitals + web_only
                                nearby_hospitals = nearby_hospitals[:NEARBY_COORD_LOOKUP_LIMIT]
                                logger.info(
                                    "Hospital list: %d from DB, %d net-new from web (ZIP %s)",
                                    len(nearby_hospitals) - len(web_only),
                                    len(web_only),
                                    merged.get('zipcode'),
                                )
                                # Queue new web-discovered hospitals for nightly enrichment
                                if web_only:
                                    try:
                                        await queue_hospitals_for_enrichment(
                                            conn, web_only, str(merged.get("zipcode") or "")
                                        )
                                    except Exception as _qe:
                                        logger.warning("Failed to queue hospitals for enrichment: %s", _qe)
                        except Exception as e:
                            logger.warning("Synthetic nearby from web failed: %s", e)

                    web_price_attempted = False
                    web_vals_all: List[float] = []
                    web_facility_catalog: List[Dict[str, Any]] = []
                    probed_targets: List[Dict[str, Any]] = []
                    results: List[Dict[str, Any]] = []
                    used_max_radius: Optional[int] = None
                    prefer_web_sourced_prices = False
                    results_for_block: List[Dict[str, Any]] = []
                    fallback_for_block: List[Dict[str, Any]] = []
                    web_first_applied = False

                    web_first = (
                        PRICE_WEB_FIRST
                        and WEB_PRICE_SEARCH_ENABLED
                        and WEB_SEARCH_ENABLED
                        and (BRAVE_API_KEY or TAVILY_API_KEY)
                        and len(nearby_hospitals) > 0
                    )

                    if web_first:
                        try:
                            (
                                web_price_attempted,
                                web_vals_all,
                                web_facility_catalog,
                                probed_targets,
                            ) = await attach_web_prices_to_facility_results(
                                [],
                                merged.get("service_query") or "",
                                merged.get("code") or "",
                                pay_mode,
                                fill_hospitals=nearby_hospitals,
                            )
                        except Exception as e:
                            logger.warning(f"Web facility price enrichment failed: {e}")
                        if not probed_targets:
                            web_first = False

                    if web_first:
                        # DB lookup for reference only (merge onto nearest hospitals already web-searched)
                        results, used_max_radius = await price_lookup_progressive(
                            conn,
                            merged["zipcode"],
                            merged["code_type"],
                            merged["code"],
                            merged.get("service_query") or "",
                            merged.get("payer_like"),
                            merged.get("plan_like"),
                            pay_mode,
                        )
                        merge_db_price_fields_into_rows(probed_targets, results)
                        try:
                            await enrich_facility_rows_from_staging(
                                conn,
                                probed_targets,
                                merged.get("code_type"),
                                merged.get("code"),
                                pay_mode,
                                merged.get("payer_like"),
                                merged.get("plan_like"),
                            )
                        except Exception as e:
                            logger.warning(f"Staging table price enrichment failed: {e}")
                        for r in probed_targets:
                            r["db_reference_price"] = _pick_transparency_price(r, pay_mode)
                        prefer_web_sourced_prices = bool(web_price_attempted and web_vals_all)
                        results_for_block = probed_targets
                        fallback_for_block = []
                        web_first_applied = True
                    else:
                        results, used_max_radius = await price_lookup_progressive(
                            conn,
                            merged["zipcode"],
                            merged["code_type"],
                            merged["code"],
                            merged.get("service_query") or "",
                            merged.get("payer_like"),
                            merged.get("plan_like"),
                            pay_mode,
                        )
                        try:
                            await enrich_facility_rows_from_staging(
                                conn,
                                results,
                                merged.get("code_type"),
                                merged.get("code"),
                                pay_mode,
                                merged.get("payer_like"),
                                merged.get("plan_like"),
                            )
                            await enrich_facility_rows_from_staging(
                                conn,
                                nearby_hospitals,
                                merged.get("code_type"),
                                merged.get("code"),
                                pay_mode,
                                merged.get("payer_like"),
                                merged.get("plan_like"),
                            )
                        except Exception as e:
                            logger.warning(f"Staging table price enrichment failed: {e}")
                        try:
                            (
                                web_price_attempted,
                                web_vals_all,
                                web_facility_catalog,
                                probed_targets,
                            ) = await attach_web_prices_to_facility_results(
                                results,
                                merged.get("service_query") or "",
                                merged.get("code") or "",
                                pay_mode,
                                fill_hospitals=nearby_hospitals,
                            )
                        except Exception as e:
                            logger.warning(f"Web facility price enrichment failed: {e}")
                        for r in probed_targets:
                            r["db_reference_price"] = _pick_transparency_price(r, pay_mode)
                        prefer_web_sourced_prices = bool(web_price_attempted and web_vals_all)
                        results_for_block = results
                        fallback_for_block = nearby_hospitals

                    disc_detail = (
                        analyze_web_vs_db_discrepancy(probed_targets, pay_mode) if probed_targets else None
                    )
                    if disc_detail:
                        await record_price_discrepancy_alert(
                            conn,
                            session_id,
                            str(merged.get("zipcode") or ""),
                            str(merged.get("code") or "") if merged.get("code") else None,
                            pay_mode,
                            disc_detail,
                        )

                    # Get user's coordinates for map
                    user_lat, user_lon = None, None
                    try:
                        zip_row = await conn.fetchrow(
                            "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1",
                            merged["zipcode"]
                        )
                        if zip_row:
                            user_lat = float(zip_row["latitude"]) if zip_row["latitude"] else None
                            user_lon = float(zip_row["longitude"]) if zip_row["longitude"] else None
                    except Exception as e:
                        logger.warning(f"Failed to get user coordinates: {e}")

                    procedure_display_name: Optional[str] = None
                    if (merged.get("code_type") or "").upper() == "CPT" and merged.get("code"):
                        cpt_c = str(merged.get("code") or "")
                        procedure_display_name = await lookup_procedure_display_for_cpt(conn, cpt_c)
                        if not procedure_display_name:
                            procedure_display_name = await lookup_cpt_descriptor_via_web(cpt_c)

                    facility_text, facility_payload, map_data = build_facility_block(
                        service_query=merged.get("service_query") or "this service",
                        payment_mode=merged.get("payment_mode") or "cash",
                        priced_results=results_for_block,
                        fallback_hospitals=fallback_for_block,
                        user_lat=user_lat,
                        user_lon=user_lon,
                        user_zipcode=merged.get("zipcode") or "",
                        web_facility_catalog=web_facility_catalog if web_facility_catalog else None,
                        procedure_code=merged.get("code"),
                        prefer_web_sourced_prices=prefer_web_sourced_prices,
                        procedure_display_name=procedure_display_name,
                        db_price_discrepancy=disc_detail,
                    )

                    state_out = {
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
                    }
                    if procedure_display_name:
                        state_out["procedure_display_name"] = procedure_display_name
                    if merged.get("code"):
                        state_out["procedure_code"] = merged.get("code")
                    if disc_detail:
                        state_out["price_discrepancy_flag"] = True

                    yield sse(
                        {
                            "type": "results",
                            "results": results_for_block[:25],
                            "facilities": facility_payload,
                            "web_facility_prices": web_facility_catalog,
                            "map_data": map_data,
                            "state": state_out,
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
                            "result_count": len(results_for_block),
                            "used_max_radius": used_max_radius,
                            "refiner_id": (refiner or {}).get("id"),
                            "price_web_first": web_first_applied,
                            "price_discrepancy": bool(disc_detail),
                        },
                    )
                    await update_session_state(conn, session_id, merged)

                    used_web = len(results) < MIN_DB_RESULTS_BEFORE_WEB or web_price_attempted
                    await log_query(
                        conn,
                        session_id,
                        req.message,
                        intent,
                        used_max_radius,
                        len(results_for_block),
                        used_web,
                        full_answer,
                    )
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


@app.get("/admin/price_discrepancies")
async def admin_price_discrepancies(request: Request, limit: int = 50):
    """
    Super-admin: recent web-vs-DB price discrepancy alerts.
    Header: X-Admin-Key: <ADMIN_API_KEY>
    Requires table public.admin_price_discrepancy_alerts (see record_price_discrepancy_alert docstring).
    """
    if not ADMIN_API_KEY:
        raise HTTPException(503, detail="ADMIN_API_KEY is not configured")
    if request.headers.get("X-Admin-Key", "") != ADMIN_API_KEY:
        raise HTTPException(403, detail="Invalid or missing X-Admin-Key")
    if pool is None:
        raise HTTPException(503, detail="Database pool not ready")
    lim = max(1, min(int(limit), 200))
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT id, created_at, session_id, zipcode, code, payment_mode, detail_json
                FROM public.admin_price_discrepancy_alerts
                ORDER BY id DESC
                LIMIT $1
                """,
                lim,
            )
        except Exception as e:
            raise HTTPException(
                503,
                detail=f"Could not read admin_price_discrepancy_alerts (create table?): {e}",
            ) from e
    out: List[Dict[str, Any]] = []
    for r in rows:
        ca = r["created_at"]
        out.append(
            {
                "id": r["id"],
                "created_at": ca.isoformat() if hasattr(ca, "isoformat") else str(ca),
                "session_id": r["session_id"],
                "zipcode": r["zipcode"],
                "code": r["code"],
                "payment_mode": r["payment_mode"],
                "detail": r["detail_json"],
            }
        )
    return out


@app.post("/chat")

async def chat(_req: ChatRequest, _request: Request):
    raise HTTPException(410, detail="Use /chat_stream")