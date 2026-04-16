import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import {
  extractExplicitCptCode,
  extractInsurance,
  extractServiceQuery,
  extractZip,
  parseZipRange,
} from "@/lib/search-utils";
import { geocodeZip } from "@/lib/geocode";

export const runtime = "nodejs";

type SessionState = {
  serviceQuery: string;
  zip: string;
  insurance: string;
  cptCode: string;
};

const sessions = new Map<string, SessionState>();

function getSessionId(body: { session_id?: string }) {
  return body.session_id || `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function escapeLike(input: string) {
  return input.replace(/[\\%_]/g, "\\$&");
}

function parseNegotiatedRate(value: unknown) {
  if (typeof value === "number") return value;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/[^0-9.]/g, "");
  if (!cleaned) return null;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatAddress(record: Record<string, unknown>) {
  const address = String(record.provider_address || "").trim();
  const city = String(record.provider_city || "").trim();
  const state = String(record.provider_state || "").trim();
  const zip = String(record.provider_zip_code || "").trim();
  return [address, city, state, zip].filter(Boolean).join(", ");
}

function haversineMiles(
  a: { latitude: number; longitude: number },
  b: { latitude: number; longitude: number }
) {
  const toRad = (n: number) => (n * Math.PI) / 180;
  const R = 3958.8;
  const dLat = toRad(b.latitude - a.latitude);
  const dLon = toRad(b.longitude - a.longitude);
  const lat1 = toRad(a.latitude);
  const lat2 = toRad(b.latitude);
  const sinLat = Math.sin(dLat / 2);
  const sinLon = Math.sin(dLon / 2);
  const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

async function geocodeProviderZip(zip: unknown) {
  const value = String(zip || "").trim();
  if (!value) return null;
  const coords = await geocodeZip(value);
  return coords ? { latitude: coords.latitude, longitude: coords.longitude } : null;
}

const HEALTHCARE_FROM = `
  FROM negotiated_rates nr
  JOIN services s ON s.id = nr.service_id
  JOIN hospitals h ON h.id = nr.hospital_id
  LEFT JOIN insurance_plans ip ON ip.id = nr.plan_id
  LEFT JOIN zip_locations z ON z.zipcode = h.zipcode
`;

const HEALTHCARE_LABEL = `
  COALESCE(
    NULLIF(TRIM(CONCAT(s.code, ' ', s.service_description)), ''),
    s.service_description,
    s.code
  )
`;

export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as {
    session_id?: string;
    message?: string;
    fresh_context?: boolean;
  };

  const sessionId = getSessionId(body);
  const current = body.fresh_context ? null : sessions.get(sessionId) || null;
  const message = String(body.message || "").trim();

  const serviceQuery = extractServiceQuery(message);
  const cptCode = extractExplicitCptCode(message) || current?.cptCode || "";
  const zip = extractZip(message) || current?.zip || "";
  const insuranceRaw = extractInsurance(message) || current?.insurance || "";
  const insurance = insuranceRaw === "insurance" ? "" : insuranceRaw;

  const nextState: SessionState = {
    serviceQuery: serviceQuery || current?.serviceQuery || "",
    zip,
    insurance,
    cptCode,
  };
  sessions.set(sessionId, nextState);

  if (!nextState.serviceQuery || !nextState.zip) {
    const clarification = !nextState.serviceQuery
      ? "Tell me the procedure or service you want to price."
      : "Tell me your ZIP code so I can find nearby prices.";

    const transcript = [
      `data: ${JSON.stringify({ type: "session", session_id: sessionId })}`,
      `data: ${JSON.stringify({ type: "delta", text: clarification })}`,
      `data: ${JSON.stringify({ type: "final", text: clarification })}`,
      "",
    ].join("\n");

    return new NextResponse(transcript, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  }

  const clauses: string[] = [];
  const params: unknown[] = [];
  const serviceSearch = nextState.cptCode || nextState.serviceQuery;
  params.push(`%${escapeLike(serviceSearch)}%`);
  const serviceParam = `$${params.length}`;
  clauses.push(`
    (
      ${HEALTHCARE_LABEL} ILIKE ${serviceParam} ESCAPE '\\'
      OR COALESCE(s.cpt_explanation, '') ILIKE ${serviceParam} ESCAPE '\\'
      OR COALESCE(s.patient_summary, '') ILIKE ${serviceParam} ESCAPE '\\'
      OR COALESCE(s.code, '') ILIKE ${serviceParam} ESCAPE '\\'
    )
  `);
  if (nextState.cptCode) {
    params.push(nextState.cptCode);
    const codeParam = `$${params.length}`;
    clauses.push(`(
      COALESCE(s.code, '') = ${codeParam}
      OR COALESCE(s.code, '') ILIKE ${codeParam} ESCAPE '\\'
    )`);
  }

  const zipRange = parseZipRange(nextState.zip);
  if (zipRange) {
    params.push(zipRange.lower, zipRange.upper);
    clauses.push(`CAST(h.zipcode AS INT) BETWEEN $${params.length - 1} AND $${params.length}`);
  }

  let insuranceMatchExpr = "false";
  if (nextState.insurance) {
    params.push(`%${escapeLike(nextState.insurance)}%`);
    insuranceMatchExpr = `COALESCE(ip.payer_name, ip.plan_name, '') ILIKE $${params.length} ESCAPE '\\'`;
  }
  const rowChoiceOrderExpr = `
    CASE WHEN ${insuranceMatchExpr} THEN 0 ELSE 1 END,
    COALESCE(nr.standard_charge_cash, nr.estimated_amount, nr.standard_charge_min) ASC NULLS LAST,
    nr.last_updated DESC NULLS LAST,
    nr.id ASC
  `;

  const where = `WHERE ${clauses.join(" AND ")}`;
  const docs = await query<Record<string, unknown>>(
    `
      WITH ranked AS (
        SELECT
          nr.id::text AS _id,
          ${HEALTHCARE_LABEL} AS billing_code_name,
          COALESCE(s.code_type, '') AS billing_code_type,
          COALESCE(ip.payer_name, ip.plan_name, '') AS reporting_entity_name_in_network_files,
          ${insuranceMatchExpr} AS insurance_match,
          h.zipcode AS provider_zip_code,
          h.name AS provider_name,
          h.address AS provider_address,
          COALESCE(z.city, '') AS provider_city,
          COALESCE(z.state, h.state, '') AS provider_state,
          COALESCE(nr.standard_charge_cash, nr.estimated_amount, nr.standard_charge_min) AS negotiated_rate,
          COALESCE(s.cpt_explanation, s.patient_summary, '') AS "Description of Service",
          ROW_NUMBER() OVER (
            PARTITION BY h.id, s.id
            ORDER BY ${rowChoiceOrderExpr}
          ) AS rn
        ${HEALTHCARE_FROM}
        ${where}
      )
      SELECT *
      FROM ranked
      WHERE rn = 1
      ORDER BY insurance_match DESC, billing_code_name NULLS LAST, provider_name NULLS LAST
      LIMIT 12
    `,
    params
  );

  const userCoords = (await geocodeZip(nextState.zip)) || { latitude: 39.8283, longitude: -98.5795 };
  const facilities = [];

  for (const [index, doc] of docs.rows.entries()) {
    const rate = parseNegotiatedRate(doc.negotiated_rate);
    const providerCoords = await geocodeProviderZip(doc.provider_zip_code);
    const distance = providerCoords ? Number(haversineMiles(userCoords, providerCoords).toFixed(1)) : null;
    const name = String(doc.provider_name || doc.billing_code_name || "Healthcare Facility");
    facilities.push({
      list_index: index + 1,
      facility_key: String(doc.id || doc._id || `${index}`),
      name,
      latitude: providerCoords?.latitude ?? userCoords.latitude,
      longitude: providerCoords?.longitude ?? userCoords.longitude,
      address: formatAddress(doc),
      price: rate,
      estimated_range: rate != null ? null : "Contact for pricing",
      website_url: null,
      distance_miles: distance,
      phone: null,
      web_price: null,
      price_is_estimate: rate == null,
      price_source: rate != null ? "db" : "estimate",
      hospital_name: name,
      insurance_match: Boolean(doc.insurance_match),
    });
  }

  const prices = facilities.map((f) => f.price).filter((p): p is number => p != null);
  const insuranceMatches = facilities.filter((f) => f.insurance_match).length;
  const text =
    prices.length > 0
      ? nextState.insurance && insuranceMatches > 0
        ? `Found ${facilities.length} nearby price option${facilities.length === 1 ? "" : "s"} for ${nextState.serviceQuery}. ${insuranceMatches} result${insuranceMatches === 1 ? "" : "s"} appear to match ${nextState.insurance}.`
        : nextState.insurance
          ? `Found ${facilities.length} nearby price option${facilities.length === 1 ? "" : "s"} for ${nextState.serviceQuery}. I didn't find a clear ${nextState.insurance} match, so I'm showing the nearby facilities for the procedure.`
          : `Found ${facilities.length} nearby price option${facilities.length === 1 ? "" : "s"} for ${nextState.serviceQuery}.`
      : `I found ${facilities.length} possible matches for ${nextState.serviceQuery}, but pricing is estimated or unavailable.`;

  const mapData = {
    center: userCoords,
    user_location: {
      ...userCoords,
      zipcode: nextState.zip,
    },
    facilities: facilities.slice(0, 5).map((facility) => ({
      list_index: facility.list_index,
      facility_key: facility.facility_key,
      name: facility.name,
      latitude: facility.latitude,
      longitude: facility.longitude,
      address: facility.address,
      price: facility.price,
      estimated_range: facility.estimated_range,
      website_url: facility.website_url,
      insurance_match: facility.insurance_match,
    })),
    google_maps_url: `https://www.google.com/maps/search/${encodeURIComponent(nextState.zip)}`,
    procedure_code: nextState.cptCode || undefined,
    procedure_display_name: nextState.serviceQuery,
    service_query: nextState.serviceQuery,
  };

  const transcript = [
    `data: ${JSON.stringify({ type: "session", session_id: sessionId })}`,
    `data: ${JSON.stringify({ type: "delta", text })}`,
    `data: ${JSON.stringify({
      type: "results",
      map_data: mapData,
      facilities,
      state: {
        code: nextState.cptCode || undefined,
        procedure_display_name: nextState.serviceQuery,
        service_query: nextState.serviceQuery,
        zip: nextState.zip,
        insurance: nextState.insurance || undefined,
      },
    })}`,
    `data: ${JSON.stringify({ type: "final", text })}`,
    "",
  ].join("\n");

  return new NextResponse(transcript, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
