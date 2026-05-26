import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { parseZipRange } from "@/lib/search-utils";

export const runtime = "nodejs";

function escapeLike(input: string) {
  return input.replace(/[\\%_]/g, "\\$&");
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

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const searchCare = searchParams.get("searchCare")?.trim() || "";
  const zipCode = searchParams.get("zipCode")?.trim() || "";
  const page = Math.max(parseInt(searchParams.get("page") || "1", 10), 1);
  const limit = Math.max(parseInt(searchParams.get("limit") || "100", 10), 1);
  const skip = (page - 1) * limit;

  if (!searchCare) {
    return NextResponse.json(
      { success: false, message: "Missing required `searchCare` param" },
      { status: 400 }
    );
  }

  const clauses: string[] = [];
  const params: unknown[] = [];
  params.push(`%${escapeLike(searchCare)}%`);
  clauses.push(`
    (
      ${HEALTHCARE_LABEL} ILIKE $1 ESCAPE '\\'
      OR COALESCE(s.cpt_explanation, '') ILIKE $1 ESCAPE '\\'
      OR COALESCE(s.patient_summary, '') ILIKE $1 ESCAPE '\\'
      OR COALESCE(s.code, '') ILIKE $1 ESCAPE '\\'
    )
  `);
  if (zipCode) {
    const range = parseZipRange(zipCode);
    if (range) {
      params.push(range.lower, range.upper);
      clauses.push(`CAST(h.zipcode AS INT) BETWEEN $${params.length - 1} AND $${params.length}`);
    }
  }

  const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
  const totalResult = await query<{ total: string }>(
    `SELECT COUNT(*)::text AS total ${HEALTHCARE_FROM} ${where}`,
    params
  );
  const docs = await query(
    `
      SELECT
        nr.id::text AS _id,
        ${HEALTHCARE_LABEL} AS billing_code_name,
        COALESCE(s.code_type, '') AS billing_code_type,
        COALESCE(ip.payer_name, ip.plan_name, '') AS reporting_entity_name_in_network_files,
        h.zipcode AS provider_zip_code,
        h.name AS provider_name,
        h.address AS provider_address,
        COALESCE(z.city, '') AS provider_city,
        COALESCE(z.state, h.state, '') AS provider_state,
        COALESCE(nr.standard_charge_cash, nr.estimated_amount, nr.standard_charge_min) AS negotiated_rate,
        COALESCE(s.cpt_explanation, s.patient_summary, '') AS "Description of Service"
      ${HEALTHCARE_FROM}
      ${where}
      ORDER BY billing_code_name NULLS LAST, provider_name NULLS LAST
      LIMIT $${params.length + 1} OFFSET $${params.length + 2}
    `,
    [...params, limit, skip]
  );

  return NextResponse.json({
    success: true,
    pagination: { total: Number(totalResult.rows[0]?.total || 0), page, limit },
    data: docs.rows,
  });
}
