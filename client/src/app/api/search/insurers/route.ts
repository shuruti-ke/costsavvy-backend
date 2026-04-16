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

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const searchCare = searchParams.get("searchCare")?.trim() || "";
  const zipCode = searchParams.get("zipCode")?.trim() || "";
  const limit = Math.max(parseInt(searchParams.get("limit") || "20", 10), 1);

  const clauses: string[] = [];
  const params: unknown[] = [];
  if (searchCare) {
    params.push(`%${escapeLike(searchCare)}%`);
    clauses.push(`
      (
        COALESCE(s.code, '') ILIKE $${params.length} ESCAPE '\\'
        OR COALESCE(s.service_description, '') ILIKE $${params.length} ESCAPE '\\'
        OR COALESCE(s.cpt_explanation, '') ILIKE $${params.length} ESCAPE '\\'
        OR COALESCE(s.patient_summary, '') ILIKE $${params.length} ESCAPE '\\'
      )
    `);
  }
  if (zipCode) {
    const range = parseZipRange(zipCode);
    if (range) {
      params.push(range.lower, range.upper);
      clauses.push(`CAST(h.zipcode AS INT) BETWEEN $${params.length - 1} AND $${params.length}`);
    }
  }

  const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
  const result = await query<{ reporting_entity_name_in_network_files: string }>(
    `
      SELECT DISTINCT COALESCE(ip.payer_name, ip.plan_name, '') AS reporting_entity_name_in_network_files
      ${HEALTHCARE_FROM}
      ${where}
      ORDER BY reporting_entity_name_in_network_files ASC
      LIMIT $${params.length + 1}
    `,
    [...params, limit]
  );
  const values = (result.rows as Array<{ reporting_entity_name_in_network_files: string | null }>)
    .map((row) => row.reporting_entity_name_in_network_files)
    .filter((value): value is string => Boolean(value));

  return NextResponse.json({ success: true, count: values.length, data: values });
}
