import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";

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
  const search = searchParams.get("search")?.trim() || "";
  const limit = Math.max(parseInt(searchParams.get("limit") || "20", 10), 1);

  const params: unknown[] = [];
  let where = "";
  if (search) {
    params.push(`%${escapeLike(search)}%`);
    where = `WHERE ${HEALTHCARE_LABEL} ILIKE $1 ESCAPE '\\'`;
  }

  const result = await query<{ billing_code_name: string }>(
    `
      SELECT DISTINCT ${HEALTHCARE_LABEL} AS billing_code_name
      ${HEALTHCARE_FROM}
      ${where}
      ORDER BY billing_code_name ASC
      LIMIT $${params.length + 1}
    `,
    [...params, limit]
  );
  const values = (result.rows as Array<{ billing_code_name: string | null }>)
    .map((row) => row.billing_code_name)
    .filter((value): value is string => Boolean(value));

  return NextResponse.json({ success: true, count: values.length, data: values });
}
