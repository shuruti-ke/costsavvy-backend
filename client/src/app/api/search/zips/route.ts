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
  const entity = searchParams.get("entity")?.trim() || "";
  const search = searchParams.get("search")?.trim() || "";

  if (!entity) {
    return NextResponse.json(
      { success: false, message: "Missing required `entity` param" },
      { status: 400 }
    );
  }

  const params: unknown[] = [`%${escapeLike(entity)}%`];
  let where = `WHERE ${HEALTHCARE_LABEL} ILIKE $1 ESCAPE '\\'`;
  if (search) {
    params.push(`%${escapeLike(search)}%`);
    where += ` AND CAST(h.zipcode AS TEXT) ILIKE $2 ESCAPE '\\'`;
  }
  const result = await query<{ provider_zip_code: string | number }>(
    `
      SELECT DISTINCT h.zipcode AS provider_zip_code
      ${HEALTHCARE_FROM}
      ${where}
      ORDER BY provider_zip_code ASC
    `,
    params
  );
  const values = (result.rows as Array<{ provider_zip_code: string | number | null }>)
    .map((row) => String(row.provider_zip_code))
    .filter((value) => !search || value.includes(search));

  return NextResponse.json({ success: true, count: values.length, data: values });
}
