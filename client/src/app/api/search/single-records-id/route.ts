import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";

export const runtime = "nodejs";

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
  const id = searchParams.get("Id")?.trim() || "";

  if (!id) {
    return NextResponse.json(
      { success: false, message: "Missing required `id` param" },
      { status: 400 }
    );
  }

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
      WHERE h.id::text = $1 OR nr.hospital_id::text = $1 OR nr.id::text = $1
      LIMIT 1
    `,
    [id]
  );

  return NextResponse.json({ success: true, data: docs.rows });
}
