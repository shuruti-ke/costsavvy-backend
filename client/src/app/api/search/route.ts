import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { parseZipRange } from "@/lib/search-utils";
import { ensureSearchLearningSchema, recordSearchLearning } from "@/lib/search-learning";
import { lookupServiceSearchHint } from "@/lib/service-lookup";
import { searchWebPricing, type WebPriceCandidate } from "@/lib/web-pricing-search";

export const runtime = "nodejs";

function escapeLike(input: string) {
  return input.replace(/[\\%_]/g, "\\$&");
}

const HEALTHCARE_FROM = `
  FROM negotiated_rates nr
  JOIN services s ON s.id = nr.service_id
  JOIN hospitals h ON h.id = nr.hospital_id
  LEFT JOIN service_search_aliases sa
    ON sa.code_type = s.code_type AND sa.code = s.code
  LEFT JOIN hospital_search_aliases ha
    ON ha.hospital_name = h.name
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

async function lookupHospital(name: string) {
  if (!name) return null;
  const result = await query<{
    name: string;
    address: string | null;
    state: string | null;
    zipcode: string | null;
    latitude: number | null;
    longitude: number | null;
  }>(
    `
      SELECT name, address, state, zipcode, latitude, longitude
      FROM hospitals
      WHERE LOWER(name) = LOWER($1)
      LIMIT 1
    `,
    [name]
  );
  return result.rows[0] || null;
}

async function buildWebRows(
  candidates: WebPriceCandidate[],
  searchCare: string,
  zipCode: string,
  cptCode: string
) {
  const rows = await Promise.all(
    candidates.map(async (candidate, index) => {
      const hospital = await lookupHospital(candidate.hospitalName);
      return {
        _id: `web-${index}`,
        billing_code_name: cptCode ? `CPT code ${cptCode}` : searchCare,
        billing_code_type: cptCode ? "CPT" : "",
        reporting_entity_name_in_network_files: candidate.websiteUrl
          ? (() => {
              try {
                return new URL(candidate.websiteUrl).hostname.replace(/^www\./, "");
              } catch {
                return candidate.hospitalName;
              }
            })()
          : candidate.hospitalName,
        provider_zip_code: candidate.zipcode || hospital?.zipcode || zipCode,
        provider_name: candidate.hospitalName,
        provider_address: candidate.address || hospital?.address || null,
        provider_city: candidate.city || "",
        provider_state: candidate.state || hospital?.state || "",
        negotiated_rate: candidate.price,
        "Description of Service": candidate.sourceSnippet || candidate.sourceTitle || searchCare,
        price_source: candidate.price != null ? "web_search" : "web_candidate",
        web_price: candidate.price,
        insurance_match: candidate.insuranceMatch,
        website_url: candidate.websiteUrl,
        latitude: hospital?.latitude ?? null,
        longitude: hospital?.longitude ?? null,
      };
    })
  );

  return rows;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const searchCare = searchParams.get("searchCare")?.trim() || "";
  const zipCode = searchParams.get("zipCode")?.trim() || "";
  const insurance = searchParams.get("insurance")?.trim() || "";
  const page = Math.max(parseInt(searchParams.get("page") || "1", 10), 1);
  const limit = Math.max(parseInt(searchParams.get("limit") || "10", 10), 1);

  await ensureSearchLearningSchema();

  const cptCode = searchParams.get("cptCode")?.trim() || "";
  const serviceHint = cptCode ? await lookupServiceSearchHint(cptCode) : null;
  const webSearch = await searchWebPricing({
    serviceQuery: searchCare,
    cptCode,
    serviceDescription: serviceHint?.searchHint || serviceHint?.serviceDescription || undefined,
    zip: zipCode,
    insurance,
  });

  if (webSearch.results.length > 0) {
    const webRows = await buildWebRows(webSearch.results, searchCare, zipCode, cptCode);
    const hasVerifiedWebPrices = webSearch.results.some((candidate) => candidate.price != null);
    await recordSearchLearning({
      source: "search",
      message: [searchCare, zipCode, insurance, "web"].filter(Boolean).join(" | "),
      serviceQuery: searchCare,
      cptCode,
      zip: zipCode,
      insurance,
      rows: webRows,
      persistToDatabase: false,
    });

    return NextResponse.json({
      success: true,
      data: webRows.slice((page - 1) * limit, (page - 1) * limit + limit),
      pagination: {
        total: webRows.length,
        page,
        limit,
        source: hasVerifiedWebPrices ? "web" : "web_candidate",
      },
    });
  }

  const buildQuery = (relaxed: boolean) => {
    const clauses: string[] = [];
    const params: unknown[] = [];
    if (!relaxed && searchCare) {
      params.push(`%${escapeLike(searchCare)}%`);
      clauses.push(`
        (
          ${HEALTHCARE_LABEL} ILIKE $${params.length} ESCAPE '\\'
          OR COALESCE(s.cpt_explanation, '') ILIKE $${params.length} ESCAPE '\\'
          OR COALESCE(s.patient_summary, '') ILIKE $${params.length} ESCAPE '\\'
          OR COALESCE(s.code, '') ILIKE $${params.length} ESCAPE '\\'
          OR COALESCE(sa.alias_text, '') ILIKE $${params.length} ESCAPE '\\'
          OR COALESCE(ha.alias_text, '') ILIKE $${params.length} ESCAPE '\\'
          OR COALESCE(h.name, '') ILIKE $${params.length} ESCAPE '\\'
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
    if (insurance) {
      params.push(`%${escapeLike(insurance)}%`);
      clauses.push(`COALESCE(ip.payer_name, ip.plan_name, '') ILIKE $${params.length} ESCAPE '\\'`);
    }

    const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
    return { params, where };
  };

  const skip = (page - 1) * limit;
  let { params, where } = buildQuery(false);
  let totalResult = await query<{ total: string }>(
    `
      SELECT COUNT(*)::text AS total
      FROM (
        SELECT DISTINCT h.id, s.id
        ${HEALTHCARE_FROM}
        ${where}
      ) deduped
    `,
    params
  );
  let dataResult = await query(
    `
      WITH ranked AS (
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
          COALESCE(s.cpt_explanation, s.patient_summary, '') AS "Description of Service",
          ROW_NUMBER() OVER (
            PARTITION BY h.id, s.id
            ORDER BY COALESCE(nr.standard_charge_cash, nr.estimated_amount, nr.standard_charge_min) ASC NULLS LAST, nr.last_updated DESC NULLS LAST, nr.id ASC
          ) AS rn
        ${HEALTHCARE_FROM}
        ${where}
      )
      SELECT *
      FROM ranked
      WHERE rn = 1
      ORDER BY billing_code_name NULLS LAST, provider_name NULLS LAST
      LIMIT $${params.length + 1} OFFSET $${params.length + 2}
    `,
    [...params, limit, skip]
  );

  if (dataResult.rows.length === 0 && searchCare && zipCode) {
    const fallback = buildQuery(true);
    params = fallback.params;
    where = fallback.where;
    totalResult = await query<{ total: string }>(
      `
        SELECT COUNT(*)::text AS total
        FROM (
          SELECT DISTINCT h.id, s.id
          ${HEALTHCARE_FROM}
          ${where}
        ) deduped
      `,
      params
    );
    dataResult = await query(
      `
        WITH ranked AS (
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
            COALESCE(s.cpt_explanation, s.patient_summary, '') AS "Description of Service",
            ROW_NUMBER() OVER (
              PARTITION BY h.id, s.id
              ORDER BY COALESCE(nr.standard_charge_cash, nr.estimated_amount, nr.standard_charge_min) ASC NULLS LAST, nr.last_updated DESC NULLS LAST, nr.id ASC
            ) AS rn
          ${HEALTHCARE_FROM}
          ${where}
        )
        SELECT *
        FROM ranked
        WHERE rn = 1
        ORDER BY billing_code_name NULLS LAST, provider_name NULLS LAST
        LIMIT $${params.length + 1} OFFSET $${params.length + 2}
      `,
      [...params, limit, skip]
    );
  }

  await recordSearchLearning({
    source: "search",
    message: [searchCare, zipCode, insurance].filter(Boolean).join(" | "),
    serviceQuery: searchCare,
    zip: zipCode,
    insurance,
    rows: dataResult.rows,
  });

  return NextResponse.json({
    success: true,
    pagination: { total: Number(totalResult.rows[0]?.total || 0) },
    data: dataResult.rows,
  });
}
