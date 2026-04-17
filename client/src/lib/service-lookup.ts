import { query } from "@/lib/postgres";

export type ServiceSearchHint = {
  code: string;
  serviceDescription: string | null;
  cptExplanation: string | null;
  patientSummary: string | null;
  searchHint: string | null;
};

export async function lookupServiceSearchHint(code: string): Promise<ServiceSearchHint | null> {
  const normalized = code.trim();
  if (!normalized) return null;

  const result = await query<{
    code: string;
    service_description: string | null;
    cpt_explanation: string | null;
    patient_summary: string | null;
  }>(
    `
      SELECT code, service_description, cpt_explanation, patient_summary
      FROM services
      WHERE code = $1
      LIMIT 1
    `,
    [normalized]
  );

  const row = result.rows[0];
  if (!row) return null;

  const searchHint =
    row.cpt_explanation?.trim() ||
    row.patient_summary?.trim() ||
    row.service_description?.trim() ||
    null;

  return {
    code: row.code,
    serviceDescription: row.service_description,
    cptExplanation: row.cpt_explanation,
    patientSummary: row.patient_summary,
    searchHint,
  };
}
