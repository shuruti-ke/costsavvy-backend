import { z } from "zod";
import { query } from "@/lib/postgres";

const learningSchema = z.object({
  normalizedServiceName: z.string().nullable().default(null),
  normalizedHospitalName: z.string().nullable().default(null),
  cptCode: z.string().nullable().default(null),
  zipCode: z.string().nullable().default(null),
  insurer: z.string().nullable().default(null),
  shouldLearn: z.boolean().default(false),
  confidence: z.number().min(0).max(1).default(0),
  rationale: z.string().nullable().default(null),
});

type LearningInput = {
  source: "chat" | "search";
  message: string;
  cptCode?: string;
  serviceQuery?: string;
  zip?: string;
  insurance?: string;
  rows: Array<Record<string, unknown>>;
};

type LearningResult = z.infer<typeof learningSchema>;
type ModelMessage = { role: "system" | "user" | "assistant"; content: string };

let schemaReady: Promise<void> | null = null;

function normalizeText(value: unknown) {
  return String(value || "").trim();
}

function inferCodeType(code: string) {
  return /^\d{5}$/.test(code) ? "CPT" : "CUSTOM";
}

function getNvidiaApiKey() {
  return process.env.NVIDIA_API_KEY || process.env.NEMOTRON_API_KEY || "";
}

function getNvidiaModel() {
  return (
    process.env.NVIDIA_MODEL ||
    process.env.COSTSAVVY_AI_MODEL ||
    "nvidia/nemotron-3-super-120b-a12b"
  );
}

async function callNemotron(messages: ModelMessage[]) {
  const apiKey = getNvidiaApiKey();
  if (!apiKey) return null;

  const response = await fetch(
    process.env.NVIDIA_BASE_URL || "https://integrate.api.nvidia.com/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: getNvidiaModel(),
        temperature: 0.2,
        messages,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Nemotron request failed with status ${response.status}`);
  }

  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string | null } }>;
  };
  return data.choices?.[0]?.message?.content?.trim() || null;
}

function parseLearningResponse(raw: string | null): LearningResult | null {
  if (!raw) return null;

  const trimmed = raw.trim();
  const cleaned = trimmed
    .replace(/^```(?:json)?/i, "")
    .replace(/```$/i, "")
    .trim();

  try {
    const parsed = JSON.parse(cleaned);
    return learningSchema.parse(parsed);
  } catch {
    return null;
  }
}

export async function ensureSearchLearningSchema() {
  if (!schemaReady) {
    schemaReady = query(`
      CREATE TABLE IF NOT EXISTS search_learnings (
        id SERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        query_text TEXT NOT NULL,
        cpt_code TEXT,
        service_query TEXT,
        zip_code TEXT,
        insurer TEXT,
        normalized_service_name TEXT,
        confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
        should_learn BOOLEAN NOT NULL DEFAULT FALSE,
        rationale TEXT,
        result_count INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT now()
      );

      CREATE TABLE IF NOT EXISTS service_search_aliases (
        id SERIAL PRIMARY KEY,
        code_type TEXT NOT NULL,
        code TEXT NOT NULL,
        alias_text TEXT NOT NULL,
        source_query TEXT NOT NULL,
        confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
        learned_at TIMESTAMP DEFAULT now(),
        UNIQUE (code_type, code, alias_text)
      );

      CREATE TABLE IF NOT EXISTS hospital_search_aliases (
        id SERIAL PRIMARY KEY,
        hospital_name TEXT NOT NULL,
        alias_text TEXT NOT NULL,
        source_query TEXT NOT NULL,
        confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
        learned_at TIMESTAMP DEFAULT now(),
        UNIQUE (hospital_name, alias_text)
      );

      CREATE TABLE IF NOT EXISTS search_learning_reviews (
        id SERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        query_text TEXT NOT NULL,
        cpt_code TEXT,
        service_query TEXT,
        hospital_name TEXT,
        zip_code TEXT,
        insurer TEXT,
        suggested_alias TEXT,
        suggested_hospital_name TEXT,
        confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
        rationale TEXT,
        result_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT now(),
        reviewed_at TIMESTAMP
      );
    `).then(() => undefined);
  }

  return schemaReady;
}

async function upsertServiceRow(args: {
  code: string;
  codeType: string;
  serviceDescription?: string | null;
  cptExplanation?: string | null;
  patientSummary?: string | null;
  category?: string | null;
}) {
  const result = await query<{ id: number }>(
    `
      INSERT INTO services (
        code,
        code_type,
        service_description,
        cpt_explanation,
        patient_summary,
        category
      )
      VALUES ($1, $2, $3, $4, $5, $6)
      ON CONFLICT (code_type, code) DO UPDATE
      SET
        service_description = COALESCE(EXCLUDED.service_description, services.service_description),
        cpt_explanation = COALESCE(EXCLUDED.cpt_explanation, services.cpt_explanation),
        patient_summary = COALESCE(EXCLUDED.patient_summary, services.patient_summary),
        category = COALESCE(EXCLUDED.category, services.category)
      RETURNING id
    `,
    [
      args.code,
      args.codeType,
      args.serviceDescription || null,
      args.cptExplanation || null,
      args.patientSummary || null,
      args.category || null,
    ]
  );

  return result.rows[0]?.id ?? null;
}

async function upsertHospitalRow(args: {
  name: string;
  address?: string | null;
  state?: string | null;
  zipcode?: string | number | null;
  description?: string | null;
  latitude?: number | null;
  longitude?: number | null;
}) {
  const result = await query<{ id: number }>(
    `
      INSERT INTO hospitals (
        name,
        address,
        state,
        zipcode,
        description,
        latitude,
        longitude
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7)
      ON CONFLICT (name) DO UPDATE
      SET
        address = COALESCE(EXCLUDED.address, hospitals.address),
        state = COALESCE(EXCLUDED.state, hospitals.state),
        zipcode = COALESCE(EXCLUDED.zipcode, hospitals.zipcode),
        description = COALESCE(EXCLUDED.description, hospitals.description),
        latitude = COALESCE(EXCLUDED.latitude, hospitals.latitude),
        longitude = COALESCE(EXCLUDED.longitude, hospitals.longitude)
      RETURNING id
    `,
    [
      args.name,
      args.address || null,
      args.state || null,
      args.zipcode != null ? String(args.zipcode) : null,
      args.description || null,
      args.latitude ?? null,
      args.longitude ?? null,
    ]
  );

  return result.rows[0]?.id ?? null;
}

async function upsertInsurancePlan(args: { payerName: string; planName?: string | null }) {
  const result = await query<{ id: number }>(
    `
      INSERT INTO insurance_plans (
        payer_name,
        plan_name
      )
      VALUES ($1, $2)
      ON CONFLICT (payer_name, plan_name) DO UPDATE
      SET
        payer_name = EXCLUDED.payer_name,
        plan_name = EXCLUDED.plan_name
      RETURNING id
    `,
    [args.payerName, args.planName || null]
  );

  return result.rows[0]?.id ?? null;
}

async function upsertNegotiatedRate(args: {
  hospitalId: number;
  serviceId: number;
  planId: number;
  negotiatedRate?: number | null;
  notes?: string | null;
}) {
  const existing = await query<{ id: number }>(
    `
      SELECT id
      FROM negotiated_rates
      WHERE hospital_id = $1 AND service_id = $2 AND plan_id = $3
      ORDER BY last_updated DESC NULLS LAST, id DESC
      LIMIT 1
    `,
    [args.hospitalId, args.serviceId, args.planId]
  );

  if (existing.rows[0]?.id) {
    await query(
      `
        UPDATE negotiated_rates
        SET
          standard_charge_cash = COALESCE($1, standard_charge_cash),
          estimated_amount = COALESCE($1, estimated_amount),
          standard_charge_min = COALESCE($1, standard_charge_min),
          additional_generic_notes = COALESCE($2, additional_generic_notes),
          last_updated = now()
        WHERE id = $3
      `,
      [args.negotiatedRate ?? null, args.notes || null, existing.rows[0].id]
    );
    return existing.rows[0].id;
  }

  const inserted = await query<{ id: number }>(
    `
      INSERT INTO negotiated_rates (
        hospital_id,
        service_id,
        plan_id,
        standard_charge_cash,
        estimated_amount,
        standard_charge_min,
        additional_generic_notes,
        last_updated
      )
      VALUES ($1, $2, $3, $4, $4, $4, $5, now())
      RETURNING id
    `,
    [
      args.hospitalId,
      args.serviceId,
      args.planId,
      args.negotiatedRate ?? null,
      args.notes || null,
    ]
  );

  return inserted.rows[0]?.id ?? null;
}

export async function recordSearchLearning(input: LearningInput) {
  const message = normalizeText(input.message);
  if (!message) return { learned: false, ai: null as LearningResult | null };

  const rows = input.rows.slice(0, 5).map((row) => ({
    providerName: normalizeText(row.provider_name),
    providerAddress: normalizeText(row.provider_address),
    providerCity: normalizeText(row.provider_city),
    providerState: normalizeText(row.provider_state),
    providerZipCode: normalizeText(row.provider_zip_code),
    codeName: normalizeText(row.billing_code_name),
    codeType: normalizeText(row.billing_code_type),
    payerName: normalizeText(row.reporting_entity_name_in_network_files),
    negotiatedRate: typeof row.negotiated_rate === "number" ? row.negotiated_rate : null,
    description: normalizeText(row["Description of Service"]),
  }));

  let ai: LearningResult | null = null;
  try {
    const content = await callNemotron([
      {
        role: "system",
        content:
          "You normalize healthcare search queries for a pricing app. Only use the user message and the returned search rows. Never invent hospitals or prices. If the query clearly refers to a CPT/service already present in the rows or to the explicit CPT code, or if it clearly refers to a hospital/provider name present in the rows, mark shouldLearn true and provide a normalized service name or hospital name plus an alias-friendly phrase. Return ONLY valid JSON matching this shape: {normalizedServiceName:string|null,normalizedHospitalName:string|null,cptCode:string|null,zipCode:string|null,insurer:string|null,shouldLearn:boolean,confidence:number,rationale:string|null}.",
      },
      {
        role: "user",
        content: JSON.stringify({
          source: input.source,
          message,
          extracted: {
            cptCode: input.cptCode || "",
            serviceQuery: input.serviceQuery || "",
            zip: input.zip || "",
            insurance: input.insurance || "",
          },
          topResults: rows,
        }),
      },
    ]);
    ai = parseLearningResponse(content);
  } catch (error) {
    console.warn("Search learning AI step failed:", error);
  }

  const confidence = ai?.confidence ?? 0;
  const shouldLearn = Boolean(ai?.shouldLearn && confidence >= 0.55);
  const cptCode = normalizeText(ai?.cptCode || input.cptCode || "");
  const zipCode = normalizeText(ai?.zipCode || input.zip || "");
  const insurer = normalizeText(ai?.insurer || input.insurance || "");
  const serviceQuery = normalizeText(input.serviceQuery);
  const normalizedServiceName = normalizeText(ai?.normalizedServiceName);
  const normalizedHospitalName = normalizeText(ai?.normalizedHospitalName);

  await ensureSearchLearningSchema();

  await query(
    `
      INSERT INTO search_learnings (
        source,
        query_text,
        cpt_code,
        service_query,
        zip_code,
        insurer,
        normalized_service_name,
        confidence,
        should_learn,
        rationale,
        result_count
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    `,
    [
      input.source,
      message,
      cptCode || null,
      serviceQuery || null,
      zipCode || null,
      insurer || null,
      normalizedServiceName || null,
      confidence,
      shouldLearn,
      ai?.rationale || null,
      rows.length,
    ]
  );

  let serviceId: number | null = null;
  const codeType = cptCode ? inferCodeType(cptCode) : "CPT";
  const serviceDescription =
    normalizedServiceName ||
    rows.find((row) => row.codeName)?.codeName ||
    serviceQuery ||
    message;

  if (cptCode) {
    serviceId = await upsertServiceRow({
      code: cptCode,
      codeType,
      serviceDescription,
      cptExplanation: normalizedServiceName || serviceDescription,
      patientSummary: ai?.rationale || serviceQuery || serviceDescription,
      category: "learned",
    });
  }

  if (cptCode && shouldLearn) {
    const aliasText = normalizedServiceName || serviceQuery || message;
    if (aliasText && serviceId) {
      await query(
        `
          INSERT INTO service_search_aliases (
            code_type,
            code,
            alias_text,
            source_query,
            confidence
          )
          VALUES ($1, $2, $3, $4, $5)
          ON CONFLICT (code_type, code, alias_text) DO UPDATE
          SET
            source_query = EXCLUDED.source_query,
            confidence = GREATEST(service_search_aliases.confidence, EXCLUDED.confidence),
            learned_at = now()
        `,
        [codeType, cptCode, aliasText, message, confidence]
      );
    }
  }

  const topHospitalName =
    normalizedHospitalName ||
    normalizeText(rows[0]?.providerName);

  if (topHospitalName && confidence >= 0.55) {
    await query(
      `
        INSERT INTO hospital_search_aliases (
          hospital_name,
          alias_text,
          source_query,
          confidence
        )
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (hospital_name, alias_text) DO UPDATE
        SET
          source_query = EXCLUDED.source_query,
          confidence = GREATEST(hospital_search_aliases.confidence, EXCLUDED.confidence),
          learned_at = now()
      `,
      [topHospitalName, serviceQuery || message, message, confidence]
    );
  }

  if (!shouldLearn || confidence < 0.55) {
    await query(
      `
        INSERT INTO search_learning_reviews (
          source,
          query_text,
          cpt_code,
          service_query,
          hospital_name,
          zip_code,
          insurer,
          suggested_alias,
          suggested_hospital_name,
          confidence,
          rationale,
          result_count
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
      `,
      [
        input.source,
        message,
        cptCode || null,
        serviceQuery || null,
        topHospitalName || null,
        zipCode || null,
        insurer || null,
        normalizedServiceName || null,
        normalizedHospitalName || topHospitalName || null,
        confidence,
        ai?.rationale || null,
        rows.length,
      ]
    );
  }

  for (const row of rows) {
    if (!row.providerName) continue;
    const hospitalId = await upsertHospitalRow({
      name: row.providerName,
      address: row.providerAddress || null,
      state: row.providerState || null,
      zipcode: row.providerZipCode || null,
      description: row.description || normalizedServiceName || null,
    });

    const payerName = row.payerName || insurer || "Self-pay";
    const planId = await upsertInsurancePlan({
      payerName,
      planName: row.payerName || insurer ? null : "Cash price",
    });

    if (hospitalId && planId && serviceId) {
      await upsertNegotiatedRate({
        hospitalId,
        serviceId,
        planId,
        negotiatedRate: row.negotiatedRate ?? null,
        notes: ai?.rationale || null,
      });
    }
  }

  return {
    learned: shouldLearn,
    ai,
  };
}
