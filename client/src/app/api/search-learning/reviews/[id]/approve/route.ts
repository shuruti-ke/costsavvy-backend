import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { requireAdmin } from "@/lib/admin-auth";
import { ensureSearchLearningSchema } from "@/lib/search-learning";

export const runtime = "nodejs";

function normalizeText(value: unknown) {
  return String(value || "").trim();
}

function inferCodeType(code: string) {
  return /^\d{5}$/.test(code) ? "CPT" : "CUSTOM";
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const admin = await requireAdmin(request);
  if (!admin) {
    return NextResponse.json({ success: false, message: "Forbidden" }, { status: 403 });
  }

  await ensureSearchLearningSchema();
  const { id } = await params;
  const reviewId = Number(id);
  if (!Number.isFinite(reviewId)) {
    return NextResponse.json({ success: false, message: "Invalid review id" }, { status: 400 });
  }

  const reviewResult = await query<{
    id: number;
    source: string;
    query_text: string;
    cpt_code: string | null;
    service_query: string | null;
    hospital_name: string | null;
    zip_code: string | null;
    insurer: string | null;
    suggested_alias: string | null;
    suggested_hospital_name: string | null;
    confidence: number;
    status: string;
  }>(
    `
      SELECT
        id,
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
        status
      FROM search_learning_reviews
      WHERE id = $1
      LIMIT 1
    `,
    [reviewId]
  );

  const review = reviewResult.rows[0];
  if (!review) {
    return NextResponse.json({ success: false, message: "Review not found" }, { status: 404 });
  }
  if (review.status !== "pending") {
    return NextResponse.json(
      { success: false, message: `Review already ${review.status}` },
      { status: 400 }
    );
  }

  const aliasText = normalizeText(review.suggested_alias || review.service_query || review.query_text);
  const hospitalAliasText = normalizeText(review.service_query || review.query_text);
  const cptCode = normalizeText(review.cpt_code);
  const suggestedHospitalName = normalizeText(review.suggested_hospital_name || review.hospital_name);
  const codeType = cptCode ? inferCodeType(cptCode) : "CPT";

  if (cptCode && aliasText) {
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
      [codeType, cptCode, aliasText, review.query_text, review.confidence]
    );
  }

  if (suggestedHospitalName && hospitalAliasText) {
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
      [suggestedHospitalName, hospitalAliasText, review.query_text, review.confidence]
    );
  }

  await query(
    `
      UPDATE search_learning_reviews
      SET status = 'approved', reviewed_at = now()
      WHERE id = $1
    `,
    [reviewId]
  );

  return NextResponse.json({
    success: true,
    message: "Review approved and learning alias saved.",
  });
}
