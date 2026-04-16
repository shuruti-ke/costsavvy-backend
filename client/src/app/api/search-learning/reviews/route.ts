import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { requireAdmin } from "@/lib/admin-auth";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const admin = await requireAdmin(request);
  if (!admin) {
    return NextResponse.json({ success: false, message: "Forbidden" }, { status: 403 });
  }

  const { searchParams } = new URL(request.url);
  const status = searchParams.get("status")?.trim() || "pending";
  const limit = Math.max(parseInt(searchParams.get("limit") || "50", 10), 1);

  const rows = await query(
    `
      SELECT
        id,
        source,
        query_text AS "queryText",
        cpt_code AS "cptCode",
        service_query AS "serviceQuery",
        hospital_name AS "hospitalName",
        zip_code AS "zipCode",
        insurer,
        suggested_alias AS "suggestedAlias",
        suggested_hospital_name AS "suggestedHospitalName",
        confidence,
        rationale,
        result_count AS "resultCount",
        status,
        created_at AS "createdAt",
        reviewed_at AS "reviewedAt"
      FROM search_learning_reviews
      WHERE status = $1
      ORDER BY created_at DESC
      LIMIT $2
    `,
    [status, limit]
  );

  return NextResponse.json({ success: true, count: rows.rows.length, data: rows.rows });
}
