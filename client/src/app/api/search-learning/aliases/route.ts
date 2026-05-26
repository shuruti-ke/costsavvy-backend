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
  const limit = Math.max(parseInt(searchParams.get("limit") || "50", 10), 1);

  const rows = await query(
    `
      SELECT *
      FROM (
        SELECT
          id,
          'service' AS "aliasType",
          code_type AS "codeType",
          code,
          NULL::text AS "hospitalName",
          alias_text AS "aliasText",
          source_query AS "sourceQuery",
          confidence,
          learned_at AS "learnedAt"
        FROM service_search_aliases

        UNION ALL

        SELECT
          id,
          'hospital' AS "aliasType",
          NULL::text AS "codeType",
          NULL::text AS code,
          hospital_name AS "hospitalName",
          alias_text AS "aliasText",
          source_query AS "sourceQuery",
          confidence,
          learned_at AS "learnedAt"
        FROM hospital_search_aliases
      ) aliases
      ORDER BY learned_at DESC, id DESC
      LIMIT $1
    `,
    [limit]
  );

  return NextResponse.json({ success: true, count: rows.rows.length, data: rows.rows });
}
