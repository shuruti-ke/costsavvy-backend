import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { getBearerToken, verifyAuthToken } from "@/lib/auth-token";

export const runtime = "nodejs";

async function requireAdmin(request: Request) {
  const payload = verifyAuthToken(getBearerToken(request));
  if (!payload) return null;
  const result = await query<{ id: string; role: string }>(
    `SELECT id, role FROM users WHERE id = $1 LIMIT 1`,
    [payload.sub]
  );
  const user = result.rows[0];
  if (!user || user.role !== "admin") return null;
  return user;
}

export async function GET(request: Request) {
  const admin = await requireAdmin(request);
  if (!admin) {
    return NextResponse.json({ success: false, message: "Forbidden" }, { status: 403 });
  }

  const users = await query<{
    id: string;
    name: string;
    email: string;
    role: string;
    avatar: string | null;
    phoneNumber: string | null;
    companyName: string | null;
    jobTitle: string | null;
    organizationType: string | null;
    zipCode: string | null;
    useCase: string | null;
    accountType: string | null;
  }>(
    `SELECT
      id,
      full_name AS name,
      email,
      role,
      NULL::text AS avatar,
      phone_number AS "phoneNumber",
      company_name AS "companyName",
      job_title AS "jobTitle",
      organization_type AS "organizationType",
      zip_code AS "zipCode",
      use_case AS "useCase",
      account_type AS "accountType"
    FROM users ORDER BY full_name ASC`
  );
  return NextResponse.json({
    success: true,
    count: users.rows.length,
    data: users.rows,
  });
}
