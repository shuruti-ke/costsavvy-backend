import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { getBearerToken, verifyAuthToken } from "@/lib/auth-token";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const token = getBearerToken(request);
  const payload = verifyAuthToken(token);

  if (!payload) {
    return NextResponse.json({ success: false, message: "Not authenticated" }, { status: 401 });
  }

  const result = await query<{
    id: string;
    name: string;
    email: string;
    role: string;
    isActive: boolean;
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
      is_active AS "isActive",
      phone_number AS "phoneNumber",
      company_name AS "companyName",
      job_title AS "jobTitle",
      organization_type AS "organizationType",
      zip_code AS "zipCode",
      use_case AS "useCase",
      account_type AS "accountType"
    FROM users WHERE id = $1 LIMIT 1`,
    [payload.sub]
  );
  const user = result.rows[0];
  if (!user || !user.isActive) {
    return NextResponse.json({ success: false, message: "Not authenticated" }, { status: 401 });
  }

  return NextResponse.json({
    success: true,
    data: {
      id: String(user.id),
      name: user.name,
      email: user.email,
      role: user.role,
      avatar: null,
      phoneNumber: user.phoneNumber,
      companyName: user.companyName,
      jobTitle: user.jobTitle,
      organizationType: user.organizationType,
      zipCode: user.zipCode,
      useCase: user.useCase,
      accountType: user.accountType,
    },
  });
}
