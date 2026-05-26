import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { verifyEmailVerificationToken } from "@/lib/auth-token";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const token = url.searchParams.get("token") || "";

  if (!token) {
    return NextResponse.json({ success: false, message: "Missing confirmation token" }, { status: 400 });
  }

  const payload = verifyEmailVerificationToken(token);
  if (!payload) {
    return NextResponse.json({ success: false, message: "Invalid or expired confirmation token" }, { status: 400 });
  }

  const result = await query<{ id: string; isActive: boolean; accountType: string | null }>(
    `UPDATE users
     SET is_active = TRUE
     WHERE email = $1
     RETURNING id, is_active AS "isActive", account_type AS "accountType"`,
    [payload.email]
  );

  const user = result.rows[0];
  if (!user) {
    return NextResponse.json({ success: false, message: "Account not found" }, { status: 404 });
  }

  return NextResponse.json({
    success: true,
    message: "Email confirmed successfully. You can now sign in.",
    dashboardPath: user.accountType === "business" ? "/dashboard/business" : "/",
    accountType: user.accountType,
  });
}
