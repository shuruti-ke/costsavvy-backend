import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { signAuthToken } from "@/lib/auth-token";
import bcrypt from "bcryptjs";

export const runtime = "nodejs";

function toPublicUser(user: {
  id: string;
  name: string;
  email: string;
  role: string;
  phoneNumber?: string | null;
  companyName?: string | null;
  jobTitle?: string | null;
  organizationType?: string | null;
  zipCode?: string | null;
  useCase?: string | null;
  accountType?: string | null;
}) {
  return {
    id: String(user.id),
    name: user.name,
    email: user.email,
    role: user.role,
    avatar: null,
    phoneNumber: user.phoneNumber ?? null,
    companyName: user.companyName ?? null,
    jobTitle: user.jobTitle ?? null,
    organizationType: user.organizationType ?? null,
    zipCode: user.zipCode ?? null,
    useCase: user.useCase ?? null,
    accountType: user.accountType ?? null,
  };
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  const email = String(body.email || "").trim().toLowerCase();
  const password = String(body.password || "");

  if (!email || !password) {
    return NextResponse.json({ success: false, message: "Missing credentials" }, { status: 400 });
  }

  const result = await query<{
    id: string;
    name: string;
    email: string;
  role: string;
  password_hash: string;
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
      password_hash,
      is_active AS "isActive",
      phone_number AS "phoneNumber",
      company_name AS "companyName",
      job_title AS "jobTitle",
      organization_type AS "organizationType",
      zip_code AS "zipCode",
      use_case AS "useCase",
      account_type AS "accountType"
    FROM users WHERE email = $1 LIMIT 1`,
    [email]
  );
  const user = result.rows[0];
  if (!user || !user.password_hash) {
    return NextResponse.json({ success: false, message: "Invalid credentials" }, { status: 401 });
  }
  if (!user.isActive) {
    return NextResponse.json(
      { success: false, message: "Please confirm your email address before signing in." },
      { status: 403 }
    );
  }

  const isBcrypt = user.password_hash.startsWith("$2");
  const matches = isBcrypt ? await bcrypt.compare(password, user.password_hash) : user.password_hash === password;
  if (!matches) {
    return NextResponse.json({ success: false, message: "Invalid credentials" }, { status: 401 });
  }

  const publicUser = toPublicUser(user);
  const token = signAuthToken({
    sub: publicUser.id,
    name: publicUser.name,
    email: publicUser.email,
    role: publicUser.role,
    avatar: publicUser.avatar,
  });

  const response = NextResponse.json({ success: true, token, user: publicUser });
  response.cookies.set("token", token, {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 90 * 24 * 60 * 60,
  });
  return response;
}
