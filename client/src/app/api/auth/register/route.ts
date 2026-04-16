import { NextResponse } from "next/server";
import { query } from "@/lib/postgres";
import { signEmailVerificationToken } from "@/lib/auth-token";
import bcrypt from "bcryptjs";
import { sendNewUserNotificationEmail, sendSignupConfirmationEmail } from "@/lib/gmail";

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
  isActive?: boolean | null;
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
    isActive: user.isActive ?? null,
  };
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  const name = String(body.name || "").trim();
  const email = String(body.email || "").trim().toLowerCase();
  const password = String(body.password || "");
  const phoneNumber = String(body.phoneNumber || "").trim() || null;
  const companyName = String(body.companyName || "").trim() || null;
  const jobTitle = String(body.jobTitle || "").trim() || null;
  const organizationType = String(body.organizationType || "").trim() || null;
  const zipCode = String(body.zipCode || "").trim() || null;
  const useCase = String(body.useCase || "").trim() || null;
  const requestedAccountType = String(body.accountType || "").trim() || null;
  const accountType =
    requestedAccountType === "business" ||
    (organizationType && organizationType !== "consumer")
      ? "business"
      : "consumer";
  const dashboardPath = accountType === "business" ? "/dashboard/business" : "/";

  if (!name || !email || !password) {
    return NextResponse.json({ success: false, message: "Missing required fields" }, { status: 400 });
  }
  if (password.length < 6) {
    return NextResponse.json({ success: false, message: "Password must be at least 6 characters" }, { status: 400 });
  }

  const existing = await query<{ id: string }>(
    `SELECT id FROM users WHERE email = $1 LIMIT 1`,
    [email]
  );
  if (existing.rows.length > 0) {
    return NextResponse.json({ success: false, message: "User already exists" }, { status: 400 });
  }

  const hashedPassword = await bcrypt.hash(password, 10);
  const created = await query<{
    id: string;
    name: string;
    email: string;
    role: string;
    phoneNumber: string | null;
    companyName: string | null;
    jobTitle: string | null;
    organizationType: string | null;
    zipCode: string | null;
    useCase: string | null;
    accountType: string | null;
    isActive: boolean | null;
  }>(
    `INSERT INTO users (
      full_name,
      email,
      password_hash,
      role,
      is_active,
      phone_number,
      company_name,
      job_title,
      organization_type,
      zip_code,
      use_case,
      account_type
    ) VALUES ($1, $2, $3, 'user', FALSE, $4, $5, $6, $7, $8, $9, $10)
    RETURNING
      id,
      full_name AS name,
      email,
      role,
      phone_number AS "phoneNumber",
      company_name AS "companyName",
      job_title AS "jobTitle",
      organization_type AS "organizationType",
      zip_code AS "zipCode",
      use_case AS "useCase",
      account_type AS "accountType",
      is_active AS "isActive"`,
    [name, email, hashedPassword, phoneNumber, companyName, jobTitle, organizationType, zipCode, useCase, accountType]
  );
  const user = created.rows[0];
  if (!user) {
    return NextResponse.json({ success: false, message: "Failed to create user" }, { status: 500 });
  }

  const publicUser = toPublicUser(user);
  try {
    const verificationToken = signEmailVerificationToken({
      email: publicUser.email,
      name: publicUser.name,
      accountType: (publicUser.accountType as "business" | "consumer") || "consumer",
      organizationType: publicUser.organizationType,
    });
    const origin = new URL(request.url).origin;
    const confirmationUrl = new URL("/auth/confirm", origin);
    confirmationUrl.searchParams.set("token", verificationToken);

    await sendSignupConfirmationEmail({
      name: publicUser.name,
      email: publicUser.email,
      confirmationUrl: confirmationUrl.toString(),
      dashboardPath,
      accountType: (publicUser.accountType as "business" | "consumer") || "consumer",
      organizationType: publicUser.organizationType,
    });

    await sendNewUserNotificationEmail({
      name: publicUser.name,
      email: publicUser.email,
      role: publicUser.role,
      userId: publicUser.id,
    });
  } catch (error) {
    console.error("Failed to send new user notification email:", error);
  }

  return NextResponse.json(
    {
      success: true,
      message:
        "Account created. Please check your email to confirm your address before signing in.",
      confirmationRequired: true,
      dashboardPath,
      user: publicUser,
    },
    { status: 201 }
  );
}
