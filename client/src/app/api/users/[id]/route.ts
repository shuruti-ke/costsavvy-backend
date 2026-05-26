import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
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

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const admin = await requireAdmin(request);
  if (!admin) {
    return NextResponse.json({ success: false, message: "Forbidden" }, { status: 403 });
  }

  const { id } = await params;
  const result = await query<{
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
  }>(
    `SELECT
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
      account_type AS "accountType"
    FROM users WHERE id = $1 LIMIT 1`,
    [id]
  );
  const user = result.rows[0];
  if (!user) {
    return NextResponse.json({ success: false, message: "User not found" }, { status: 404 });
  }

  return NextResponse.json({
    success: true,
    data: toPublicUser(user),
  });
}

export async function PUT(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const admin = await requireAdmin(request);
  if (!admin) {
    return NextResponse.json({ success: false, message: "Forbidden" }, { status: 403 });
  }

  const { id } = await params;
  const body = await request.json().catch(() => ({}));
  const fields: string[] = [];
  const values: unknown[] = [];

  if (body.name) {
    values.push(String(body.name).trim());
    fields.push(`full_name = $${values.length}`);
  }
  if (body.email) {
    values.push(String(body.email).trim().toLowerCase());
    fields.push(`email = $${values.length}`);
  }
  if (body.role) {
    values.push(String(body.role));
    fields.push(`role = $${values.length}`);
  }
  if (body.phoneNumber !== undefined) {
    values.push(String(body.phoneNumber || "").trim() || null);
    fields.push(`phone_number = $${values.length}`);
  }
  if (body.companyName !== undefined) {
    values.push(String(body.companyName || "").trim() || null);
    fields.push(`company_name = $${values.length}`);
  }
  if (body.jobTitle !== undefined) {
    values.push(String(body.jobTitle || "").trim() || null);
    fields.push(`job_title = $${values.length}`);
  }
  if (body.organizationType !== undefined) {
    values.push(String(body.organizationType || "").trim() || null);
    fields.push(`organization_type = $${values.length}`);
  }
  if (body.zipCode !== undefined) {
    values.push(String(body.zipCode || "").trim() || null);
    fields.push(`zip_code = $${values.length}`);
  }
  if (body.useCase !== undefined) {
    values.push(String(body.useCase || "").trim() || null);
    fields.push(`use_case = $${values.length}`);
  }
  if (body.accountType !== undefined) {
    values.push(String(body.accountType || "").trim() || null);
    fields.push(`account_type = $${values.length}`);
  }
  if (body.password) {
    const hashed = await bcrypt.hash(String(body.password), 10);
    values.push(hashed);
    fields.push(`password_hash = $${values.length}`);
  }
  if (body.isActive !== undefined) {
    values.push(Boolean(body.isActive));
    fields.push(`is_active = $${values.length}`);
  }

  values.push(id);
  const updateResult = fields.length
    ? await query<{ id: string; name: string; email: string; role: string }>(
        `UPDATE users SET ${fields.join(", ")} WHERE id = $${values.length} RETURNING
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
          account_type AS "accountType"`,
        values
      )
    : await query<{
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
      }>(
        `SELECT
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
          account_type AS "accountType"
        FROM users WHERE id = $1 LIMIT 1`,
        [id]
      );

  const user = updateResult.rows[0];
  if (!user) {
    return NextResponse.json({ success: false, message: "User not found" }, { status: 404 });
  }

  return NextResponse.json({
    success: true,
    data: toPublicUser(user),
  });
}

export async function DELETE(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const admin = await requireAdmin(request);
  if (!admin) {
    return NextResponse.json({ success: false, message: "Forbidden" }, { status: 403 });
  }

  const { id } = await params;
  const result = await query(`DELETE FROM users WHERE id = $1 RETURNING id`, [id]);
  if (result.rows.length === 0) {
    return NextResponse.json({ success: false, message: "User not found" }, { status: 404 });
  }

  return NextResponse.json({ success: true, data: {} });
}
