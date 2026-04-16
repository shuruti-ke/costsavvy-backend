import { NextResponse } from "next/server";
import { sendBusinessIntakeEmail } from "@/lib/gmail";
import { getBearerToken, verifyAuthToken } from "@/lib/auth-token";
import { query } from "@/lib/postgres";

export const runtime = "nodejs";

async function requireBusinessUser(request: Request) {
  const payload = verifyAuthToken(getBearerToken(request));
  if (!payload) return null;

  const result = await query<{ id: string; role: string; accountType: string | null }>(
    `SELECT id, role, account_type AS "accountType"
     FROM users
     WHERE id = $1 AND is_active = TRUE
     LIMIT 1`,
    [payload.sub]
  );
  const user = result.rows[0];
  if (!user) return null;
  if (user.role !== "admin" && user.accountType !== "business") return null;
  return user;
}

export async function POST(request: Request) {
  const user = await requireBusinessUser(request);
  if (!user) {
    return NextResponse.json({ success: false, message: "Forbidden" }, { status: 403 });
  }

  const formData = await request.formData();
  const contactName = String(formData.get("contactName") || "").trim();
  const email = String(formData.get("email") || "").trim();
  const companyName = String(formData.get("companyName") || "").trim();
  const organizationType = String(formData.get("organizationType") || "").trim();
  const uploadType = String(formData.get("uploadType") || "").trim();
  const notes = String(formData.get("notes") || "").trim();
  const file = formData.get("file");
  const fileName = file instanceof File && file.size > 0 ? file.name : null;
  const attachment =
    file instanceof File && file.size > 0
      ? {
          filename: file.name,
          content: Buffer.from(await file.arrayBuffer()),
          contentType: file.type || "application/octet-stream",
        }
      : null;

  if (!contactName || !email || !companyName || !organizationType || !uploadType) {
    return NextResponse.json(
      { success: false, message: "Missing required business intake fields" },
      { status: 400 }
    );
  }

  try {
    await sendBusinessIntakeEmail({
      contactName,
      email,
      companyName,
      organizationType,
      uploadType,
      notes,
      fileName,
      attachment,
    });
    return NextResponse.json({
      success: true,
      message: "Data intake sent to the platform team.",
    });
  } catch (error) {
    console.error("Business intake email failed:", error);
    return NextResponse.json(
      { success: false, message: "Failed to send data intake request" },
      { status: 500 }
    );
  }
}
