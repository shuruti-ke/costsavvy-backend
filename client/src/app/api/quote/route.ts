import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  const required = ["insuranceType", "firstName", "lastName", "zipCode", "email", "refSource", "phone"];
  const missing = required.filter((key) => !body?.[key]);

  if (missing.length > 0) {
    return NextResponse.json(
      { success: false, message: `Missing required fields: ${missing.join(", ")}` },
      { status: 400 }
    );
  }

  return NextResponse.json({
    success: true,
    message: "Quote request sent successfully",
  });
}

