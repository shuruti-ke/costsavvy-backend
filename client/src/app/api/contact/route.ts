import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  const required = ["firstName", "lastName", "email", "phone", "howHeard", "problem"];
  const missing = required.filter((key) => !body?.[key]);

  if (missing.length > 0) {
    return NextResponse.json(
      { success: false, message: `Missing required fields: ${missing.join(", ")}` },
      { status: 400 }
    );
  }

  return NextResponse.json({
    success: true,
    message: "Contact message sent successfully",
  });
}

