import { NextResponse } from "next/server";

import { syncHospitalDirectoryFromSanity } from "@/lib/hospital-directory";

export const runtime = "nodejs";

function isAuthorized(request: Request) {
  const secret = process.env.SANITY_DIRECTORY_SYNC_SECRET;
  if (!secret) return false;

  const headerSecret =
    request.headers.get("x-sanity-sync-secret") ||
    request.headers.get("x-webhook-secret");

  if (headerSecret && headerSecret === secret) {
    return true;
  }

  const authHeader = request.headers.get("authorization");
  if (!authHeader) return false;
  const token = authHeader.startsWith("Bearer ")
    ? authHeader.slice("Bearer ".length).trim()
    : "";

  return token === secret;
}

export async function POST(request: Request) {
  if (!isAuthorized(request)) {
    return NextResponse.json({ success: false, message: "Unauthorized" }, { status: 401 });
  }

  try {
    const summary = await syncHospitalDirectoryFromSanity();
    return NextResponse.json({
      success: true,
      message: "Hospital directory synced from Sanity.",
      data: summary,
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : "Directory sync failed",
      },
      { status: 500 }
    );
  }
}

