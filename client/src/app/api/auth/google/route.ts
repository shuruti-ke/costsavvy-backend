import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const url = new URL(request.url);
  url.pathname = "/auth";
  url.searchParams.set("oauth", "unavailable");
  return NextResponse.redirect(url);
}

