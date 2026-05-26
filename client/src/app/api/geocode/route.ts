// app/api/geocode/route.ts
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const address = searchParams.get("address");
  const zip = searchParams.get("zip");
  const query = address || (zip ? `${zip} USA` : null);

  if (!query) {
    return NextResponse.json({ error: "Missing address or zip param" }, { status: 400 });
  }

  const apiKey = process.env.GOOGLE_MAPS_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "Geocoding not configured" }, { status: 500 });
  }

  try {
    const res = await fetch(
      `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(query)}&key=${apiKey}`
    );
    const data = await res.json();

    if (data.status !== "OK" || !data.results?.length) {
      return NextResponse.json([]);
    }

    const { lat, lng } = data.results[0].geometry.location;
    // Return in same shape as before so map.tsx doesn't need changes
    return NextResponse.json([{ lat: String(lat), lon: String(lng) }]);
  } catch (err) {
    console.error("Geocode error:", err);
    return NextResponse.json({ error: "Failed to fetch geocode data" }, { status: 500 });
  }
}
