const cache = new Map<string, { latitude: number; longitude: number } | null>();

const GOOGLE_API_KEY = process.env.GOOGLE_MAPS_API_KEY;

async function googleGeocode(query: string): Promise<{ latitude: number; longitude: number } | null> {
  if (!GOOGLE_API_KEY) return null;
  try {
    const res = await fetch(
      `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(query)}&key=${GOOGLE_API_KEY}`
    );
    const data = await res.json();
    if (data.status !== "OK" || !data.results?.length) return null;
    const { lat, lng } = data.results[0].geometry.location;
    return { latitude: lat, longitude: lng };
  } catch {
    return null;
  }
}

export async function geocodeZip(zip: string) {
  const z = (zip || "").trim();
  if (!z) return null;
  if (cache.has(z)) return cache.get(z) || null;
  const coords = await googleGeocode(`${z} USA`);
  cache.set(z, coords);
  return coords;
}

export async function geocodeAddress(parts: Array<string | null | undefined>) {
  const query = parts
    .map((part) => String(part || "").trim())
    .filter(Boolean)
    .join(", ");

  if (!query) return null;

  const key = `addr:${query.toLowerCase()}`;
  if (cache.has(key)) return cache.get(key) || null;

  const coords = await googleGeocode(query);
  cache.set(key, coords);
  return coords;
}
