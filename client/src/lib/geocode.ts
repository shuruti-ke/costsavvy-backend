const cache = new Map<string, { latitude: number; longitude: number } | null>();

export async function geocodeZip(zip: string) {
  const z = (zip || "").trim();
  if (!z) return null;
  if (cache.has(z)) return cache.get(z) || null;

  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?postalcode=${encodeURIComponent(z)}&country=US&format=json&limit=1`,
      {
        headers: {
          "User-Agent": "CostSavvyHealth/1.0 (migration@costsavvy.health)",
        },
      }
    );
    if (!response.ok) {
      cache.set(z, null);
      return null;
    }
    const data = (await response.json()) as Array<{ lat: string; lon: string }>;
    if (!Array.isArray(data) || data.length === 0) {
      cache.set(z, null);
      return null;
    }
    const coords = {
      latitude: Number.parseFloat(data[0].lat),
      longitude: Number.parseFloat(data[0].lon),
    };
    if (Number.isNaN(coords.latitude) || Number.isNaN(coords.longitude)) {
      cache.set(z, null);
      return null;
    }
    cache.set(z, coords);
    return coords;
  } catch {
    cache.set(z, null);
    return null;
  }
}

export async function geocodeAddress(parts: Array<string | null | undefined>) {
  const query = parts
    .map((part) => String(part || "").trim())
    .filter(Boolean)
    .join(", ");

  if (!query) return null;

  const key = `addr:${query.toLowerCase()}`;
  if (cache.has(key)) return cache.get(key) || null;

  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&country=US&format=json&limit=1`,
      {
        headers: {
          "User-Agent": "CostSavvyHealth/1.0 (migration@costsavvy.health)",
        },
      }
    );
    if (!response.ok) {
      cache.set(key, null);
      return null;
    }

    const data = (await response.json()) as Array<{ lat: string; lon: string }>;
    if (!Array.isArray(data) || data.length === 0) {
      cache.set(key, null);
      return null;
    }

    const coords = {
      latitude: Number.parseFloat(data[0].lat),
      longitude: Number.parseFloat(data[0].lon),
    };

    if (Number.isNaN(coords.latitude) || Number.isNaN(coords.longitude)) {
      cache.set(key, null);
      return null;
    }

    cache.set(key, coords);
    return coords;
  } catch {
    cache.set(key, null);
    return null;
  }
}
