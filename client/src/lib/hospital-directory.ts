import { query } from "@/lib/postgres";
import client from "@/lib/sanity";
import { geocodeAddress } from "@/lib/geocode";

export type HospitalDirectoryType = "provider" | "health-system";

export interface HospitalDirectoryRecord {
  id: number;
  name: string;
  systemName: string | null;
  directoryType: HospitalDirectoryType;
  facilityType: string | null;
  address: string | null;
  city: string | null;
  state: string | null;
  zipcode: string | null;
  description: string | null;
  website: string | null;
  phone: string | null;
  latitude: number | null;
  longitude: number | null;
  googlePlaceId: string | null;
  googleMapsUrl: string | null;
  npi: string | null;
  cmsProviderId: string | null;
  ownership: string | null;
  bedCount: number | null;
  isVerified: boolean;
  active: boolean;
  nearbyHospitals: string[];
  clinicalServices: string[];
}

export interface HospitalDirectoryListItem {
  id: number;
  name: string;
  systemName: string | null;
  directoryType: HospitalDirectoryType;
  location: string;
  description: string;
  state: string | null;
  website: string | null;
  isVerified: boolean;
  tab: "dynProviders" | "healthSystems";
}

export interface HospitalDirectoryServiceItem {
  id: number;
  hospitalId: number;
  serviceName: string;
  serviceCategory: string | null;
  cptCode: string | null;
  isPrimary: boolean;
}

export interface HospitalDirectoryInput {
  id?: number | null;
  name: string;
  systemName?: string | null;
  directoryType?: HospitalDirectoryType;
  facilityType?: string | null;
  address?: string | null;
  city?: string | null;
  state?: string | null;
  zipcode?: string | null;
  description?: string | null;
  website?: string | null;
  phone?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  googlePlaceId?: string | null;
  googleMapsUrl?: string | null;
  npi?: string | null;
  cmsProviderId?: string | null;
  ownership?: string | null;
  bedCount?: number | null;
  isVerified?: boolean;
  active?: boolean;
  nearbyHospitals?: string[];
  clinicalServices?: string[];
}

interface SanityProviderDoc {
  _id: string;
  name: string;
  address?: {
    street?: string;
    city?: string;
    state?: string;
    zip?: string;
  };
  phone?: string | null;
  medicareProviderId?: string | null;
  npi?: string | null;
  website?: string | null;
  providerType?: string | null;
  ownership?: string | null;
  beds?: number | null;
  nearbyProviders?: string[] | null;
  clinicalServices?: string[] | null;
}

interface SanityHealthSystemDoc {
  _id: string;
  name: string;
  isVerified?: boolean | null;
  claimUrl?: string | null;
  locations?: Array<{
    facilityName?: string | null;
    street?: string | null;
    city?: string | null;
    state?: string | null;
    zip?: string | null;
  }> | null;
  services?: string[] | null;
}

export interface HospitalDirectorySyncResult {
  providersSynced: number;
  healthSystemsSynced: number;
  healthSystemFacilitiesSynced: number;
  totalRecordsTouched: number;
}

let ensureSchemaPromise: Promise<void> | null = null;

export async function ensureHospitalDirectorySchema() {
  if (!ensureSchemaPromise) {
    ensureSchemaPromise = (async () => {
      await query(`
        ALTER TABLE hospitals
        ADD COLUMN IF NOT EXISTS system_name TEXT,
        ADD COLUMN IF NOT EXISTS directory_type TEXT NOT NULL DEFAULT 'provider',
        ADD COLUMN IF NOT EXISTS facility_type TEXT,
        ADD COLUMN IF NOT EXISTS city TEXT,
        ADD COLUMN IF NOT EXISTS website TEXT,
        ADD COLUMN IF NOT EXISTS phone TEXT,
        ADD COLUMN IF NOT EXISTS google_place_id TEXT,
        ADD COLUMN IF NOT EXISTS google_maps_url TEXT,
        ADD COLUMN IF NOT EXISTS npi TEXT,
        ADD COLUMN IF NOT EXISTS cms_provider_id TEXT,
        ADD COLUMN IF NOT EXISTS ownership TEXT,
        ADD COLUMN IF NOT EXISTS bed_count INTEGER,
        ADD COLUMN IF NOT EXISTS is_verified BOOLEAN NOT NULL DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE,
        ADD COLUMN IF NOT EXISTS nearby_hospitals TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
        ADD COLUMN IF NOT EXISTS clinical_services TEXT[] NOT NULL DEFAULT ARRAY[]::text[]
      `);

      await query(`
        CREATE INDEX IF NOT EXISTS hospitals_directory_type_idx ON hospitals(directory_type);
      `);
      await query(`
        CREATE INDEX IF NOT EXISTS hospitals_active_idx ON hospitals(active);
      `);

      await query(`
        CREATE TABLE IF NOT EXISTS hospital_services (
          id BIGSERIAL PRIMARY KEY,
          hospital_id BIGINT NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
          service_name TEXT NOT NULL,
          service_category TEXT,
          cpt_code TEXT,
          is_primary BOOLEAN NOT NULL DEFAULT FALSE,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
      `);
      await query(`
        CREATE UNIQUE INDEX IF NOT EXISTS hospital_services_unique_idx
        ON hospital_services(hospital_id, service_name, COALESCE(service_category, ''), COALESCE(cpt_code, ''));
      `);
    })().catch((error) => {
      ensureSchemaPromise = null;
      throw error;
    });
  }

  return ensureSchemaPromise;
}

function uniqueTextArray(values: string[] | null | undefined) {
  return Array.from(
    new Set(
      (values || [])
        .map((value) => String(value || "").trim())
        .filter(Boolean)
    )
  );
}

function normalizeText(value: string | null | undefined) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function splitCsvText(value: string | null | undefined) {
  return uniqueTextArray(
    String(value || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
  );
}

function toListItem(row: HospitalDirectoryRecord): HospitalDirectoryListItem {
  const location = [row.city, row.state].filter(Boolean).join(", ");
  return {
    id: row.id,
    name: row.name,
    systemName: row.systemName,
    directoryType: row.directoryType,
    location,
    description: row.description || row.facilityType || "",
    state: row.state,
    website: row.website,
    isVerified: row.isVerified,
    tab: row.directoryType === "health-system" ? "healthSystems" : "dynProviders",
  };
}

async function findExistingHospitalDirectoryRecord(input: {
  name: string;
  systemName?: string | null;
  address?: string | null;
  city?: string | null;
  state?: string | null;
  zipcode?: string | null;
  directoryType: HospitalDirectoryType;
}) {
  const exact = await query<{ id: number }>(
    `
      SELECT id
      FROM hospitals
      WHERE directory_type = $1
        AND lower(name) = lower($2)
        AND (
          COALESCE(lower(system_name), '') = COALESCE(lower($3), '')
          OR COALESCE(lower(address), '') = COALESCE(lower($4), '')
          OR (
            COALESCE(lower(city), '') = COALESCE(lower($5), '')
            AND COALESCE(lower(state), '') = COALESCE(lower($6), '')
            AND COALESCE(zipcode, '') = COALESCE($7, '')
          )
        )
      ORDER BY id ASC
      LIMIT 1
    `,
    [
      input.directoryType,
      input.name,
      input.systemName || null,
      input.address || null,
      input.city || null,
      input.state || null,
      input.zipcode || null,
    ]
  );

  if (exact.rows[0]?.id) {
    return exact.rows[0].id;
  }

  const fuzzy = await query<{
    id: number;
    name: string;
    systemName: string | null;
    city: string | null;
    state: string | null;
    zipcode: string | null;
  }>(
    `
      SELECT
        id,
        name,
        system_name AS "systemName",
        city,
        state,
        zipcode
      FROM hospitals
      WHERE directory_type = $1
        AND (
          COALESCE(lower(city), '') = COALESCE(lower($2), '')
          OR COALESCE(lower(state), '') = COALESCE(lower($3), '')
          OR COALESCE(zipcode, '') = COALESCE($4, '')
        )
      ORDER BY id ASC
      LIMIT 50
    `,
    [input.directoryType, input.city || null, input.state || null, input.zipcode || null]
  );

  const targetName = normalizeText(input.name);
  const targetSystem = normalizeText(input.systemName || "");

  for (const row of fuzzy.rows) {
    const nameScore =
      normalizeText(row.name) === targetName ||
      normalizeText(row.name).includes(targetName) ||
      targetName.includes(normalizeText(row.name));
    const systemScore =
      !targetSystem ||
      normalizeText(row.systemName || "") === targetSystem ||
      normalizeText(row.systemName || "").includes(targetSystem) ||
      targetSystem.includes(normalizeText(row.systemName || ""));

    if (nameScore && systemScore) {
      return row.id;
    }
  }

  return null;
}

export async function listHospitalDirectory(args: {
  directoryType?: HospitalDirectoryType | "all";
  search?: string;
  state?: string;
  limit?: number;
}) {
  await ensureHospitalDirectorySchema();

  const params: unknown[] = [];
  const where: string[] = ["COALESCE(active, TRUE) = TRUE"];

  if (args.directoryType && args.directoryType !== "all") {
    params.push(args.directoryType);
    where.push(`directory_type = $${params.length}`);
  }

  if (args.state) {
    params.push(args.state);
    where.push(`state = $${params.length}`);
  }

  if (args.search) {
    params.push(`%${String(args.search).trim()}%`);
    const idx = params.length;
    where.push(`(
      name ILIKE $${idx}
      OR COALESCE(system_name, '') ILIKE $${idx}
      OR COALESCE(city, '') ILIKE $${idx}
      OR EXISTS (
        SELECT 1 FROM unnest(COALESCE(clinical_services, ARRAY[]::text[])) AS service
        WHERE service ILIKE $${idx}
      )
    )`);
  }

  params.push(Math.max(args.limit || 100, 1));

  const result = await query<HospitalDirectoryRecord>(
    `
      SELECT
        id,
        name,
        system_name AS "systemName",
        directory_type AS "directoryType",
        facility_type AS "facilityType",
        address,
        city,
        state,
        zipcode,
        description,
        website,
        phone,
        latitude,
        longitude,
        google_place_id AS "googlePlaceId",
        google_maps_url AS "googleMapsUrl",
        npi,
        cms_provider_id AS "cmsProviderId",
        ownership,
        bed_count AS "bedCount",
        is_verified AS "isVerified",
        active,
        nearby_hospitals AS "nearbyHospitals",
        clinical_services AS "clinicalServices"
      FROM hospitals
      WHERE ${where.join(" AND ")}
      ORDER BY is_verified DESC, name ASC
      LIMIT $${params.length}
    `,
    params
  );

  return result.rows.map(toListItem);
}

export async function getHospitalDirectoryById(id: number) {
  await ensureHospitalDirectorySchema();

  const result = await query<HospitalDirectoryRecord>(
    `
      SELECT
        id,
        name,
        system_name AS "systemName",
        directory_type AS "directoryType",
        facility_type AS "facilityType",
        address,
        city,
        state,
        zipcode,
        description,
        website,
        phone,
        latitude,
        longitude,
        google_place_id AS "googlePlaceId",
        google_maps_url AS "googleMapsUrl",
        npi,
        cms_provider_id AS "cmsProviderId",
        ownership,
        bed_count AS "bedCount",
        is_verified AS "isVerified",
        active,
        nearby_hospitals AS "nearbyHospitals",
        clinical_services AS "clinicalServices"
      FROM hospitals
      WHERE id = $1
      LIMIT 1
    `,
    [id]
  );

  const record = result.rows[0] || null;
  if (!record) return null;

  const servicesResult = await query<HospitalDirectoryServiceItem>(
    `
      SELECT
        id,
        hospital_id AS "hospitalId",
        service_name AS "serviceName",
        service_category AS "serviceCategory",
        cpt_code AS "cptCode",
        is_primary AS "isPrimary"
      FROM hospital_services
      WHERE hospital_id = $1
      ORDER BY is_primary DESC, service_name ASC
    `,
    [id]
  );

  return {
    ...record,
    clinicalServices:
      servicesResult.rows.length > 0
        ? uniqueTextArray(servicesResult.rows.map((service) => service.serviceName))
        : uniqueTextArray(record.clinicalServices),
  };
}

export async function upsertHospitalDirectory(input: HospitalDirectoryInput) {
  await ensureHospitalDirectorySchema();

  const payload = {
    name: String(input.name || "").trim(),
    systemName: String(input.systemName || "").trim() || null,
    directoryType: input.directoryType || "provider",
    facilityType: String(input.facilityType || "").trim() || null,
    address: String(input.address || "").trim() || null,
    city: String(input.city || "").trim() || null,
    state: String(input.state || "").trim() || null,
    zipcode: String(input.zipcode || "").trim() || null,
    description: String(input.description || "").trim() || null,
    website: String(input.website || "").trim() || null,
    phone: String(input.phone || "").trim() || null,
    latitude: input.latitude ?? null,
    longitude: input.longitude ?? null,
    googlePlaceId: String(input.googlePlaceId || "").trim() || null,
    googleMapsUrl: String(input.googleMapsUrl || "").trim() || null,
    npi: String(input.npi || "").trim() || null,
    cmsProviderId: String(input.cmsProviderId || "").trim() || null,
    ownership: String(input.ownership || "").trim() || null,
    bedCount: input.bedCount ?? null,
    isVerified: Boolean(input.isVerified),
    active: input.active !== false,
    nearbyHospitals: uniqueTextArray(input.nearbyHospitals),
    clinicalServices: uniqueTextArray(input.clinicalServices),
  };

  if (!payload.name) {
    throw new Error("Hospital name is required.");
  }

  const params = [
    payload.name,
    payload.systemName,
    payload.directoryType,
    payload.facilityType,
    payload.address,
    payload.city,
    payload.state,
    payload.zipcode,
    payload.description,
    payload.website,
    payload.phone,
    payload.latitude,
    payload.longitude,
    payload.googlePlaceId,
    payload.googleMapsUrl,
    payload.npi,
    payload.cmsProviderId,
    payload.ownership,
    payload.bedCount,
    payload.isVerified,
    payload.active,
    payload.nearbyHospitals,
    payload.clinicalServices,
  ];

  const resolvedId =
    input.id ||
    (await findExistingHospitalDirectoryRecord({
      name: payload.name,
      systemName: payload.systemName,
      address: payload.address,
      city: payload.city,
      state: payload.state,
      zipcode: payload.zipcode,
      directoryType: payload.directoryType,
    }));

  const result = resolvedId
    ? await query<{ id: number }>(
        `
          UPDATE hospitals
          SET
            name = $1,
            system_name = $2,
            directory_type = $3,
            facility_type = $4,
            address = $5,
            city = $6,
            state = $7,
            zipcode = $8,
            description = $9,
            website = $10,
            phone = $11,
            latitude = $12,
            longitude = $13,
            google_place_id = $14,
            google_maps_url = $15,
            npi = $16,
            cms_provider_id = $17,
            ownership = $18,
            bed_count = $19,
            is_verified = $20,
            active = $21,
            nearby_hospitals = $22,
            clinical_services = $23
          WHERE id = $24
          RETURNING id
        `,
        [...params, resolvedId]
      )
    : await query<{ id: number }>(
        `
          INSERT INTO hospitals (
            name,
            system_name,
            directory_type,
            facility_type,
            address,
            city,
            state,
            zipcode,
            description,
            website,
            phone,
            latitude,
            longitude,
            google_place_id,
            google_maps_url,
            npi,
            cms_provider_id,
            ownership,
            bed_count,
            is_verified,
            active,
            nearby_hospitals,
            clinical_services
          )
          VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23
          )
          RETURNING id
        `,
        params
      );

  const id = result.rows[0]?.id;
  if (!id) {
    throw new Error("Failed to save hospital.");
  }

  await syncHospitalServices(id, payload.clinicalServices);
  return getHospitalDirectoryById(id);
}

async function syncHospitalServices(hospitalId: number, clinicalServices: string[]) {
  await query(`DELETE FROM hospital_services WHERE hospital_id = $1`, [hospitalId]);
  for (const serviceName of clinicalServices) {
    await query(
      `
        INSERT INTO hospital_services (
          hospital_id,
          service_name
        )
        VALUES ($1, $2)
        ON CONFLICT DO NOTHING
      `,
      [hospitalId, serviceName]
    );
  }
}

export async function deleteHospitalDirectory(id: number) {
  await ensureHospitalDirectorySchema();
  await query(`DELETE FROM hospitals WHERE id = $1`, [id]);
}

export async function listHospitalStates(directoryType?: HospitalDirectoryType | "all") {
  await ensureHospitalDirectorySchema();
  const params: unknown[] = [];
  let where = `WHERE COALESCE(active, TRUE) = TRUE AND COALESCE(state, '') <> ''`;
  if (directoryType && directoryType !== "all") {
    params.push(directoryType);
    where += ` AND directory_type = $1`;
  }

  const result = await query<{ state: string }>(
    `
      SELECT DISTINCT state
      FROM hospitals
      ${where}
      ORDER BY state ASC
    `,
    params
  );

  return result.rows.map((row) => row.state).filter(Boolean);
}

export async function syncHospitalDirectoryFromSanity(): Promise<HospitalDirectorySyncResult> {
  await ensureHospitalDirectorySchema();

  const [providers, healthSystems] = await Promise.all([
    client.fetch<SanityProviderDoc[]>(
      `*[_type == "provider"]{
        _id,
        name,
        address,
        phone,
        medicareProviderId,
        npi,
        website,
        providerType,
        ownership,
        beds,
        nearbyProviders,
        clinicalServices
      }`
    ),
    client.fetch<SanityHealthSystemDoc[]>(
      `*[_type == "healthSystem"]{
        _id,
        name,
        isVerified,
        claimUrl,
        locations,
        services
      }`
    ),
  ]);

  let providersSynced = 0;
  let healthSystemsSynced = 0;
  let healthSystemFacilitiesSynced = 0;

  for (const provider of providers || []) {
    if (!provider?.name) continue;
    const providerCoords = await geocodeAddress([
      provider.address?.street,
      provider.address?.city,
      provider.address?.state,
      provider.address?.zip,
      "USA",
    ]);
    const providerMapsUrl =
      providerCoords
        ? `https://www.google.com/maps?q=${providerCoords.latitude},${providerCoords.longitude}`
        : null;

    await upsertHospitalDirectory({
      name: provider.name,
      directoryType: "provider",
      facilityType: provider.providerType || "Provider",
      address: provider.address?.street || null,
      city: provider.address?.city || null,
      state: provider.address?.state || null,
      zipcode: provider.address?.zip || null,
      website: provider.website || null,
      phone: provider.phone || null,
      latitude: providerCoords?.latitude ?? null,
      longitude: providerCoords?.longitude ?? null,
      googleMapsUrl: providerMapsUrl,
      npi: provider.npi || null,
      cmsProviderId: provider.medicareProviderId || null,
      ownership: provider.ownership || null,
      bedCount: provider.beds ?? null,
      nearbyHospitals: uniqueTextArray(provider.nearbyProviders || []),
      clinicalServices: uniqueTextArray(provider.clinicalServices || []),
      description: provider.providerType || "Healthcare provider",
      isVerified: Boolean(provider.website || provider.npi || provider.medicareProviderId),
      active: true,
    });
    providersSynced += 1;
  }

  for (const system of healthSystems || []) {
    if (!system?.name) continue;
    const locations = (system.locations || []).filter((location) => location?.facilityName);
    const firstLocation = locations[0];
    const systemCoords = await geocodeAddress([
      firstLocation?.street,
      firstLocation?.city,
      firstLocation?.state,
      firstLocation?.zip,
      "USA",
    ]);
    const systemMapsUrl =
      system.claimUrl ||
      (systemCoords
        ? `https://www.google.com/maps?q=${systemCoords.latitude},${systemCoords.longitude}`
        : null);

    await upsertHospitalDirectory({
      name: system.name,
      systemName: system.name,
      directoryType: "health-system",
      facilityType: "Health system",
      address: firstLocation?.street || null,
      city: firstLocation?.city || null,
      state: firstLocation?.state || null,
      zipcode: firstLocation?.zip || null,
      website: system.claimUrl || null,
      latitude: systemCoords?.latitude ?? null,
      longitude: systemCoords?.longitude ?? null,
      googleMapsUrl: systemMapsUrl,
      description:
        locations.length > 0
          ? `${locations.length} affiliated location${locations.length === 1 ? "" : "s"}`
          : "Health system",
      nearbyHospitals: uniqueTextArray(
        locations.map((location) => location.facilityName || "").filter(Boolean)
      ),
      clinicalServices: uniqueTextArray(system.services || []),
      isVerified: Boolean(system.isVerified),
      active: true,
    });
    healthSystemsSynced += 1;

    for (const location of locations) {
      const locationCoords = await geocodeAddress([
        location.street,
        location.city,
        location.state,
        location.zip,
        "USA",
      ]);
      const locationMapsUrl =
        system.claimUrl ||
        (locationCoords
          ? `https://www.google.com/maps?q=${locationCoords.latitude},${locationCoords.longitude}`
          : null);

      await upsertHospitalDirectory({
        name: location.facilityName || system.name,
        systemName: system.name,
        directoryType: "provider",
        facilityType: "Hospital",
        address: location.street || null,
        city: location.city || null,
        state: location.state || null,
        zipcode: location.zip || null,
        website: system.claimUrl || null,
        latitude: locationCoords?.latitude ?? null,
        longitude: locationCoords?.longitude ?? null,
        googleMapsUrl: locationMapsUrl,
        description: `${system.name} affiliated hospital`,
        nearbyHospitals: uniqueTextArray(
          locations
            .map((item) => item.facilityName || "")
            .filter((facilityName) => facilityName && facilityName !== location.facilityName)
        ),
        clinicalServices: uniqueTextArray(system.services || []),
        isVerified: Boolean(system.isVerified),
        active: true,
      });
      healthSystemFacilitiesSynced += 1;
    }
  }

  return {
    providersSynced,
    healthSystemsSynced,
    healthSystemFacilitiesSynced,
    totalRecordsTouched:
      providersSynced + healthSystemsSynced + healthSystemFacilitiesSynced,
  };
}
