import { Pool } from "pg";
import type { QueryResultRow } from "pg";

declare global {
  // eslint-disable-next-line no-var
  var pgPool: Pool | undefined;
}

function createPool() {
  const connectionString = process.env.DATABASE_URL || process.env.POSTGRES_URL;
  if (!connectionString) {
    throw new Error("Missing DATABASE_URL environment variable");
  }

  let ssl: { rejectUnauthorized: boolean } | undefined;
  try {
    const url = new URL(connectionString);
    if (url.hostname.includes("render.com")) {
      ssl = { rejectUnauthorized: false };
    }
  } catch {
    ssl = undefined;
  }

  return new Pool({
    connectionString,
    ssl,
    max: 10,
    idleTimeoutMillis: 30_000,
    connectionTimeoutMillis: 5_000,
  });
}

function getPool() {
  if (!global.pgPool) {
    global.pgPool = createPool();
  }
  return global.pgPool;
}

export async function query<T extends QueryResultRow = QueryResultRow>(
  text: string,
  params: unknown[] = []
) {
  return getPool().query<T>(text, params);
}
