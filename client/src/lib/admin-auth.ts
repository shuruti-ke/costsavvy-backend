import { query } from "@/lib/postgres";
import { getBearerToken, verifyAuthToken } from "@/lib/auth-token";

export async function requireAdmin(request: Request) {
  const payload = verifyAuthToken(getBearerToken(request));
  if (!payload) return null;
  const result = await query<{ id: string; role: string }>(
    `SELECT id, role FROM users WHERE id = $1 LIMIT 1`,
    [payload.sub]
  );
  const user = result.rows[0];
  if (!user || user.role !== "admin") return null;
  return user;
}
