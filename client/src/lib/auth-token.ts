import crypto from "crypto";

type TokenPayload = {
  sub: string;
  name: string;
  email: string;
  role: string;
  avatar: string | null;
  iat: number;
  exp: number;
};

type EmailVerificationPayload = {
  kind: "email_verification";
  email: string;
  name: string;
  accountType: "business" | "consumer";
  organizationType: string | null;
  iat: number;
  exp: number;
};

const SECRET = process.env.AUTH_SECRET || process.env.JWT_SECRET || "cost-savvy-dev-secret";

function base64Url(input: Buffer | string) {
  return typeof input === "string"
    ? Buffer.from(input).toString("base64url")
    : input.toString("base64url");
}

function unbase64Url(input: string) {
  return Buffer.from(input, "base64url").toString("utf8");
}

function tokenExpiryDays() {
  const raw = process.env.JWT_EXPIRES_IN || "90d";
  const match = raw.match(/^(\d+)([dh]?)$/i);
  if (!match) return 90;
  const value = Number(match[1]);
  const unit = (match[2] || "d").toLowerCase();
  if (unit === "h") return Math.max(1, Math.ceil(value / 24));
  return Math.max(1, value);
}

export function signAuthToken(payload: Omit<TokenPayload, "iat" | "exp">) {
  const now = Math.floor(Date.now() / 1000);
  const exp = now + tokenExpiryDays() * 24 * 60 * 60;
  const body: TokenPayload = {
    ...payload,
    iat: now,
    exp,
  };

  const encodedBody = base64Url(JSON.stringify(body));
  const signature = crypto.createHmac("sha256", SECRET).update(encodedBody).digest("base64url");
  return `${encodedBody}.${signature}`;
}

export function verifyAuthToken(token: string) {
  if (!token || typeof token !== "string") return null;
  const [encodedBody, signature] = token.split(".");
  if (!encodedBody || !signature) return null;

  const expected = crypto.createHmac("sha256", SECRET).update(encodedBody).digest();
  const actual = Buffer.from(signature, "base64url");
  if (expected.length !== actual.length || !crypto.timingSafeEqual(expected, actual)) {
    return null;
  }

  try {
    const payload = JSON.parse(unbase64Url(encodedBody)) as TokenPayload;
    if (!payload?.sub || !payload?.email || !payload?.role) return null;
    if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) return null;
    return payload;
  } catch {
    return null;
  }
}

function verificationExpiryDays() {
  const raw = process.env.EMAIL_VERIFICATION_EXPIRES_IN || "2d";
  const match = raw.match(/^(\d+)([dh]?)$/i);
  if (!match) return 2;
  const value = Number(match[1]);
  const unit = (match[2] || "d").toLowerCase();
  if (unit === "h") return Math.max(1, Math.ceil(value / 24));
  return Math.max(1, value);
}

export function signEmailVerificationToken(payload: Omit<EmailVerificationPayload, "kind" | "iat" | "exp">) {
  const now = Math.floor(Date.now() / 1000);
  const exp = now + verificationExpiryDays() * 24 * 60 * 60;
  const body: EmailVerificationPayload = {
    kind: "email_verification",
    ...payload,
    iat: now,
    exp,
  };

  const encodedBody = base64Url(JSON.stringify(body));
  const signature = crypto.createHmac("sha256", SECRET).update(encodedBody).digest("base64url");
  return `${encodedBody}.${signature}`;
}

export function verifyEmailVerificationToken(token: string) {
  if (!token || typeof token !== "string") return null;
  const [encodedBody, signature] = token.split(".");
  if (!encodedBody || !signature) return null;

  const expected = crypto.createHmac("sha256", SECRET).update(encodedBody).digest();
  const actual = Buffer.from(signature, "base64url");
  if (expected.length !== actual.length || !crypto.timingSafeEqual(expected, actual)) {
    return null;
  }

  try {
    const payload = JSON.parse(unbase64Url(encodedBody)) as EmailVerificationPayload;
    if (payload.kind !== "email_verification") return null;
    if (!payload?.email || !payload?.name) return null;
    if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) return null;
    return payload;
  } catch {
    return null;
  }
}

export function getBearerToken(request: Request) {
  const auth = request.headers.get("authorization");
  if (auth?.startsWith("Bearer ")) return auth.slice(7).trim();

  const cookie = request.headers.get("cookie") || "";
  const match = cookie.match(/(?:^|;\s*)token=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}
