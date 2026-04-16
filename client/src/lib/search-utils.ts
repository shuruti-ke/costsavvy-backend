export function escapeRegex(str = "") {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function parseZipRange(zipCode: string) {
  const zip = parseInt(zipCode, 10);
  if (Number.isNaN(zip)) return null;
  const firstDigit = Math.floor(zip / 1000);
  const lower = firstDigit * 1000;
  const upper = lower + 999;
  return { lower, upper };
}

const SYMPTOM_KEYWORDS = [
  "hurts",
  "hurt",
  "pain",
  "ache",
  "aching",
  "sore",
  "swollen",
  "swelling",
  "bleeding",
  "dizzy",
  "dizziness",
  "nausea",
  "vomiting",
  "fever",
  "cough",
  "headache",
  "chest pain",
  "shortness of breath",
  "can't breathe",
  "broken",
  "sprain",
  "twisted",
  "injured",
  "injury",
  "sick",
  "unwell",
  "feel bad",
  "emergency",
  "urgent",
  "help me find",
  "where can i go",
  "need a doctor",
  "need to see",
  "should i go to",
];

const EDUCATION_KEYWORDS = [
  "what is",
  "what's",
  "explain",
  "tell me about",
  "how does",
  "why do",
  "symptoms of",
  "causes of",
  "treatment for",
  "recovery from",
  "risks of",
  "side effects",
  "preparation for",
  "after a",
  "before a",
  "difference between",
  "is it safe",
  "should i worry",
  "normal to",
  "how long does",
];

const PRICING_KEYWORDS = [
  "cost",
  "price",
  "pricing",
  "how much",
  "fee",
  "fees",
  "estimate",
  "estimated",
  "charge",
  "charges",
  "cash price",
  "self pay",
  "self-pay",
  "out of pocket",
  "insurance",
];

export function isSymptomQuery(message: string) {
  const msg = (message || "").toLowerCase();
  return SYMPTOM_KEYWORDS.some((kw) => msg.includes(kw));
}

export function isEducationQuery(message: string) {
  const msg = (message || "").toLowerCase();
  return EDUCATION_KEYWORDS.some((kw) => msg.includes(kw)) &&
    !PRICING_KEYWORDS.some((kw) => msg.includes(kw));
}

export function extractExplicitCptCode(message: string): string | null {
  const match = (message || "").match(/\bCPT\s*(?:code)?\s*(\d{5})\b/i);
  return match?.[1] || null;
}

export function extractZip(message: string): string | null {
  const msg = (message || "").trim();
  const explicit = msg.match(/\b(?:my\s+)?zip\s*(?:code)?\s*(?:is)?[:\s]+(\d{5})\b/i);
  if (explicit) return explicit[1];
  const near = msg.match(/\bnear\s+(?:zip\s*)?(\d{5})\b/i);
  if (near) return near[1];
  const blocks = msg.match(/\b(\d{5})\b/g);
  if (!blocks || blocks.length === 0) return null;
  const cpt = extractExplicitCptCode(msg);
  if (blocks.length === 1 && cpt && blocks[0] === cpt) return null;
  if (cpt) {
    const nonCpt = blocks.find((b) => b !== cpt);
    return nonCpt || null;
  }
  return blocks[0];
}

export function extractServiceQuery(message: string) {
  const msg = (message || "").trim().replace(/[?.!]+$/g, "");
  if (!msg) return "";
  const lowered = msg.toLowerCase();
  const prefixes = [
    "how much does",
    "how much is",
    "find",
    "search",
    "price for",
    "prices for",
    "cost of",
    "what is the cost of",
  ];
  let query = msg;
  for (const prefix of prefixes) {
    if (lowered.startsWith(prefix)) {
      query = msg.slice(prefix.length).trim();
      break;
    }
  }

  if (!query) return "";

  const cutoffPatterns = [
    /\b(?:my\s+)?zip(?:\s+code)?(?:\s+is)?\b/i,
    /\b(?:i(?:'m| am)?|we're|we are)\s+paying\s+with\b/i,
    /\b(?:paying|pay|paid)\s+with\b/i,
    /\b(?:using|via)\b/i,
    /\binsurance\b/i,
    /\binsured\b/i,
  ];

  let cutoff = query.length;
  for (const pattern of cutoffPatterns) {
    const match = query.match(pattern);
    if (match?.index != null && match.index >= 0) {
      cutoff = Math.min(cutoff, match.index);
    }
  }

  const pricingWordIndex = query.match(/\b(?:cost|price|pricing|charge|charges|fee|fees)\b/i)?.index;
  if (pricingWordIndex != null && pricingWordIndex >= 0) {
    cutoff = Math.min(cutoff, pricingWordIndex);
  }

  return query.slice(0, cutoff).replace(/[?.!,]+$/g, "").trim();
}

export function extractInsurance(message: string) {
  const msg = (message || "").toLowerCase();
  const carriers = [
    "blue cross blue shield",
    "unitedhealthcare",
    "united healthcare",
    "kaiser permanente",
    "blue cross",
    "bcbs",
    "anthem",
    "aetna",
    "cigna",
    "united",
    "uhc",
    "humana",
    "molina",
    "centene",
    "wellcare",
    "medicaid",
    "medicare",
    "tricare",
    "oscar",
  ];
  const found = carriers.find((carrier) => msg.includes(carrier));
  if (found) return found;
  if (msg.includes("insurance") || msg.includes("insured")) return "insurance";
  return "";
}
