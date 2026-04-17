import { z } from "zod";
import * as cheerio from "cheerio";
import { callNemotron } from "@/lib/nemotron";

type WebSearchInput = {
  serviceQuery: string;
  cptCode?: string;
  zip?: string;
  insurance?: string;
};

export type WebPriceCandidate = {
  hospitalName: string;
  address: string | null;
  city: string | null;
  state: string | null;
  zipcode: string | null;
  websiteUrl: string | null;
  sourceTitle: string | null;
  sourceSnippet: string | null;
  price: number | null;
  insuranceMatch: boolean;
  confidence: number;
};

const AI_RESPONSE_SCHEMA = z.object({
  results: z.array(
    z.object({
      hospitalName: z.string().min(1),
      address: z.string().nullable().default(null),
      city: z.string().nullable().default(null),
      state: z.string().nullable().default(null),
      zipcode: z.string().nullable().default(null),
      websiteUrl: z.string().nullable().default(null),
      sourceTitle: z.string().nullable().default(null),
      sourceSnippet: z.string().nullable().default(null),
      price: z.number().nullable().default(null),
      insuranceMatch: z.boolean().default(false),
      confidence: z.number().min(0).max(1).default(0),
    })
  ),
});

type BraveResponse = Record<string, unknown>;

type BraveResult = {
  title: string;
  url: string;
  description: string;
  sourceName: string | null;
  extraSnippets: string[];
};

type BravePageDocument = BraveResult & {
  finalUrl: string;
  contentType: string | null;
  pageText: string;
};

function normalizeText(value: unknown) {
  return String(value || "").trim();
}

function normalizeWhitespace(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function clipText(value: string, maxLength = 12000) {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength)}\n...[truncated]`;
}

function extractPricingExcerpts(text: string) {
  const lines = text
    .split(/\n+/)
    .map((line) => normalizeWhitespace(line))
    .filter(Boolean);
  const priceLine = /(\$[\s0-9,]+|\b(price|cost|charge|estimate|estimated|amount|cash|self-pay|negotiated|fee)\b)/i;
  return clipText(lines.filter((line) => priceLine.test(line)).slice(0, 40).join("\n"), 6000);
}

function extractVisibleHtmlText(html: string) {
  const $ = cheerio.load(html);
  $("script,style,noscript,svg,iframe,header,footer,nav,form,button").remove();

  const title = normalizeWhitespace($("title").first().text());
  const description =
    normalizeWhitespace($('meta[name="description"]').attr("content") || "") ||
    normalizeWhitespace($('meta[property="og:description"]').attr("content") || "");

  const tableRows: string[] = [];
  $("table")
    .slice(0, 10)
    .each((_, table) => {
      $(table)
        .find("tr")
        .slice(0, 80)
        .each((__, tr) => {
          const rowText = normalizeWhitespace($(tr).text());
          if (rowText) {
            tableRows.push(rowText);
          }
        });
    });

  const bodyText = normalizeWhitespace($("body").text());
  return clipText([title, description, ...tableRows, bodyText].filter(Boolean).join("\n\n"));
}

async function fetchBravePageDocument(result: BraveResult) {
  if (!result.url) return null;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(result.url, {
      method: "GET",
      redirect: "follow",
      signal: controller.signal,
      headers: {
        Accept: "text/html,application/pdf;q=0.9,*/*;q=0.8",
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      },
    });

    if (!response.ok) return null;

    const contentType = response.headers.get("content-type") || null;
    const finalUrl = response.url || result.url;

    if (contentType?.includes("pdf") || finalUrl.toLowerCase().endsWith(".pdf")) {
      const buffer = Buffer.from(await response.arrayBuffer());
      const pdfParseModule = await import("pdf-parse");
      const pdfParse =
        (pdfParseModule as unknown as {
          default?: (data: Buffer) => Promise<{ text: string }>;
        }).default || (pdfParseModule as unknown as (data: Buffer) => Promise<{ text: string }>);
      const parsed = await pdfParse(buffer);
      const pageText = clipText(normalizeWhitespace(parsed.text || ""));
      return {
        ...result,
        url: result.url,
        finalUrl,
        contentType,
        pageText,
      } satisfies BravePageDocument;
    }

    const html = await response.text();
    const pageText = extractVisibleHtmlText(html);
    return {
      ...result,
      url: result.url,
      finalUrl,
      contentType,
      pageText,
    } satisfies BravePageDocument;
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

function extractPrice(value: string) {
  const priceMatch = value.match(/\$\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2})?)/);
  if (!priceMatch) return null;
  const parsed = Number(priceMatch[1].replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function collectBraveResults(payload: BraveResponse) {
  const rawResults =
    (payload.web && typeof payload.web === "object" && Array.isArray((payload.web as Record<string, unknown>).results)
      ? ((payload.web as Record<string, unknown>).results as Record<string, unknown>[])
      : null) ||
    (Array.isArray(payload.results) ? (payload.results as Record<string, unknown>[]) : []);

  return rawResults
    .map((item) => {
      const extraSnippets = Array.isArray(item.extra_snippets)
        ? item.extra_snippets.map((entry) => normalizeText(entry))
        : Array.isArray(item.extraSnippets)
          ? item.extraSnippets.map((entry) => normalizeText(entry))
          : [];
      const sourceName =
        normalizeText(item.profile && typeof item.profile === "object" ? (item.profile as Record<string, unknown>).name : "") ||
        normalizeText(item.source_name) ||
        normalizeText(item.sourceName) ||
        null;

      return {
        title: normalizeText(item.title),
        url: normalizeText(item.url || item.link),
        description: normalizeText(item.description || item.snippet),
        sourceName,
        extraSnippets,
      } satisfies BraveResult;
    })
    .filter((item) => item.title || item.url || item.description);
}

async function searchBrave(query: string) {
  const apiKey = process.env.BRAVE_SEARCH_API_KEY || process.env.BRAVE_API_KEY || "";
  if (!apiKey) return [];

  const url = new URL("https://api.search.brave.com/res/v1/web/search");
  url.searchParams.set("q", query);
  url.searchParams.set("count", "10");
  url.searchParams.set("search_lang", "en");
  url.searchParams.set("country", "us");
  url.searchParams.set("spellcheck", "1");

  const response = await fetch(url.toString(), {
    headers: {
      Accept: "application/json",
      "X-Subscription-Token": apiKey,
    },
  });

  if (!response.ok) {
    throw new Error(`Brave search failed with status ${response.status}`);
  }

  const payload = (await response.json()) as BraveResponse;
  return collectBraveResults(payload);
}

function buildQueries(input: WebSearchInput) {
  const primary = input.cptCode
    ? `${input.cptCode} ${input.serviceQuery} price transparency ${input.zip || ""} ${input.insurance || ""}`.trim()
    : `${input.serviceQuery} price transparency ${input.zip || ""} ${input.insurance || ""}`.trim();
  const alternate = input.cptCode
    ? `${input.cptCode} hospital price ${input.zip || ""} ${input.insurance || ""}`.trim()
    : `${input.serviceQuery} hospital price ${input.zip || ""} ${input.insurance || ""}`.trim();
  const insurerQuery = input.insurance
    ? `${input.serviceQuery} ${input.insurance} price transparency ${input.zip || ""}`.trim()
    : "";
  return [primary, alternate, insurerQuery].filter(Boolean);
}

function toCandidateMap(results: BravePageDocument[], input: WebSearchInput) {
  const candidates = results
    .map((result) => {
      const combinedText = [result.title, result.description, result.pageText, ...result.extraSnippets].filter(Boolean).join(" ");
      const price = extractPrice(combinedText);
      if (price == null) return null;
      const host = (() => {
        try {
          return new URL(result.finalUrl || result.url).hostname.replace(/^www\./, "");
        } catch {
          return "";
        }
      })();

      return {
        hospitalName: result.sourceName || result.title || host || input.serviceQuery,
        address: null,
        city: null,
        state: null,
        zipcode: input.zip || null,
        websiteUrl: result.finalUrl || result.url || null,
        sourceTitle: result.title || null,
        sourceSnippet: result.pageText || result.description || null,
        price,
        insuranceMatch: Boolean(
          input.insurance &&
            combinedText.toLowerCase().includes(input.insurance.toLowerCase())
        ),
        confidence: result.description ? 0.72 : 0.6,
      } satisfies WebPriceCandidate;
    })
    .filter(Boolean) as WebPriceCandidate[];

  const seen = new Set<string>();
  return candidates.filter((candidate) => {
    const key = `${candidate.hospitalName.toLowerCase()}|${candidate.price}|${candidate.websiteUrl || ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function searchWebPricing(input: WebSearchInput) {
  const queryList = buildQueries(input);
  const braveResults = (
    await Promise.all(
      queryList.map(async (query) => ({
        query,
        results: await searchBrave(query).catch(() => [] as BraveResult[]),
      }))
    )
  ).flatMap((entry) => entry.results);

  if (!braveResults.length) {
    return { results: [] as WebPriceCandidate[], sources: [] as BraveResult[] };
  }

  const bravePageDocuments = (
    await Promise.all(
      braveResults.slice(0, 8).map(async (result) => fetchBravePageDocument(result))
    )
  ).filter(Boolean) as BravePageDocument[];

  if (!bravePageDocuments.length) {
    return { results: [] as WebPriceCandidate[], sources: braveResults };
  }

  const rawCandidates = toCandidateMap(bravePageDocuments, input).slice(0, 10);
  if (!rawCandidates.length) {
    return { results: [] as WebPriceCandidate[], sources: braveResults };
  }

  const aiPrompt = {
    serviceQuery: input.serviceQuery,
    cptCode: input.cptCode || "",
    zip: input.zip || "",
    insurance: input.insurance || "",
    instructions:
      "Only return items that explicitly show a price in the opened page content, PDF text, or table rows. Do not invent hospitals, prices, or addresses. Prefer the strongest direct match to the service and ZIP.",
    pages: bravePageDocuments.map((page) => ({
      title: page.title,
      url: page.finalUrl || page.url,
      sourceName: page.sourceName,
      description: page.description,
      pageText: extractPricingExcerpts(page.pageText),
    })),
  };

  try {
    const content = await callNemotron([
      {
        role: "system",
        content:
          "You are selecting actual web-search price results for a healthcare pricing app. Only use the provided candidates. Do not change a price unless it is explicitly present in the candidate text. Return strict JSON with this shape: {results:[{hospitalName:string,address:string|null,city:string|null,state:string|null,zipcode:string|null,websiteUrl:string|null,sourceTitle:string|null,sourceSnippet:string|null,price:number|null,insuranceMatch:boolean,confidence:number}]}",
      },
      {
        role: "user",
        content: JSON.stringify(aiPrompt),
      },
    ]);

    if (content) {
      const cleaned = content
        .trim()
        .replace(/^```(?:json)?/i, "")
        .replace(/```$/i, "")
        .trim();
      const parsed = AI_RESPONSE_SCHEMA.parse(JSON.parse(cleaned));
      return {
        results: parsed.results
        .filter((entry) => entry.price != null)
        .map((entry) => ({
          hospitalName: entry.hospitalName,
          address: entry.address,
          city: entry.city,
          state: entry.state,
          zipcode: entry.zipcode,
          websiteUrl: entry.websiteUrl,
          sourceTitle: entry.sourceTitle,
          sourceSnippet: entry.sourceSnippet,
          price: entry.price,
          insuranceMatch: entry.insuranceMatch,
          confidence: entry.confidence,
        })) as WebPriceCandidate[],
        sources: braveResults,
      };
    }
  } catch (error) {
    console.warn("Web pricing AI extraction failed, falling back to raw Brave candidates:", error);
  }

  return {
    results: rawCandidates,
    sources: braveResults,
  };
}
