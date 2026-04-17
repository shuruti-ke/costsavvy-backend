import { z } from "zod";
import * as cheerio from "cheerio";
import { callNemotron } from "@/lib/nemotron";

type WebSearchInput = {
  serviceQuery: string;
  cptCode?: string;
  serviceDescription?: string;
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

function extractPdfLinks(html: string, baseUrl: string) {
  try {
    const $ = cheerio.load(html);
    const links = new Set<string>();
    $("a[href]").each((_, el) => {
      const href = normalizeText($(el).attr("href"));
      if (!href) return;
      const isPdf = /\.pdf([?#].*)?$/i.test(href) || /pdf/i.test($(el).text());
      if (!isPdf) return;
      try {
        links.add(new URL(href, baseUrl).toString());
      } catch {
        // ignore invalid URLs
      }
    });
    return Array.from(links).slice(0, 2);
  } catch {
    return [];
  }
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

  const structuredSnippets: string[] = [];
  $("script")
    .slice(0, 8)
    .each((_, script) => {
      const type = normalizeText($(script).attr("type")).toLowerCase();
      const scriptText = normalizeWhitespace($(script).text());
      const shouldCapture =
        type.includes("ld+json") ||
        type.includes("json") ||
        scriptText.includes("price") ||
        scriptText.includes("cost") ||
        scriptText.includes("charge");
      if (!shouldCapture || !scriptText) return;
      structuredSnippets.push(clipText(scriptText, 3000));
    });

  const bodyText = normalizeWhitespace($("body").text());
  return clipText([title, description, ...tableRows, ...structuredSnippets, bodyText].filter(Boolean).join("\n\n"));
}

async function fetchBravePageDocument(result: BraveResult, depth = 0) {
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
    const linkedPdfTexts: string[] = [];
    if (depth < 1) {
      const linkedPdfs = extractPdfLinks(html, finalUrl);
      for (const pdfUrl of linkedPdfs) {
        const linked = await fetchBravePageDocument(
          {
            title: `${result.title} PDF`,
            url: pdfUrl,
            description: result.description,
            sourceName: result.sourceName,
            extraSnippets: [],
          },
          depth + 1
        );
        if (linked?.pageText) {
          linkedPdfTexts.push(`PDF from ${linked.finalUrl || pdfUrl}\n${linked.pageText}`);
        }
      }
    }
    const pageText = extractVisibleHtmlText(html);
    return {
      ...result,
      url: result.url,
      finalUrl,
      contentType,
      pageText: clipText([pageText, ...linkedPdfTexts].filter(Boolean).join("\n\n")),
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

function scorePatientPortalSignals(value: string) {
  const text = value.toLowerCase();
  let score = 0;

  const weightedSignals: Array<[RegExp, number]> = [
    [/mychartplus|my chart plus/i, 6],
    [/mychart/i, 5],
    [/patient estimate|guest estimate|get estimate|estimate details/i, 5],
    [/estimate for|your estimate|service estimate/i, 4],
    [/you pay|total fees|insurance covers|coverage information/i, 4],
    [/patient portal|guest estimates?|estimate details/i, 3],
    [/start over|select a different service/i, 2],
    [/hartford health|hhc|hartford healthcare/i, 2],
    [/patient|guests?|consumer/i, 1],
  ];

  for (const [pattern, weight] of weightedSignals) {
    if (pattern.test(text)) score += weight;
  }

  return score;
}

function isHospitalLinkedDocument(result: BravePageDocument) {
  const combined = [result.title, result.sourceName, result.description, result.pageText, result.finalUrl || result.url]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return (
    /\bhospital\b/.test(combined) ||
    /\bhealth\s*care\b/.test(combined) ||
    /\bhealthcare\b/.test(combined) ||
    /\bhealth system\b/.test(combined) ||
    /\bmedical center\b/.test(combined) ||
    /\bmedical centre\b/.test(combined) ||
    /\bpatient portal\b/.test(combined) ||
    /\bguest estimate\b/.test(combined) ||
    /\bmychartplus\b/.test(combined) ||
    /\bmychart\b/.test(combined) ||
    /\bhartford health\b/.test(combined) ||
    /\bhhc\b/.test(combined) ||
    /\bclinic\b/.test(combined) ||
    /\bphysician\b/.test(combined) ||
    /\bprovider\b/.test(combined)
  );
}

function scoreWebDocument(result: BravePageDocument) {
  const combined = [result.title, result.description, result.pageText, result.finalUrl, ...result.extraSnippets]
    .filter(Boolean)
    .join(" ");
  const portalScore = scorePatientPortalSignals(combined);
  const priceScore = extractPrice(combined) != null ? 4 : 0;
  const tableScore = /table|fee schedule|estimate|you pay|coverage/i.test(combined) ? 2 : 0;
  const payerScore = /aetna|blue cross|bcbs|uhc|unitedhealthcare|humana|cigna|medicare|medicaid/i.test(combined) ? 2 : 0;
  return portalScore + priceScore + tableScore + payerScore;
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
  url.searchParams.set("count", "20");
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
  const serviceText = input.serviceDescription || input.serviceQuery;
  const base = input.cptCode ? `${input.cptCode} ${serviceText}`.trim() : serviceText;
  const zipPart = input.zip ? ` ${input.zip}` : "";
  const insurerPart = input.insurance ? ` ${input.insurance}` : "";

  const queries = [
    `${base} mychartplus estimate${zipPart}${insurerPart}`,
    `${base} mychartplus patient estimate${zipPart}${insurerPart}`,
    `${base} mychartplus guest estimate${zipPart}${insurerPart}`,
    `${base} mychart estimate${zipPart}${insurerPart}`,
    `${base} patient estimate${zipPart}${insurerPart}`,
    `${base} guest estimate${zipPart}${insurerPart}`,
    `${base} estimate details${zipPart}${insurerPart}`,
    `${base} get estimate${zipPart}${insurerPart}`,
    `${base} hartford health estimate${zipPart}${insurerPart}`,
    `${base} estimate${zipPart}${insurerPart}`,
    `${base} patient estimates${zipPart}${insurerPart}`,
    `${base} price transparency${zipPart}${insurerPart}`,
    `${base} fee schedule${zipPart}${insurerPart}`,
    `${base} chargemaster${zipPart}${insurerPart}`,
    `${base} shoppable services${zipPart}${insurerPart}`,
    `${base} hospital price${zipPart}${insurerPart}`,
    `${base} standard charges${zipPart}${insurerPart}`,
    `${base} PDF${zipPart}${insurerPart}`,
  ];

  if (input.insurance) {
    queries.push(`${base} ${input.insurance} contracted rate${zipPart}`.trim());
  }

  return Array.from(new Set(queries.map((query) => query.trim()).filter(Boolean)));
}

function toCandidateMap(results: BravePageDocument[], input: WebSearchInput, options: { allowNoPrice?: boolean } = {}) {
  const candidates = results
    .map((result) => {
      const combinedText = [result.title, result.description, result.pageText, ...result.extraSnippets].filter(Boolean).join(" ");
      const price = extractPrice(combinedText);
      const host = (() => {
        try {
          return new URL(result.finalUrl || result.url).hostname.replace(/^www\./, "");
        } catch {
          return "";
        }
      })();
      const confidence = price != null ? (result.description ? 0.72 : 0.6) : 0.35;
      if (price == null && !options.allowNoPrice) return null;

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
        confidence,
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

  const uniqueBraveResults = Array.from(
    new Map(braveResults.map((result) => [result.url || result.title, result])).values()
  );

  const bravePageDocuments = (
    await Promise.all(
      uniqueBraveResults.map(async (result) => fetchBravePageDocument(result))
    )
  ).filter(Boolean) as BravePageDocument[];

  if (!bravePageDocuments.length) {
    return { results: [] as WebPriceCandidate[], sources: uniqueBraveResults };
  }

  bravePageDocuments.sort((a, b) => scoreWebDocument(b) - scoreWebDocument(a));
  const hospitalLinkedDocuments = bravePageDocuments.filter((page) => isHospitalLinkedDocument(page));

  if (!hospitalLinkedDocuments.length) {
    return { results: [] as WebPriceCandidate[], sources: uniqueBraveResults };
  }

  const rawCandidates = toCandidateMap(hospitalLinkedDocuments, input);
  if (!rawCandidates.length) {
    const fallbackCandidates = toCandidateMap(hospitalLinkedDocuments, input, { allowNoPrice: true });
    return {
      results: fallbackCandidates.map((candidate) => ({
        ...candidate,
        price: candidate.price,
        confidence: 0.35,
      })) as WebPriceCandidate[],
      sources: uniqueBraveResults,
    };
  }

  const aiPrompt = {
    serviceQuery: input.serviceQuery,
    cptCode: input.cptCode || "",
    zip: input.zip || "",
    insurance: input.insurance || "",
    instructions:
      "Only return items that explicitly show a price in the opened page content, PDF text, or table rows. Do not invent hospitals, prices, or addresses. Prefer the strongest direct match to the service and ZIP.",
    pages: hospitalLinkedDocuments.map((page) => ({
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
        sources: uniqueBraveResults,
      };
    }
  } catch (error) {
    console.warn("Web pricing AI extraction failed, falling back to raw Brave candidates:", error);
  }

  return {
    results: rawCandidates,
    sources: uniqueBraveResults,
  };
}
