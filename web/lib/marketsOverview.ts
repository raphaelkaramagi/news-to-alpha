import { TICKERS } from "./tickers";
import { chartWindowDays, type ChartWindow } from "./chartWindow";
import { getApiBaseUrl } from "./env";

export type MarketsOverviewData = {
  window: number;
  price_index: { date: string; index: number }[];
  accuracy_series: { date: string; accuracy: number }[];
  summary: { n: number; hits: number; accuracy: number | null };
};

function sanitizeJson(text: string): string {
  return text.replace(/:\s*NaN\b/g, ": null").replace(/:\s*-NaN\b/g, ": null");
}

async function fetchAccuracySummaryAll(window: ChartWindow): Promise<MarketsOverviewData["summary"] & { rows: { date: string; hit: number }[] }> {
  const res = await fetch(`/api/accuracy-summary?ticker=ALL&window=${window}`, { cache: "no-store" });
  if (!res.ok) throw new Error("accuracy-summary failed");
  const data = await res.json();
  return {
    n: data.n ?? 0,
    hits: data.hits ?? 0,
    accuracy: data.accuracy ?? null,
    rows: (data.rows ?? []).map((r: { date: string; hit: number }) => ({
      date: r.date,
      hit: r.hit ?? 0,
    })),
  };
}

function accuracySeriesFromRows(rows: { date: string; hit: number }[]): { date: string; accuracy: number }[] {
  const byDate = new Map<string, { hits: number; n: number }>();
  for (const r of rows) {
    const cur = byDate.get(r.date) ?? { hits: 0, n: 0 };
    cur.n += 1;
    cur.hits += r.hit ? 1 : 0;
    byDate.set(r.date, cur);
  }
  return Array.from(byDate.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, { hits, n }]) => ({ date, accuracy: n ? hits / n : 0 }));
}

async function buildPriceIndexClient(days: number): Promise<{ date: string; index: number }[]> {
  const histories = await Promise.all(
    TICKERS.map(async (ticker) => {
      const res = await fetch(`/api/history?ticker=${ticker}&window=${days}`, { cache: "no-store" });
      if (!res.ok) return null;
      const data = JSON.parse(sanitizeJson(await res.text())) as {
        prices: { date: string; close: number }[];
      };
      return { ticker, prices: data.prices ?? [] };
    })
  );

  const valid = histories.filter(Boolean) as { ticker: string; prices: { date: string; close: number }[] }[];
  if (!valid.length) return [];

  const allDates = new Set<string>();
  for (const h of valid) {
    for (const p of h.prices) allDates.add(p.date);
  }
  const dates = Array.from(allDates).sort().slice(-days);
  const byDate: Record<string, number[]> = {};

  for (const h of valid) {
    const priceMap = new Map(h.prices.map((p) => [p.date, p.close]));
    const seriesDates = dates.filter((d) => priceMap.has(d));
    if (!seriesDates.length) continue;
    const base = priceMap.get(seriesDates[0])!;
    if (!base) continue;
    for (const d of seriesDates) {
      const close = priceMap.get(d)!;
      (byDate[d] ??= []).push((close / base) * 100);
    }
  }

  return dates
    .filter((d) => byDate[d]?.length)
    .map((d) => ({
      date: d,
      index: byDate[d].reduce((a, b) => a + b, 0) / byDate[d].length,
    }));
}

/** Client-side fallback when Flask /api/markets-overview is unavailable. */
export async function fetchMarketsOverviewFallback(window: ChartWindow): Promise<MarketsOverviewData> {
  const days = chartWindowDays(window);
  const summaryBlock = await fetchAccuracySummaryAll(window);
  const accuracy_series = accuracySeriesFromRows(summaryBlock.rows);
  const price_index = await buildPriceIndexClient(days);

  return {
    window: days,
    price_index,
    accuracy_series,
    summary: {
      n: summaryBlock.n,
      hits: summaryBlock.hits,
      accuracy: summaryBlock.accuracy,
    },
  };
}

export async function fetchMarketsOverview(window: ChartWindow): Promise<MarketsOverviewData> {
  const res = await fetch(`/api/markets-overview?window=${window}`, { cache: "no-store" });
  if (res.ok) {
    const data = (await res.json()) as MarketsOverviewData & { error?: string };
    if (!data.error && (data.price_index?.length || data.accuracy_series?.length)) {
      return data;
    }
  }
  return fetchMarketsOverviewFallback(window);
}

/** Server-side: try Flask markets-overview, else aggregate via Flask sub-routes. */
export async function fetchMarketsOverviewServer(window: ChartWindow): Promise<MarketsOverviewData | null> {
  const base = getApiBaseUrl();
  const days = chartWindowDays(window);
  if (base) {
    try {
      const res = await fetch(`${base}/api/markets-overview?window=${days}`, {
        cache: "no-store",
        signal: AbortSignal.timeout(15_000),
      });
      if (res.ok) {
        return (await res.json()) as MarketsOverviewData;
      }
    } catch {
      /* fall through */
    }
  }
  return null;
}
