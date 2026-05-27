import { NextRequest, NextResponse } from "next/server";
import { backendUrl, forwardToFlask } from "@/lib/backend";
import { getApiBaseUrl } from "@/lib/env";
import { TICKERS } from "@/lib/tickers";
import { chartWindowDays, type ChartWindow } from "@/lib/chartWindow";
import type { MarketsOverviewData } from "@/lib/marketsOverview";

export const dynamic = "force-dynamic";

function sanitizeJson(text: string): string {
  return text.replace(/:\s*NaN\b/g, ": null").replace(/:\s*-NaN\b/g, ": null");
}

async function buildFromFlaskSubroutes(window: ChartWindow): Promise<MarketsOverviewData | null> {
  const base = getApiBaseUrl();
  if (!base) return null;
  const days = chartWindowDays(window);

  try {
    const summaryRes = await fetch(`${base}/api/accuracy-summary?ticker=ALL&window=${window}`, {
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
    if (!summaryRes.ok) return null;
    const summaryJson = await summaryRes.json();
    const rows: { date: string; hit: number }[] = (summaryJson.rows ?? []).map(
      (r: { date: string; hit: number }) => ({ date: r.date, hit: r.hit ?? 0 })
    );

    const byDate = new Map<string, { hits: number; n: number }>();
    for (const r of rows) {
      const cur = byDate.get(r.date) ?? { hits: 0, n: 0 };
      cur.n += 1;
      cur.hits += r.hit ? 1 : 0;
      byDate.set(r.date, cur);
    }
    const accuracy_series = Array.from(byDate.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, { hits, n }]) => ({ date, accuracy: n ? hits / n : 0 }));

    const histories = await Promise.all(
      TICKERS.map(async (ticker) => {
        const res = await fetch(`${base}/api/history?ticker=${ticker}&window=${days}`, {
          cache: "no-store",
          signal: AbortSignal.timeout(15_000),
        });
        if (!res.ok) return null;
        const data = JSON.parse(sanitizeJson(await res.text())) as {
          prices: { date: string; close: number }[];
        };
        return data.prices ?? [];
      })
    );

    const allDates = new Set<string>();
    for (const prices of histories) {
      if (!prices) continue;
      for (const p of prices) allDates.add(p.date);
    }
    const dates = Array.from(allDates).sort().slice(-days);
    const byDateIndex: Record<string, number[]> = {};

    for (const prices of histories) {
      if (!prices?.length) continue;
      const map = new Map(prices.map((p) => [p.date, p.close]));
      const seriesDates = dates.filter((d) => map.has(d));
      if (!seriesDates.length) continue;
      const basePrice = map.get(seriesDates[0])!;
      for (const d of seriesDates) {
        (byDateIndex[d] ??= []).push((map.get(d)! / basePrice) * 100);
      }
    }

    const price_index = dates
      .filter((d) => byDateIndex[d]?.length)
      .map((d) => ({
        date: d,
        index: byDateIndex[d].reduce((a, b) => a + b, 0) / byDateIndex[d].length,
      }));

    return {
      window: days,
      price_index,
      accuracy_series,
      summary: {
        n: summaryJson.n ?? 0,
        hits: summaryJson.hits ?? 0,
        accuracy: summaryJson.accuracy ?? null,
      },
    };
  } catch {
    return null;
  }
}

export async function GET(req: NextRequest) {
  const windowParam = (req.nextUrl.searchParams.get("window") || "30") as ChartWindow;
  const target = backendUrl("/api/markets-overview", req.nextUrl.searchParams);

  if (target) {
    try {
      const res = await fetch(target, { cache: "no-store", signal: AbortSignal.timeout(20_000) });
      if (res.ok) {
        const text = await res.text();
        return new NextResponse(text, {
          status: 200,
          headers: { "content-type": "application/json; charset=utf-8" },
        });
      }
    } catch {
      /* fallback below */
    }
  }

  const built = await buildFromFlaskSubroutes(windowParam);
  if (built) {
    return NextResponse.json(built);
  }

  return NextResponse.json(
    { error: "Could not build markets overview — check API_BASE_URL and restart Flask." },
    { status: 503 }
  );
}
