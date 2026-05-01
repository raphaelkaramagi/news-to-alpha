import Link from "next/link";
import { notFound } from "next/navigation";
import { TICKERS } from "@/lib/tickers";
import { fetchTickerServer } from "@/lib/data";
import { TickerDetail } from "@/components/TickerDetail";

export default async function TickerPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol: raw } = await params;
  const symbol = raw.toUpperCase();
  if (!TICKERS.includes(symbol)) {
    notFound();
  }

  const { ok, data, status, apiError } = await fetchTickerServer(symbol, "ensemble");
  let err: string | null = null;
  if (!ok || !data) {
    err =
      apiError ??
      (status === 503
        ? "API unavailable — check API_BASE_URL and predictions CSV on the server."
        : status === 404
          ? "No prediction for this ticker yet."
          : "Could not load prediction.");
  }

  return (
    <TickerDetail symbol={symbol} initial={data} initialError={err} />
  );
}
