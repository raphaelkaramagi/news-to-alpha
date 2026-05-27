import { notFound } from "next/navigation";
import { fetchTickerServer } from "@/lib/data";
import { TICKERS } from "@/lib/tickers";
import { TickerDetailClient } from "./TickerDetailClient";

export const dynamic = "force-dynamic";

interface Props {
  params: Promise<{ symbol: string }>;
  searchParams: Promise<{ date?: string }>;
}

export default async function TickerPage({ params, searchParams }: Props) {
  const { symbol } = await params;
  const { date: urlDate } = await searchParams;
  const upper = symbol.toUpperCase();
  if (!TICKERS.includes(upper as typeof TICKERS[number])) notFound();

  const result = await fetchTickerServer(upper, "ensemble");
  const initialData = result.ok ? result.data : null;
  const latestDate = initialData?.prediction_date ?? null;

  return (
    <TickerDetailClient
      symbol={upper}
      initialData={initialData}
      initialDate={latestDate}
      urlDate={urlDate ?? null}
      backendError={!result.ok ? result.apiError : undefined}
    />
  );
}
