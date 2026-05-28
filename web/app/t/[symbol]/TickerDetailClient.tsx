"use client";
import { useState, useCallback, useEffect } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import type { TickerApiResponse } from "@/lib/types";
import type { ModelId } from "@/lib/tickers";
import type { ChartWindow } from "@/lib/chartWindow";
import { CallHero } from "@/components/ticker/CallHero";
import { ModelPicker } from "@/components/ticker/ModelPicker";
import { DateScrubber } from "@/components/ticker/DateScrubber";
import { HeadlinesList } from "@/components/ticker/HeadlinesList";
import { RationaleBars } from "@/components/ticker/RationaleBars";
import { PriceAccuracySection } from "@/components/ticker/PriceAccuracySection";
import { AdvancedPanel } from "@/components/ticker/AdvancedPanel";
import { ModelBlurb } from "@/components/ticker/ModelBlurb";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { cn } from "@/lib/utils";

async function fetchTicker(
  ticker: string,
  model: ModelId,
  date: string | null
): Promise<TickerApiResponse> {
  const params = new URLSearchParams({ ticker, model });
  if (date) params.set("date", date);
  const res = await fetch(`/api/ticker?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

interface Props {
  symbol: string;
  initialData: TickerApiResponse | null;
  initialDate: string | null;
  urlDate?: string | null;
  backendError?: string;
}

type Tab = "headlines" | "why" | "advanced";

export function TickerDetailClient({
  symbol,
  initialData,
  initialDate,
  urlDate,
  backendError,
}: Props) {
  const { selectedDate, setSelectedDate } = useSelectedDate();
  const [model, setModel] = useState<ModelId>("ensemble");
  const [tab, setTab] = useState<Tab>("headlines");
  const [chartWindow, setChartWindow] = useState<ChartWindow>("30");

  // Sync global date from URL on first load
  useEffect(() => {
    if (urlDate) setSelectedDate(urlDate);
    else if (initialDate) setSelectedDate(initialDate);
  }, [urlDate, initialDate, setSelectedDate]);

  const activeDate = selectedDate ?? initialDate;

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["ticker-detail", symbol, model, activeDate],
    queryFn: () => fetchTicker(symbol, model, activeDate),
    initialData:
      model === "ensemble" && activeDate === initialDate
        ? (initialData ?? undefined)
        : undefined,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });

  const handleDateChange = useCallback(
    (newDate: string) => {
      setSelectedDate(newDate);
    },
    [setSelectedDate]
  );

  if (backendError && !data) {
    return (
      <div className="rounded-lg border px-4 py-3 text-sm text-muted-foreground">
        Backend unavailable: {backendError}
      </div>
    );
  }

  if ((!data || !activeDate) && isLoading) {
    return (
      <div className="space-y-4">
        <div className="h-20 rounded-lg bg-muted animate-pulse" />
        <div className="h-10 rounded-lg bg-muted animate-pulse" />
        <div className="h-48 rounded-lg bg-muted animate-pulse" />
      </div>
    );
  }

  if (!data || !activeDate) {
    return null;
  }

  return (
    <div className={cn("space-y-6 animate-fade-in", isFetching && "opacity-80")}>
      <CallHero data={data} date={activeDate} />

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <ModelPicker selected={model} onChange={setModel} perModel={data.per_model} />
        <DateScrubber ticker={symbol} selected={activeDate} onChange={handleDateChange} />
      </div>

      <ModelBlurb model={model} />

      <div>
        <div className="flex gap-1 border-b mb-4">
          {(["headlines", "why", "advanced"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                tab === t
                  ? "border-foreground text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {t === "headlines" ? "Headlines" : t === "why" ? "Why this call" : "Advanced"}
            </button>
          ))}
        </div>
        {tab === "headlines" && <HeadlinesList ticker={symbol} date={activeDate} />}
        {tab === "why" && (
          <RationaleBars
            ticker={symbol}
            date={activeDate}
            model={model}
            onOpenHeadlines={() => setTab("headlines")}
          />
        )}
        {tab === "advanced" && (
          <AdvancedPanel
            ticker={symbol}
            date={activeDate}
            model={model}
            perModel={data.per_model}
            tickerData={data}
          />
        )}
      </div>

      <PriceAccuracySection
        ticker={symbol}
        selectedDate={activeDate}
        targetDate={data.price_context?.target_date}
        model={model}
        window={chartWindow}
        onWindowChange={setChartWindow}
      />
    </div>
  );
}
