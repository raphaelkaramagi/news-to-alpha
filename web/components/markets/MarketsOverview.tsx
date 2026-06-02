"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChartWindowPicker } from "@/components/charts/ChartWindowPicker";
import { PriceIndexChart } from "@/components/charts/PriceIndexChart";
import { AccuracyTraceChart } from "@/components/charts/AccuracyTraceChart";
import { VolatilityTraceChart } from "@/components/charts/VolatilityTraceChart";
import { chartWindowDays, type ChartWindow } from "@/lib/chartWindow";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { fetchMarketsOverview } from "@/lib/marketsOverview";

export function MarketsOverview() {
  const [window, setWindow] = useState<ChartWindow>("30");
  const { selectedDate } = useSelectedDate();

  const { data, isLoading, isError } = useQuery({
    queryKey: ["markets-overview", window],
    queryFn: () => fetchMarketsOverview(window),
    staleTime: 60_000,
    retry: 1,
  });

  return (
    <section className="space-y-6 border rounded-lg p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-medium">All tickers · price & accuracy</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Equal-weight basket vs. % of correct ensemble calls · last {chartWindowDays(window)} sessions
          </p>
        </div>
        <ChartWindowPicker value={window} onChange={setWindow} />
      </div>

      {isLoading ? (
        <div className="space-y-4">
          <div className="h-60 rounded-lg bg-muted animate-pulse" />
          <div className="h-40 rounded-lg bg-muted animate-pulse" />
        </div>
      ) : isError || !data ? (
        <p className="text-sm text-muted-foreground py-6 text-center">
          Could not load overview data. Restart Flask (<code className="text-xs">python app/server.py --port 8000</code>).
        </p>
      ) : (
        <>
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">Basket price (indexed)</p>
            <p className="text-[10px] text-muted-foreground mb-2">100 = start of window · all 20 tickers weighted equally</p>
            <PriceIndexChart
              data={data.price_index}
              selectedDate={selectedDate ?? undefined}
            />
          </div>

          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">Hit rate by session</p>
            <p className="text-[10px] text-muted-foreground mb-2">Share of tickers the ensemble got right each day</p>
            <AccuracyTraceChart
              window={window}
              selectedDate={selectedDate ?? undefined}
              series={data.accuracy_series}
              label="Hit rate"
            />
          </div>

          <div>
            <p className="text-sm font-medium mb-1">Direction · window total</p>
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-bold tabular-nums">
                {data.summary.accuracy !== null
                  ? `${(data.summary.accuracy * 100).toFixed(0)}%`
                  : "—"}
              </span>
              <span className="text-sm text-muted-foreground">
                {data.summary.hits}/{data.summary.n} calls correct
              </span>
            </div>
          </div>

          <div className="pt-4 border-t space-y-6">
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">
                Expected-move band accuracy by session
              </p>
              <p className="text-[10px] text-muted-foreground mb-2">
                Share of tickers whose realized |return| stayed inside the ±% band
              </p>
              <VolatilityTraceChart
                window={window}
                selectedDate={selectedDate ?? undefined}
                series={data.volatility_series}
              />
            </div>

            <div>
              <p className="text-sm font-medium mb-1">Volatility · window total</p>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-bold tabular-nums">
                  {data.volatility_summary.accuracy !== null
                    ? `${(data.volatility_summary.accuracy * 100).toFixed(0)}%`
                    : "—"}
                </span>
                <span className="text-sm text-muted-foreground">
                  {data.volatility_summary.hits}/{data.volatility_summary.n} within band
                </span>
              </div>
              {data.volatility_summary.mae_pct != null && (
                <p className="text-xs text-muted-foreground mt-2 tabular-nums">
                  Avg calibration error {data.volatility_summary.mae_pct.toFixed(2)}%
                </p>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
