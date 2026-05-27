"use client";
import { useQuery } from "@tanstack/react-query";
import type { RationaleResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import type { ModelId } from "@/lib/tickers";
import { buildClientExplanation } from "@/lib/ensembleExplainClient";
import { CONFIDENCE_HELP, confidenceLabel } from "@/lib/confidence";

interface Props {
  ticker: string;
  date: string;
  model: ModelId;
}

type Explanation = NonNullable<RationaleResponse["explanation"]>;
type Vote = Explanation["base_votes"][number];
type Driver = Explanation["drivers"][number];

const BAR_UP = "#22c55e";
const BAR_DOWN = "#ef4444";
const BAR_NEUTRAL = "#d4d4d8";

async function fetchRationale(
  ticker: string,
  date: string,
  model: string
): Promise<RationaleResponse> {
  const res = await fetch(`/api/rationale?ticker=${ticker}&date=${date}&model=${model}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

function EmText({ text }: { text: string }) {
  const parts = text.split(/\*\*(.*?)\*\*/g);
  return (
    <span>
      {parts.map((part, i) =>
        i % 2 === 1 ? <strong key={i}>{part}</strong> : <span key={i}>{part}</span>
      )}
    </span>
  );
}

function VotePill({ vote }: { vote: Vote }) {
  const isUp = vote.direction === "UP";
  return (
    <div className="rounded-md border px-3 py-2 text-center min-w-0 flex-1">
      <p className="text-[11px] text-muted-foreground">{vote.label}</p>
      <p className={cn("text-sm font-semibold tabular-nums mt-0.5", isUp ? "text-up" : "text-down")}>
        {(vote.proba * 100).toFixed(0)}% UP
      </p>
    </div>
  );
}

function DriverRow({ driver }: { driver: Driver }) {
  const neutral = driver.direction === "neutral";
  const isUp = driver.direction === "up";
  const barColor = neutral ? BAR_NEUTRAL : isUp ? BAR_UP : BAR_DOWN;
  const pct = neutral ? 8 : Math.min(100, Math.max(14, Math.abs(driver.effect) * 400));
  const tag = neutral ? "No impact" : isUp ? "Pushed UP" : "Pushed DOWN";

  return (
    <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 items-center text-sm">
      <span className="text-muted-foreground truncate">{driver.label}</span>
      <span
        className={cn(
          "text-[11px] font-medium shrink-0",
          neutral ? "text-muted-foreground" : isUp ? "text-up" : "text-down"
        )}
      >
        {tag}
      </span>
      <div className="col-span-2 h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={cn("h-full rounded-full", neutral && "opacity-60")}
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}

export function RationaleBars({ ticker, date, model }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["rationale", ticker, date, model],
    queryFn: () => fetchRationale(ticker, date, model),
    staleTime: 300_000,
    enabled: model === "ensemble",
  });

  if (model !== "ensemble") {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Switch to <span className="font-medium text-foreground">Ensemble</span> for the combined explanation.
      </p>
    );
  }

  if (isLoading) {
    return <div className="h-28 rounded-lg bg-muted animate-pulse" />;
  }

  if (error || !data) {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Why-this-call breakdown isn&apos;t available for this date yet.
      </p>
    );
  }

  const exp = data.explanation ?? buildClientExplanation(data);
  if (!exp) {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Why-this-call breakdown isn&apos;t available for this date yet.
      </p>
    );
  }

  const isUp = exp.ensemble_direction === "UP";
  const confPct = (exp.ensemble_confidence * 100).toFixed(0);
  const label = exp.confidence_label ?? confidenceLabel(exp.ensemble_confidence);

  return (
    <div className="space-y-5">
      <div
        className={cn(
          "rounded-lg border px-4 py-3 space-y-2",
          isUp ? "border-up/30 bg-up/5" : "border-down/30 bg-down/5"
        )}
      >
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <p className={cn("text-lg font-semibold", isUp ? "text-up" : "text-down")}>
            Ensemble: {exp.ensemble_direction}
          </p>
          <p className="text-sm text-muted-foreground tabular-nums">
            {(exp.ensemble_proba * 100).toFixed(0)}% chance of going up
          </p>
        </div>
        <p className="text-sm">
          <span className="font-medium">{label} confidence</span>
          <span className="text-muted-foreground tabular-nums"> · {confPct}%</span>
          <span className="text-muted-foreground"> — {exp.confidence_help ?? CONFIDENCE_HELP}</span>
        </p>
      </div>

      {(exp.bullets ?? [exp.summary]).map((b, i) => (
        <p key={i} className="text-sm leading-relaxed text-muted-foreground">
          <EmText text={b} />
        </p>
      ))}

      <div className="flex gap-2">
        {exp.base_votes.map((v) => (
          <VotePill key={v.model} vote={v} />
        ))}
      </div>

      {exp.drivers.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm font-medium">What moved the final score</p>
          <div className="space-y-3">
            {exp.drivers.map((d) => (
              <DriverRow key={d.feature} driver={d} />
            ))}
          </div>
        </div>
      )}

      {exp.news_weight_note && (
        <p className="text-xs text-muted-foreground border-l-2 border-muted pl-3">
          {exp.news_weight_note}
        </p>
      )}
    </div>
  );
}
