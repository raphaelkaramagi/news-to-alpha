"use client";
import { useQuery } from "@tanstack/react-query";
import { BarChart3, ChevronRight, Newspaper, TrendingDown, TrendingUp } from "lucide-react";
import type { EnsembleExplanation, RationaleResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import type { ModelId } from "@/lib/tickers";
import { ENSEMBLE_INPUT_WHY, MODEL_DISPLAY_LABELS } from "@/lib/models";
import { buildClientExplanation } from "@/lib/ensembleExplainClient";
import { confidenceLabel } from "@/lib/confidence";

interface Props {
  ticker: string;
  date: string;
  model: ModelId;
  onOpenHeadlines?: () => void;
}

type Vote = EnsembleExplanation["base_votes"][number];

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

function voteLabel(vote: Vote): string {
  const map: Record<string, string> = {
    lstm: MODEL_DISPLAY_LABELS.lstm,
    tfidf: MODEL_DISPLAY_LABELS.tfidf,
    embeddings: MODEL_DISPLAY_LABELS.embeddings,
  };
  return map[vote.model] ?? vote.label;
}

function inputWhyCopy(vote: Vote): string | null {
  if (vote.model === "lstm" || vote.model === "tfidf" || vote.model === "embeddings") {
    return ENSEMBLE_INPUT_WHY[vote.model];
  }
  return null;
}

function headlineCount(data: RationaleResponse): number | null {
  const row = data.meta_features?.find((f) => f.key === "n_headlines");
  if (row && row.value > 0) return Math.round(row.value);
  return data.has_news ? null : 0;
}

function agreementSummary(votes: Vote[]): { label: string; tone: "up" | "down" | "mixed" | "neutral" } {
  const active = votes.filter((v) => v.active !== false && v.direction !== "N/A");
  const up = active.filter((v) => v.direction === "UP").length;
  const down = active.length - up;

  if (active.length === 0) {
    return { label: "Price only", tone: "neutral" };
  }
  if (up === active.length) {
    return { label: `${up}/${active.length} inputs lean UP`, tone: "up" };
  }
  if (down === active.length) {
    return { label: `${down}/${active.length} inputs lean DOWN`, tone: "down" };
  }
  return { label: `Mixed · ${up} up · ${down} down`, tone: "mixed" };
}

function StatCard({
  children,
  className,
  onClick,
  disabled,
}: {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
}) {
  const Tag = onClick && !disabled ? "button" : "div";
  return (
    <Tag
      type={onClick && !disabled ? "button" : undefined}
      onClick={disabled ? undefined : onClick}
      className={cn(
        "rounded-lg border p-3 flex flex-col justify-center gap-1 min-h-[4.5rem] text-left w-full",
        onClick && !disabled && "hover:bg-accent transition-colors cursor-pointer group",
        disabled && "opacity-70",
        className
      )}
    >
      {children}
    </Tag>
  );
}

/** 0–100% gauge with 50% midpoint */
function ProbabilityGauge({
  proba,
  direction,
}: {
  proba: number;
  direction: "UP" | "DOWN";
}) {
  const pct = Math.round(proba * 100);
  const isUp = direction === "UP";

  return (
    <div className="space-y-2">
      <div className="flex items-end justify-between gap-2">
        <div className="flex items-center gap-2">
          {isUp ? (
            <TrendingUp className="size-5 text-up" aria-hidden />
          ) : (
            <TrendingDown className="size-5 text-down" aria-hidden />
          )}
          <span className={cn("text-2xl font-bold", isUp ? "text-up" : "text-down")}>
            {direction}
          </span>
        </div>
        <span className="text-sm font-mono tabular-nums text-muted-foreground">
          {pct}% chance of rising
        </span>
      </div>
      <div className="relative h-3 rounded-full bg-muted overflow-hidden">
        <div
          className="absolute top-0 bottom-0 w-px bg-foreground/30 z-10"
          style={{ left: "50%" }}
          aria-hidden
        />
        <div
          className={cn(
            "absolute top-0 bottom-0 rounded-full transition-all",
            isUp ? "bg-up/70" : "bg-down/70"
          )}
          style={{
            left: isUp ? "50%" : `${pct}%`,
            width: isUp ? `${pct - 50}%` : `${50 - pct}%`,
          }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 size-3.5 rounded-full border-2 border-background bg-foreground shadow-sm z-20"
          style={{ left: `calc(${pct}% - 7px)` }}
          aria-hidden
        />
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
        <span>0%</span>
        <span>50% even</span>
        <span>100%</span>
      </div>
    </div>
  );
}

function ModelVoteCard({ vote }: { vote: Vote }) {
  const inactive = vote.active === false || vote.direction === "N/A";
  const isUp = !inactive && vote.direction === "UP";
  const pct = Math.round(vote.proba * 100);
  const barWidth = inactive ? 0 : isUp ? pct : 100 - pct;
  const why = inputWhyCopy(vote);

  return (
    <div
      className={cn(
        "rounded-lg border p-4 flex flex-col gap-2 min-w-0 h-full",
        inactive && "opacity-50 bg-muted/20",
        !inactive && isUp && "border-up/25 bg-up/5",
        !inactive && !isUp && "border-down/25 bg-down/5"
      )}
    >
      <div className="flex items-center justify-between gap-1">
        <span className="text-xs font-semibold text-foreground truncate">
          {voteLabel(vote)}
        </span>
        {!inactive &&
          (isUp ? (
            <TrendingUp className="size-4 text-up shrink-0" />
          ) : (
            <TrendingDown className="size-4 text-down shrink-0" />
          ))}
      </div>
      {why && (
        <p className="text-[11px] text-muted-foreground leading-snug flex-1">{why}</p>
      )}
      <p
        className={cn(
          "text-xl font-bold tabular-nums leading-none",
          inactive ? "text-muted-foreground" : isUp ? "text-up" : "text-down"
        )}
      >
        {inactive ? "—" : isUp ? "UP" : "DOWN"}
      </p>
      {!inactive && (
        <>
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <div
              className={cn("h-full rounded-full", isUp ? "bg-up" : "bg-down")}
              style={{ width: `${barWidth}%` }}
            />
          </div>
          <p className="text-[11px] text-muted-foreground font-mono tabular-nums">
            {pct}% chance of rising
          </p>
        </>
      )}
      {inactive && (
        <p className="text-[11px] text-muted-foreground">Skipped — no headlines</p>
      )}
    </div>
  );
}

export function RationaleBars({ ticker, date, model, onOpenHeadlines }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["rationale", ticker, date, model],
    queryFn: () => fetchRationale(ticker, date, model),
    staleTime: 300_000,
    enabled: model === "ensemble",
  });

  if (model !== "ensemble") {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Switch to <span className="font-medium text-foreground">Ensemble</span> to see
        how the call was built.
      </p>
    );
  }

  if (isLoading) {
    return <div className="h-40 rounded-lg bg-muted animate-pulse" />;
  }

  if (error || !data) {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Explanation isn&apos;t available for this date yet.
      </p>
    );
  }

  const exp = data.explanation ?? buildClientExplanation(data);
  if (!exp) {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Explanation isn&apos;t available for this date yet.
      </p>
    );
  }

  const hasNews = data.has_news ?? exp.has_news;
  const headlines = headlineCount(data);
  const confLabel = exp.confidence_label ?? confidenceLabel(exp.ensemble_confidence);
  const confPct = Math.round(exp.ensemble_confidence * 100);
  const agreement = agreementSummary(exp.base_votes);
  const canOpenHeadlines = hasNews && headlines !== 0 && !!onOpenHeadlines;

  return (
    <div className="space-y-5">
      <div className="rounded-lg border p-4 space-y-4">
        <ProbabilityGauge proba={exp.ensemble_proba} direction={exp.ensemble_direction} />

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          <StatCard
            className={cn(
              agreement.tone === "up" && "border-up/30 bg-up/5",
              agreement.tone === "down" && "border-down/30 bg-down/5"
            )}
          >
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
              Input agreement
            </span>
            <span
              className={cn(
                "text-sm font-semibold leading-snug",
                agreement.tone === "up" && "text-up",
                agreement.tone === "down" && "text-down"
              )}
            >
              {agreement.label}
            </span>
          </StatCard>

          <StatCard>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground inline-flex items-center gap-1">
              <BarChart3 className="size-3" aria-hidden />
              Confidence
            </span>
            <span className="text-sm font-semibold">{confLabel} confidence</span>
            <span className="text-2xl font-bold tabular-nums leading-none">{confPct}%</span>
            <span className="text-[10px] text-muted-foreground">from 50% even</span>
          </StatCard>

          <StatCard
            onClick={canOpenHeadlines ? onOpenHeadlines : undefined}
            disabled={!canOpenHeadlines}
          >
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground inline-flex items-center gap-1">
              <Newspaper className="size-3" aria-hidden />
              Headlines
            </span>
            <span className="text-sm font-semibold flex items-center gap-1">
              {hasNews
                ? headlines != null
                  ? `${headlines} before close`
                  : "Used in call"
                : "None · price only"}
              {canOpenHeadlines && (
                <ChevronRight className="size-4 text-muted-foreground group-hover:text-foreground" />
              )}
            </span>
            {canOpenHeadlines && (
              <span className="text-[10px] text-muted-foreground">View headlines</span>
            )}
          </StatCard>
        </div>

        <p className="text-[11px] text-muted-foreground leading-relaxed border-t pt-3">
          <span className="font-medium text-foreground">Confidence </span> measures how far
          the rising chance sits from 50% even — a stronger lean, not how often the model
          has been right. Past accuracy is under Price &amp; accuracy below.
        </p>
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-1">Three inputs</p>
        <p className="text-[11px] text-muted-foreground mb-3 leading-relaxed">
          Each model outputs P(up): probability the next close is above today&apos;s close.
          {hasNews
            ? " With headlines, a trained combiner merges the three votes into the gauge above (weights in Advanced)."
            : " No headlines today — only the price model ran, then the price-only combiner."}
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 items-stretch">
          {exp.base_votes.map((v) => (
            <ModelVoteCard key={v.model} vote={v} />
          ))}
        </div>
      </div>

      {exp.models_disagree && hasNews && (
        <p className="text-xs text-muted-foreground border-l-2 border-muted pl-3">
          Inputs disagreed — the combiner weighted them to reach the final call above.
        </p>
      )}
    </div>
  );
}
