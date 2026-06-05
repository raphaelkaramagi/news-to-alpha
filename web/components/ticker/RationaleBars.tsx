"use client";
import { useQuery } from "@tanstack/react-query";
import { ChevronRight } from "lucide-react";
import type { EnsembleExplanation, RationaleResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import type { ModelId } from "@/lib/tickers";
import { MODEL_DISPLAY_LABELS } from "@/lib/models";
import { buildClientExplanation } from "@/lib/ensembleExplainClient";
import { confidenceLabel } from "@/lib/confidence";

interface Props {
  ticker: string;
  date: string;
  model: ModelId;
  onOpenHeadlines?: () => void;
}

type Vote = EnsembleExplanation["base_votes"][number];
type Driver = EnsembleExplanation["drivers"][number];

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

function headlineCount(data: RationaleResponse): number | null {
  const row = data.meta_features?.find((f) => f.key === "n_headlines");
  if (row && row.value > 0) return Math.round(row.value);
  return data.has_news ? null : 0;
}

function activeVotes(votes: Vote[]): Vote[] {
  return votes.filter((v) => v.active !== false && v.direction !== "N/A");
}

function agreementSummary(votes: Vote[]): string {
  const active = activeVotes(votes);
  const up = active.filter((v) => v.direction === "UP").length;
  const down = active.length - up;
  if (active.length === 0) return "Price only";
  if (up === active.length) return `${up}/${active.length} lean UP`;
  if (down === active.length) return `${down}/${active.length} lean DOWN`;
  return `${up} up · ${down} down`;
}

function summarySentence(
  exp: EnsembleExplanation,
  hasNews: boolean,
  confLabel: string,
  ensPct: number
): string {
  const dir = exp.ensemble_direction;
  const active = activeVotes(exp.base_votes);
  const up = active.filter((v) => v.direction === "UP").length;
  const down = active.length - up;

  if (!hasNews) {
    return `Price-only call: ${dir} at ${ensPct}% chance of rising (${confLabel.toLowerCase()} confidence).`;
  }
  if (up === active.length || down === active.length) {
    return `All inputs lean ${dir}; ensemble agrees at ${ensPct}% UP (${confLabel.toLowerCase()} confidence).`;
  }
  if (exp.disagreement?.flips_base_lean) {
    const lean = Math.round(exp.disagreement.base_lean_proba * 100);
    return `Inputs average ${lean}% UP, but the combiner calls ${dir} at ${ensPct}% — context shifted the score.`;
  }
  return `${dir} at ${ensPct}% UP (${confLabel.toLowerCase()} confidence) with ${up} of ${active.length} inputs leaning up.`;
}

function inputsSentence(exp: EnsembleExplanation, hasNews: boolean): string {
  if (!hasNews) {
    return "Only the price model ran today — no headlines before the close.";
  }
  const active = activeVotes(exp.base_votes);
  const up = active.filter((v) => v.direction === "UP").length;
  if (up === active.length) return "All three models see a rising chance; dots mark each model's estimate on the 0–100% scale.";
  if (up === 0) return "All three models lean down; dots mark each model's estimate on the 0–100% scale.";
  return "Each model's rising chance before the combiner merges them — dot position is the estimate, 50 is even.";
}

function weightedSentence(
  drivers: Driver[],
  baselinePct: number,
  finalPct: number,
  hasNews: boolean
): string {
  const setup = hasNews ? "news day" : "price-only day";
  const directional = drivers.filter((d) => d.direction !== "neutral");
  const start = `Before today's signals, a ${setup} like this averages ${baselinePct}% chance up (the combiner's starting point).`;
  if (directional.length === 0) {
    return `${start} Today's signals barely moved it — final ${finalPct}% chance up.`;
  }
  const top = directional[0];
  const push = top.direction === "up" ? "up" : "down";
  return `${start} ${top.label} moved it the most (${push} ${Math.abs(Math.round(top.effect * 100))} pts); the bars add up to the final ${finalPct}% chance up.`;
}

/** 0–100 scale with dot at model estimate; 50% tick = even odds. */
function ProbabilityScale({ proba }: { proba: number }) {
  const pct = Math.round(proba * 100);
  const isUp = proba >= 0.5;

  return (
    <div className="space-y-1">
      <div className="relative h-2 rounded-full bg-muted overflow-hidden">
        <div
          className={cn(
            "absolute inset-y-0 left-0 rounded-full opacity-50",
            isUp ? "bg-up" : "bg-down"
          )}
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute top-0 bottom-0 w-px bg-foreground/25 z-10"
          style={{ left: "50%" }}
          aria-hidden
        />
        <div
          className={cn(
            "absolute top-1/2 -translate-y-1/2 size-2.5 rounded-full border-2 border-background shadow-sm z-20",
            isUp ? "bg-up" : "bg-down"
          )}
          style={{ left: `calc(${pct}% - 5px)` }}
          aria-hidden
        />
      </div>
      <div className="flex justify-between text-[9px] text-muted-foreground font-mono tabular-nums">
        <span>0</span>
        <span>50 even</span>
        <span>100</span>
      </div>
    </div>
  );
}

function ModelVoteCard({ vote }: { vote: Vote }) {
  const inactive = vote.active === false || vote.direction === "N/A";
  const isUp = !inactive && vote.direction === "UP";
  const pct = Math.round(vote.proba * 100);

  return (
    <div
      className={cn(
        "rounded-lg border p-3 flex flex-col gap-2 min-w-0",
        inactive ? "opacity-50 bg-muted/20" : "bg-background"
      )}
    >
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {voteLabel(vote)}
      </span>
      {inactive ? (
        <span className="text-sm text-muted-foreground">No headlines</span>
      ) : (
        <>
          <div className="flex items-baseline justify-between gap-2">
            <span
              className={cn(
                "text-sm font-semibold tabular-nums",
                isUp ? "text-up" : "text-down"
              )}
            >
              {isUp ? "UP" : "DOWN"}
            </span>
            <span className="text-xs font-mono tabular-nums text-muted-foreground">
              {pct}% rising
            </span>
          </div>
          <ProbabilityScale proba={vote.proba} />
        </>
      )}
    </div>
  );
}

/** Driver effect is a Shapley contribution in probability points (signed). */
function DriverBar({ driver, maxAbs }: { driver: Driver; maxAbs: number }) {
  const neutral = driver.direction === "neutral";
  const isUp = driver.direction === "up";
  const pts = Math.round(driver.effect * 100);
  const width = neutral ? 4 : Math.min(100, Math.max(10, (Math.abs(driver.effect) / maxAbs) * 100));

  return (
    <div className="flex items-center gap-3 text-sm min-w-0">
      <span className="text-muted-foreground truncate w-[7.5rem] shrink-0 text-xs">
        {driver.label}
      </span>
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full",
            neutral ? "bg-border" : isUp ? "bg-up/70" : "bg-down/70"
          )}
          style={{ width: `${width}%` }}
        />
      </div>
      <span
        className={cn(
          "text-[10px] font-mono tabular-nums shrink-0 w-12 text-right",
          neutral ? "text-muted-foreground" : isUp ? "text-up" : "text-down"
        )}
      >
        {neutral ? "0" : `${pts >= 0 ? "+" : ""}${pts}`}
      </span>
    </div>
  );
}

/** Small labelled anchor (baseline / final) for the waterfall. */
function WaterfallAnchor({ label, pct }: { label: string; pct: number }) {
  const up = pct >= 50;
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn("font-semibold tabular-nums", up ? "text-up" : "text-down")}>
        {pct}% chance up
      </span>
    </div>
  );
}

function MetricCell({
  label,
  value,
  sub,
  onClick,
  clickable,
}: {
  label: string;
  value: string;
  sub?: string;
  onClick?: () => void;
  clickable?: boolean;
}) {
  const Tag = clickable ? "button" : "div";
  return (
    <Tag
      type={clickable ? "button" : undefined}
      onClick={onClick}
      className={cn(
        "px-3 py-2.5 flex flex-col gap-0.5 min-w-0 text-left",
        clickable && "hover:bg-background/60 transition-colors cursor-pointer group"
      )}
    >
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <span className="text-sm font-medium leading-snug">{value}</span>
      {sub && (
        <span className="text-[10px] text-muted-foreground flex items-center gap-0.5">
          {sub}
          {clickable && (
            <ChevronRight className="size-3 opacity-60 group-hover:opacity-100" />
          )}
        </span>
      )}
    </Tag>
  );
}

function Section({
  title,
  sentence,
  children,
  accent,
}: {
  title: string;
  sentence: string;
  children: React.ReactNode;
  accent?: "up" | "down" | "neutral" | "highlight";
}) {
  return (
    <div
      className={cn(
        "rounded-lg border p-3 space-y-3",
        accent === "up" && "border-up/25 bg-up/[0.04]",
        accent === "down" && "border-down/25 bg-down/[0.04]",
        accent === "highlight" && "border-up/30 bg-up/[0.06]",
        accent === "highlight" && "dark:bg-up/[0.08]",
        (!accent || accent === "neutral") && "bg-muted/30"
      )}
    >
      <div>
        <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
          {title}
        </p>
        <p className="text-sm text-muted-foreground leading-snug">{sentence}</p>
      </div>
      {children}
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
    return <div className="h-32 rounded-lg bg-muted animate-pulse" />;
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
  const ensPct = Math.round(exp.ensemble_proba * 100);
  const canOpenHeadlines = hasNews && headlines !== 0 && !!onOpenHeadlines;
  const isUp = exp.ensemble_direction === "UP";

  // Shapley contributions: a "typical call" baseline plus signed per-signal
  // effects (probability points) that sum exactly to the final call. Showing
  // the baseline is what lets the bars reconcile with the call direction.
  const drivers = (exp.drivers ?? []).slice(0, 5);
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.effect)), 0.01);
  const baselinePct =
    exp.baseline_proba != null ? Math.round(exp.baseline_proba * 100) : null;

  // News-wording signals present but flat — explains the common "~0" bars.
  const newsWordingDrivers = drivers.filter(
    (d) => d.feature === "news_tfidf_pred_proba" || d.feature === "news_embeddings_pred_proba"
  );
  const sentimentMuted =
    !!hasNews &&
    newsWordingDrivers.length > 0 &&
    newsWordingDrivers.every((d) => d.direction === "neutral");

  return (
    <div className="space-y-4">
      {/* Summary + context metrics */}
      <Section
        title="Why this call"
        sentence={summarySentence(exp, !!hasNews, confLabel, ensPct)}
        accent={isUp ? "up" : "down"}
      >
        <div className="rounded-md border bg-background/80 grid grid-cols-3 divide-x overflow-hidden">
          <MetricCell label="Agreement" value={agreementSummary(exp.base_votes)} />
          <MetricCell
            label="Confidence"
            value={`${confLabel} · ${confPct}%`}
            sub="from 50% even"
          />
          <MetricCell
            label="Headlines"
            value={
              hasNews
                ? headlines != null
                  ? `${headlines} before close`
                  : "Used"
                : "None"
            }
            sub={canOpenHeadlines ? "View" : hasNews ? undefined : "Price only"}
            onClick={canOpenHeadlines ? onOpenHeadlines : undefined}
            clickable={canOpenHeadlines}
          />
        </div>
      </Section>

      {/* Model inputs */}
      <Section
        title="Model inputs"
        sentence={inputsSentence(exp, !!hasNews)}
        accent="neutral"
      >
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {exp.base_votes.map((v) => (
            <ModelVoteCard key={v.model} vote={v} />
          ))}
        </div>
      </Section>

      {/* What weighted the call — waterfall: baseline -> contributions -> final */}
      {drivers.length > 0 && (
        <Section
          title="What weighted the call"
          sentence={weightedSentence(
            drivers,
            baselinePct ?? ensPct,
            ensPct,
            !!hasNews
          )}
          accent="neutral"
        >
          <div className="rounded-md border bg-background/80 p-3 space-y-2.5">
            {baselinePct != null && (
              <>
                <WaterfallAnchor
                  label={hasNews ? "Starting point (typical news day)" : "Starting point (typical price-only day)"}
                  pct={baselinePct}
                />
                <div className="border-t" />
              </>
            )}
            {drivers.map((d) => (
              <DriverBar key={d.feature} driver={d} maxAbs={maxAbs} />
            ))}
            <div className="border-t" />
            <WaterfallAnchor label="Final call" pct={ensPct} />
          </div>
          <p className="text-[10px] text-muted-foreground leading-snug">
            Bars are each signal&apos;s contribution in probability points; they add to
            the starting point to reach the final call. News volume = headline count vs a
            typical day; Price = the LSTM price-direction model.
          </p>
          {sentimentMuted && (
            <p className="text-[10px] text-muted-foreground/80 leading-snug">
              Keywords and Sentiment show ~0 here: the combiner counts that headlines
              exist (News volume), but learned that their wording/tone barely predicts the
              next session&apos;s direction, so it leans on price and market context instead.
            </p>
          )}
        </Section>
      )}
    </div>
  );
}
