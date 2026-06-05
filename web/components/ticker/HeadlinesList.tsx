"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { HeadlinesResponse } from "@/lib/types";
import { ChevronDown, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

const DEFAULT_VISIBLE = 3;

interface Props {
  ticker: string;
  date: string;
}

async function fetchHeadlines(ticker: string, date: string): Promise<HeadlinesResponse> {
  const res = await fetch(`/api/headlines?ticker=${ticker}&date=${date}`, { cache: "no-store" });
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

// Dot reflects FinBERT headline sentiment (signed P(pos) − P(neg), range −1..1).
// Legend at the top of the list explains the colors.
function SentimentDot({ score }: { score: number | null | undefined }) {
  const color =
    score === null || score === undefined
      ? "bg-muted-foreground/30"
      : score > 0.1
        ? "bg-up"
        : score < -0.1
          ? "bg-down"
          : "bg-muted-foreground/40";
  return <span className={`size-1.5 rounded-full inline-block flex-shrink-0 mt-2 ${color}`} />;
}

function HeadlineRow({ h }: { h: HeadlinesResponse["headlines"][0] }) {
  const text = h.headline || h.title || "";
  const sentiment = h.finnhub_sentiment ?? h.sentiment_finnhub ?? null;

  return (
    <div className="flex items-start gap-3 py-3 border-b last:border-0">
      <SentimentDot score={sentiment} />
      <div className="flex-1 min-w-0">
        {h.url ? (
          <a
            href={h.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm leading-snug hover:underline underline-offset-2"
          >
            {text || "View article"}
          </a>
        ) : (
          <p className="text-sm leading-snug">{text || "Untitled"}</p>
        )}
        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
          {h.source && <span>{h.source}</span>}
          {h.published_at && (
            <span>
              {new Date(h.published_at).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          )}
        </div>
      </div>
      {h.url && (
        <a
          href={h.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-muted-foreground hover:text-foreground transition-colors flex-shrink-0 mt-0.5"
          aria-label="Open article"
        >
          <ExternalLink size={14} />
        </a>
      )}
    </div>
  );
}

export function HeadlinesList({ ticker, date }: Props) {
  const [expanded, setExpanded] = useState(false);
  const { data, isLoading, error } = useQuery({
    queryKey: ["headlines", ticker, date],
    queryFn: () => fetchHeadlines(ticker, date),
    staleTime: 300_000,
  });

  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-14 rounded-md bg-muted animate-pulse" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Could not load headlines — API unavailable. Check that the backend is running.
      </p>
    );
  }

  if (!data?.headlines?.length) {
    return (
      <p className="text-sm text-muted-foreground py-4">
        No headlines found for {date}. The news feed may not have covered this ticker on this date.
      </p>
    );
  }

  const headlines = data.headlines;
  const hasMore = headlines.length > DEFAULT_VISIBLE;
  const visible = expanded ? headlines : headlines.slice(0, DEFAULT_VISIBLE);

  return (
    <div>
      <div className="flex items-center gap-3 text-[11px] text-muted-foreground pb-2 mb-1 border-b">
        <span className="font-medium uppercase tracking-wide">FinBERT sentiment</span>
        <span className="flex items-center gap-1"><span className="size-1.5 rounded-full bg-up inline-block" /> positive</span>
        <span className="flex items-center gap-1"><span className="size-1.5 rounded-full bg-down inline-block" /> negative</span>
        <span className="flex items-center gap-1"><span className="size-1.5 rounded-full bg-muted-foreground/40 inline-block" /> neutral</span>
      </div>
      <div className="space-y-1">
        {visible.map((h, i) => (
          <HeadlineRow key={`${h.url ?? h.title}-${i}`} h={h} />
        ))}
      </div>
      {hasMore && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className={cn(
            "mt-3 flex items-center gap-1.5 text-xs text-muted-foreground",
            "hover:text-foreground transition-colors"
          )}
        >
          <ChevronDown
            size={14}
            className={cn("transition-transform", expanded && "rotate-180")}
          />
          {expanded
            ? "Show fewer"
            : `Show ${headlines.length - DEFAULT_VISIBLE} more headline${headlines.length - DEFAULT_VISIBLE !== 1 ? "s" : ""}`}
        </button>
      )}
    </div>
  );
}
