"use client";
import type { TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

type Props = {
  data: TickerApiResponse | undefined;
  className?: string;
  size?: number;
};

const UP = "#22c55e";
const DOWN = "#ef4444";
const MUTED = "#a1a1aa";

function ColorDot({
  color,
  size,
  ring,
  className,
}: {
  color: string;
  size: number;
  ring?: boolean;
  className?: string;
}) {
  if (ring) {
    return (
      <div
        className={cn("shrink-0", className)}
        style={{
          width: size,
          height: size,
          borderRadius: "50%",
          border: `2px solid ${color}`,
          backgroundColor: "#ffffff",
          boxSizing: "border-box",
          display: "inline-block",
        }}
        aria-hidden
      />
    );
  }
  return (
    <div
      className={cn("shrink-0", className)}
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        backgroundColor: color,
        display: "inline-block",
      }}
      aria-hidden
    />
  );
}

function isResolved(data: TickerApiResponse): boolean {
  const a = data.actual_binary;
  return a !== null && a !== undefined && !Number.isNaN(Number(a));
}

/** Filled = resolved (green correct / red wrong). Ring = pending forecast. */
export function OutcomeDot({ data, className, size = 10 }: Props) {
  if (!data) {
    return <ColorDot color={MUTED} size={size} className={className} />;
  }

  if (isResolved(data)) {
    const correct = Number(data.hit) === 1;
    return (
      <div
        className={cn("shrink-0", className)}
        title={correct ? "Correct call" : "Wrong call"}
        aria-label={correct ? "Correct" : "Wrong"}
      >
        <ColorDot color={correct ? UP : DOWN} size={size} />
      </div>
    );
  }

  const ringColor = data.binary === 1 ? UP : DOWN;
  return <ColorDot color={ringColor} size={size} ring className={className} />;
}

export function OutcomeLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-2">
        <ColorDot color={UP} size={10} />
        correct
      </span>
      <span className="inline-flex items-center gap-2">
        <ColorDot color={DOWN} size={10} />
        wrong
      </span>
      <span className="inline-flex items-center gap-2">
        <ColorDot color={UP} size={10} ring />
        pending
      </span>
    </div>
  );
}
