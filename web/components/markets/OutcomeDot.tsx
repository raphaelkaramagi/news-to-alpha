"use client";
import type { TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { OutcomeMark, type OutcomeMarkVariant } from "@/components/shared/OutcomeMark";

type Props = {
  data: TickerApiResponse | undefined;
  className?: string;
  size?: "sm" | "md";
};

function isResolved(data: TickerApiResponse): boolean {
  const a = data.actual_binary;
  return a !== null && a !== undefined && !Number.isNaN(Number(a));
}

function variantFor(data: TickerApiResponse | undefined): OutcomeMarkVariant {
  if (!data) return "loading";
  if (!isResolved(data)) return "pending";
  return Number(data.hit) === 1 ? "correct" : "wrong";
}

export function OutcomeDot({ data, className, size = "sm" }: Props) {
  const variant = variantFor(data);
  const title =
    variant === "correct"
      ? "Correct call"
      : variant === "wrong"
        ? "Wrong call"
        : variant === "pending"
          ? "Outcome pending"
          : undefined;

  return (
    <OutcomeMark
      variant={variant}
      size={size}
      className={cn(className)}
      title={title}
    />
  );
}

export { OutcomeLegend } from "@/components/shared/OutcomeMark";
