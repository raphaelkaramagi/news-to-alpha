"use client";
import { cn } from "@/lib/utils";

export type OutcomeMarkVariant = "correct" | "wrong" | "pending" | "loading";

type Props = {
  variant: OutcomeMarkVariant;
  size?: "sm" | "md";
  className?: string;
  title?: string;
};

const SIZE = {
  sm: "size-[18px] text-[10px]",
  md: "size-5 text-xs",
};

/** Minimal ✓ / ✗ / · mark — distinct from UP/DOWN direction colors. */
export function OutcomeMark({
  variant,
  size = "sm",
  className,
  title,
}: Props) {
  const label =
    variant === "correct"
      ? "Correct"
      : variant === "wrong"
        ? "Wrong"
        : variant === "pending"
          ? "Pending"
          : "Loading";

  return (
    <span
      title={title ?? label}
      aria-label={label}
      className={cn(
        "shrink-0 rounded-sm inline-flex items-center justify-center font-medium leading-none",
        SIZE[size],
        variant === "correct" && "bg-up/15 text-up",
        variant === "wrong" && "bg-down/15 text-down",
        (variant === "pending" || variant === "loading") &&
          "border border-border bg-muted/30 text-muted-foreground",
        className
      )}
    >
      {variant === "correct" && "✓"}
      {variant === "wrong" && "✗"}
      {(variant === "pending" || variant === "loading") && "·"}
    </span>
  );
}

export function OutcomeLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-2">
        <OutcomeMark variant="correct" size="sm" />
        correct
      </span>
      <span className="inline-flex items-center gap-2">
        <OutcomeMark variant="wrong" size="sm" />
        wrong
      </span>
      <span className="inline-flex items-center gap-2">
        <OutcomeMark variant="pending" size="sm" />
        pending
      </span>
    </div>
  );
}
