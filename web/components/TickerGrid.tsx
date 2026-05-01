import Link from "next/link";
import { TICKERS, TICKER_TO_COMPANY } from "@/lib/tickers";

export function TickerGrid({
  active,
  compact = false,
}: {
  active?: string;
  compact?: boolean;
}) {
  return (
    <div
      className={
        compact
          ? "grid grid-cols-3 gap-2 sm:grid-cols-5"
          : "grid grid-cols-3 gap-2 sm:grid-cols-5 md:grid-cols-8 lg:grid-cols-8"
      }
    >
      {TICKERS.map((t) => {
        const isActive = active === t;
        return (
          <Link
            key={t}
            href={`/ticker/${t}`}
            className={`rounded-xl border px-2 py-2 text-center transition sm:py-2.5 ${
              isActive
                ? "border-accent bg-surface-2 shadow-[0_0_0_1px_rgba(91,140,255,0.35)]"
                : "border-border bg-surface hover:border-accent/60 hover:bg-surface-2"
            } `}
          >
            <div className="text-sm font-bold tracking-tight">{t}</div>
            {!compact && (
              <div className="mt-0.5 truncate text-[0.65rem] text-muted">
                {TICKER_TO_COMPANY[t] ?? ""}
              </div>
            )}
          </Link>
        );
      })}
    </div>
  );
}
