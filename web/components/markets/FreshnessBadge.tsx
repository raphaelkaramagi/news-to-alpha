import type { DataStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  status: DataStatus;
}

export function FreshnessBadge({ status }: Props) {
  const behind = status.trading_sessions_behind ?? -1;
  const latest = status.latest_prediction_date;
  const expected = status.expected_latest_prediction_date ?? status.latest_price_date;
  const isCurrent =
    status.is_current ??
    Boolean(latest && expected && latest >= expected);

  if (!latest) {
    return <span className="text-xs text-muted-foreground">No data yet</span>;
  }

  if (behind === 0 || isCurrent) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs">
        <span className="size-1.5 rounded-full bg-up" />
        <span className="text-muted-foreground">Through {latest}</span>
      </span>
    );
  }

  if (behind > 0) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs" title={`Prices through ${expected ?? "?"}`}>
        <span className={cn("size-1.5 rounded-full", behind <= 1 ? "bg-yellow-500" : "bg-down")} />
        <span className="text-muted-foreground">
          {behind} behind · preds {latest}
          {expected ? ` / prices ${expected}` : ""}
        </span>
      </span>
    );
  }

  return (
    <span className="text-xs text-muted-foreground">Through {latest}</span>
  );
}
