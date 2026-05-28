import { fetchDataStatusServer } from "@/lib/data";
import { TickerGrid } from "@/components/markets/TickerGrid";

export const dynamic = "force-dynamic";

export default async function MarketsPage() {
  const status = await fetchDataStatusServer().catch(() => null);
  const latest =
    status?.primary_prediction_date ??
    status?.expected_latest_prediction_date ??
    status?.latest_price_date ??
    null;
  const noData = !latest;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Markets</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Next-session direction forecasts
        </p>
      </div>

      {noData && (
        <div className="rounded-lg border border-border bg-muted/40 px-4 py-3 text-sm text-muted-foreground">
          No predictions available yet. Run the pipeline locally:
          <code className="ml-2 font-mono text-xs bg-muted px-1.5 py-0.5 rounded">
            python scripts/run_pipeline.py --preset balanced
          </code>
        </div>
      )}

      {!noData && <TickerGrid />}
    </div>
  );
}
