import { fetchDataStatusServer } from "@/lib/data";
import type { DataStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

export const dynamic = "force-dynamic";

function Row({ label, value, mono = false }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between py-3 border-b last:border-0 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn("font-medium", mono && "font-mono text-xs")}>{value}</span>
    </div>
  );
}

function StatusIndicator({ ok }: { ok: boolean }) {
  return (
    <span className={cn(
      "inline-flex items-center gap-1.5",
      ok ? "text-up" : "text-down"
    )}>
      <span className={cn("size-1.5 rounded-full", ok ? "bg-up" : "bg-down")} />
      {ok ? "Current" : "Stale"}
    </span>
  );
}

export default async function StatusPage() {
  const status: DataStatus | null = await fetchDataStatusServer().catch(() => null);

  if (!status) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold tracking-tight">Status</h1>
        <div className="rounded-lg border px-4 py-3 text-sm text-muted-foreground">
          Backend unreachable. Check that the Flask API is running and{" "}
          <code className="font-mono text-xs bg-muted px-1 rounded">API_BASE_URL</code> is set.
        </div>
      </div>
    );
  }

  const behind = status.trading_sessions_behind;
  const isCurrent =
    status.is_current ??
    Boolean(
      status.latest_prediction_date &&
        status.latest_price_date &&
        status.latest_prediction_date >= status.latest_price_date
    );

  return (
    <div className="space-y-8 max-w-lg">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Status</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Data freshness and system health.
        </p>
      </div>

      <div className="rounded-lg border">
        <div className="px-4 py-3 border-b">
          <p className="text-sm font-medium">Predictions</p>
        </div>
        <div className="px-4">
          <Row
            label="Freshness"
            value={<StatusIndicator ok={isCurrent} />}
          />
          <Row
            label="Latest prediction"
            value={status.latest_prediction_date ?? "—"}
            mono
          />
          <Row
            label="Expected through"
            value={status.expected_latest_prediction_date ?? status.latest_price_date ?? "—"}
            mono
          />
          <Row
            label="Last trading session"
            value={status.last_trading_session ?? "—"}
            mono
          />
          <Row
            label="Sessions behind"
            value={
              behind < 0 ? "—" : behind === 0
                ? <span className="text-up">0</span>
                : <span className="text-down">{behind}</span>
            }
          />
          <Row
            label="Prediction rows"
            value={status.prediction_rows.toLocaleString()}
          />
        </div>
      </div>

      <div className="rounded-lg border">
        <div className="px-4 py-3 border-b">
          <p className="text-sm font-medium">Data</p>
        </div>
        <div className="px-4">
          <Row label="Latest price" value={status.latest_price_date ?? "—"} mono />
          <Row label="Latest news" value={status.latest_news_date ?? "—"} mono />
          <Row label="Price rows" value={status.price_rows.toLocaleString()} />
          <Row label="News rows" value={status.news_rows.toLocaleString()} />
        </div>
      </div>

      <div className="rounded-lg border">
        <div className="px-4 py-3 border-b">
          <p className="text-sm font-medium">System</p>
        </div>
        <div className="px-4">
          <Row label="Deploy mode" value={status.deploy_mode} mono />
          <Row
            label="Last published"
            value={
              status.last_published_at
                ? new Date(status.last_published_at).toLocaleString()
                : "—"
            }
          />
          <Row label="As of" value={status.today} mono />
        </div>
      </div>

      <div className="rounded-lg border px-4 py-3 text-sm text-muted-foreground space-y-1">
        <p className="font-medium text-foreground">Keeping data current</p>
        <p>
          Run the pipeline locally and publish:
        </p>
        <code className="block mt-2 font-mono text-xs bg-muted px-3 py-2 rounded leading-relaxed">
          python scripts/daily_update.py{"\n"}
          python scripts/publish_deploy_bundle.py --target railway
        </code>
        <p className="mt-2">
          Or enable the Mac cron to run automatically after market close.
          See{" "}
          <code className="font-mono text-xs bg-muted px-1 rounded">
            docs/local_cron.plist.example
          </code>
          .
        </p>
      </div>
    </div>
  );
}
