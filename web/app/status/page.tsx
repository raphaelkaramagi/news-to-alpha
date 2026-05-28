import { fetchDataStatusServer } from "@/lib/data";
import { getApiBaseUrl } from "@/lib/env";
import type { DataStatus, MetricsResponse } from "@/lib/types";
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

async function fetchMetricsServer(): Promise<MetricsResponse | null> {
  const base = getApiBaseUrl();
  if (!base) return null;
  try {
    const res = await fetch(`${base}/api/metrics`, {
      cache: "no-store",
      signal: AbortSignal.timeout(8_000),
    });
    if (!res.ok) return null;
    return (await res.json()) as MetricsResponse;
  } catch {
    return null;
  }
}

function BenchRow({
  label,
  accuracy,
  auc,
  n,
}: {
  label: string;
  accuracy: number;
  auc: number;
  n: number;
}) {
  return (
    <div className="flex items-baseline justify-between py-2 border-b last:border-0 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums font-medium">
        {(accuracy * 100).toFixed(1)}% acc · AUC {(auc * 100).toFixed(0)}% · n={n.toLocaleString()}
      </span>
    </div>
  );
}

export default async function StatusPage() {
  const [status, metrics] = await Promise.all([
    fetchDataStatusServer().catch(() => null),
    fetchMetricsServer(),
  ]);

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

  const ensembleRows =
    metrics?.overall?.filter(
      (r) => r.model === "ensemble" && r.split === "test"
    ) ?? [];
  const benchBySubset = Object.fromEntries(
    ensembleRows.map((r) => [r.subset, r])
  );
  const cfg = status.train_config;

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

      {benchBySubset.all && (
        <div className="rounded-lg border">
          <div className="px-4 py-3 border-b">
            <p className="text-sm font-medium">Backtest (held-out test split)</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              Offline accuracy from last train — not live trading performance.
            </p>
          </div>
          <div className="px-4 py-1">
            <BenchRow
              label="All sessions"
              accuracy={benchBySubset.all.accuracy}
              auc={benchBySubset.all.auc}
              n={benchBySubset.all.n}
            />
            {benchBySubset.has_news && (
              <BenchRow
                label="Days with headlines"
                accuracy={benchBySubset.has_news.accuracy}
                auc={benchBySubset.has_news.auc}
                n={benchBySubset.has_news.n}
              />
            )}
            {benchBySubset.high_conf && (
              <BenchRow
                label="High-confidence calls"
                accuracy={benchBySubset.high_conf.accuracy}
                auc={benchBySubset.high_conf.auc}
                n={benchBySubset.high_conf.n}
              />
            )}
          </div>
        </div>
      )}

      {cfg && (
        <div className="rounded-lg border">
          <div className="px-4 py-3 border-b">
            <p className="text-sm font-medium">Train config (local artifact)</p>
          </div>
          <div className="px-4">
            <Row label="News encoder" value={cfg.encoder_model ?? "—"} mono />
            <Row
              label="Conditional ensemble"
              value={cfg.conditional_ensemble ? "yes" : "no"}
            />
            <Row
              label="Min move filter"
              value={cfg.min_move_pct != null ? `${cfg.min_move_pct}%` : "—"}
            />
            <Row
              label="LSTM epochs"
              value={cfg.lstm_epochs != null ? String(cfg.lstm_epochs) : "—"}
            />
          </div>
        </div>
      )}

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
