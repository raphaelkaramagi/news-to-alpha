import Link from "next/link";
import { fetchDataStatusServer } from "@/lib/data";
import { backendConfigured } from "@/lib/backend";
import { TICKERS } from "@/lib/tickers";
import { TickerGrid } from "@/components/TickerGrid";
import { HeroSearch } from "@/components/HeroSearch";

export default async function HomePage() {
  const configured = backendConfigured();
  const status = configured ? await fetchDataStatusServer() : null;

  return (
    <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6 sm:py-14">
      <section className="text-center">
        <TickerGrid compact active={undefined} />
        <h1 className="mt-8 text-balance text-4xl font-bold tracking-tight sm:text-5xl">
          US equities —{" "}
          <span className="text-accent">next session</span>
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-balance text-muted">
          Ensemble of price and news models: pick a ticker for direction,
          confidence, and the headlines behind the latest call.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <span className="rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-muted">
            {TICKERS.length} tickers
          </span>
          <span className="rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-muted">
            4 model tracks
          </span>
          {status?.latest_prediction_date && (
            <span className="rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-muted">
              Latest pred {status.latest_prediction_date}
            </span>
          )}
        </div>
        <div className="mt-8">
          <HeroSearch />
        </div>
        <Link
          href="/ticker/AAPL"
          className="mt-6 inline-flex rounded-xl bg-accent px-8 py-3 text-sm font-semibold text-white shadow-lg transition hover:opacity-90"
        >
          View sample ticker →
        </Link>
      </section>

      {!configured && (
        <p className="mx-auto mt-12 max-w-lg rounded-xl border border-warn/40 bg-warn/10 p-4 text-center text-sm text-warn">
          Set <code className="rounded bg-surface px-1 font-mono text-xs">API_BASE_URL</code> on
          Vercel (and locally in <code className="font-mono text-xs">web/.env.local</code>) to your
          Flask API origin. The UI proxies{' '}
          <code className="font-mono text-xs">/api/ticker</code> and job routes server-side.
        </p>
      )}

      <section className="mt-16 grid gap-4 sm:grid-cols-3">
        <div className="rounded-2xl border border-border bg-surface p-5 text-sm text-muted">
          <h2 className="font-semibold text-foreground">Data</h2>
          <p className="mt-2">
            Prices and Finnhub headlines feed labels and features; refresh jobs
            run on the Python host.
          </p>
        </div>
        <div className="rounded-2xl border border-border bg-surface p-5 text-sm text-muted">
          <h2 className="font-semibold text-foreground">News models</h2>
          <p className="mt-2">
            TF-IDF and embedding classifiers; compare bars when embeddings and
            TF-IDF diverge from LSTM.
          </p>
        </div>
        <div className="rounded-2xl border border-border bg-surface p-5 text-sm text-muted">
          <h2 className="font-semibold text-foreground">Ensemble</h2>
          <p className="mt-2">
            Blended call with calibrated confidence — use Pipeline to refresh
            data or rerun training.
          </p>
        </div>
      </section>
    </main>
  );
}
