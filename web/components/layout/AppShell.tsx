import Link from "next/link";
import { fetchDataStatusServer } from "@/lib/data";

export const dynamic = "force-dynamic";
import { FreshnessBadge } from "@/components/markets/FreshnessBadge";
import { SelectedDateProvider } from "@/components/layout/SelectedDateProvider";
import { GlobalDateNav } from "@/components/layout/GlobalDateNav";

export async function AppShell({ children }: { children: React.ReactNode }) {
  const status = await fetchDataStatusServer().catch(() => null);
  const latest =
    status?.primary_prediction_date ??
    status?.expected_latest_prediction_date ??
    status?.latest_price_date ??
    status?.latest_prediction_date ??
    null;

  return (
    <SelectedDateProvider initialLatest={latest}>
      <div className="flex flex-col min-h-screen">
        <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
          <div className="max-w-5xl mx-auto px-3 sm:px-4 h-14 flex items-center gap-3 sm:gap-4 min-w-0">
            <Link
              href="/"
              className="font-semibold tracking-tight text-sm shrink-0"
              title="Stock Price and Sentiment Predictor"
            >
              Stock Predictor
            </Link>
            <nav className="flex items-center gap-3 sm:gap-4 text-sm text-muted-foreground shrink-0">
              <Link href="/" className="hover:text-foreground transition-colors">
                Markets
              </Link>
              <Link href="/status" className="hover:text-foreground transition-colors">
                Status
              </Link>
            </nav>
            <div className="ml-auto flex items-center gap-2 sm:gap-4 min-w-0">
              <GlobalDateNav />
              {status && (
                <span className="hidden sm:inline-flex shrink-0">
                  <FreshnessBadge status={status} />
                </span>
              )}
            </div>
          </div>
        </header>
        <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-8">
          {children}
        </main>
      </div>
    </SelectedDateProvider>
  );
}
