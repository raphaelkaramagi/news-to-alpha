import Link from "next/link";
import { fetchDataStatusServer } from "@/lib/data";

export const dynamic = "force-dynamic";
import { FreshnessBadge } from "@/components/markets/FreshnessBadge";
import { SelectedDateProvider } from "@/components/layout/SelectedDateProvider";
import { GlobalDateNav } from "@/components/layout/GlobalDateNav";

export async function AppShell({ children }: { children: React.ReactNode }) {
  const status = await fetchDataStatusServer().catch(() => null);
  const latest = status?.latest_prediction_date ?? null;

  return (
    <SelectedDateProvider initialLatest={latest}>
      <div className="flex flex-col min-h-screen">
        <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
          <div className="max-w-5xl mx-auto px-4 h-14 flex items-center gap-4">
            <Link
              href="/"
              className="font-semibold tracking-tight text-[11px] sm:text-sm leading-tight max-w-[9rem] sm:max-w-none shrink min-w-0"
              title="Stock Price and Sentiment Predictor"
            >
              Stock Price and Sentiment Predictor
            </Link>
            <nav className="flex items-center gap-4 text-sm text-muted-foreground">
              <Link href="/" className="hover:text-foreground transition-colors">
                Markets
              </Link>
              <Link href="/status" className="hover:text-foreground transition-colors">
                Status
              </Link>
            </nav>
            <div className="ml-auto flex items-center gap-4">
              <GlobalDateNav />
              {status && <FreshnessBadge status={status} />}
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
