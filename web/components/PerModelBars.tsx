import type { TickerApiResponse } from "@/lib/types";

const MODEL_LABEL: Record<string, string> = {
  ensemble: "Ensemble",
  lstm: "LSTM / price",
  tfidf: "News TF-IDF",
  embeddings: "News embeddings",
};

const BAR_COLORS = [
  "bg-accent",
  "bg-violet-400",
  "bg-cyan-400",
  "bg-rose-400",
];

export function PerModelBars({ data }: { data: TickerApiResponse }) {
  const entries = Object.entries(data.per_model);
  if (entries.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-muted">Signal breakdown</h3>
      <ul className="space-y-3">
        {entries.map(([key, m], i) => {
          const pct = Math.round(m.proba * 1000) / 10;
          const label = MODEL_LABEL[key] ?? key;
          const bar = BAR_COLORS[i % BAR_COLORS.length];
          return (
            <li key={key}>
              <div className="mb-1 flex justify-between text-xs">
                <span className="font-medium text-foreground">{label}</span>
                <span className="text-muted">
                  {pct}% up · {m.binary === 1 ? "Up" : "Down"}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                <div
                  className={`h-full rounded-full transition-[width] duration-500 ${bar}`}
                  style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
