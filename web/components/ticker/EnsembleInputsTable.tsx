"use client";
import type { MetaFeatureRow, LstmContextField } from "@/lib/types";

function formatValue(key: string, value: number): string {
  if (key.endsWith("_proba") || (key.includes("confidence") && value <= 1 && value >= 0)) {
    if (key.endsWith("_proba")) return `${(value * 100).toFixed(1)}%`;
    return `${(value * 100).toFixed(0)}%`;
  }
  if (key === "has_news" || key === "all_agree") return value >= 0.5 ? "Yes" : "No";
  if (key === "n_headlines") return String(Math.round(value));
  if (key.includes("return") || key.includes("spy")) return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
  return value.toFixed(3);
}

interface Props {
  features: MetaFeatureRow[];
}

export function EnsembleInputsTable({ features }: Props) {
  if (!features.length) {
    return (
      <p className="text-sm text-muted-foreground">
        Ensemble feature values not available for this date.
      </p>
    );
  }

  return (
    <div className="rounded-lg border overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/40 text-left text-xs text-muted-foreground">
            <th className="px-3 py-2 font-medium">Input</th>
            <th className="px-3 py-2 font-medium text-right">Value</th>
          </tr>
        </thead>
        <tbody>
          {features.map((f) => (
            <tr key={f.key} className="border-b last:border-0">
              <td className="px-3 py-2">
                <span className="font-medium">{f.label}</span>
                {f.hint && (
                  <p className="text-[10px] text-muted-foreground mt-0.5 leading-snug">{f.hint}</p>
                )}
              </td>
              <td className="px-3 py-2 text-right tabular-nums font-mono text-xs">
                {formatValue(f.key, f.value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-3 py-2 text-[10px] text-muted-foreground border-t">
        Raw RSI, MACD, and VIX are LSTM inputs only — summarized in the price P(UP) row above.
      </p>
    </div>
  );
}

interface LstmProps {
  context: {
    available: boolean;
    note?: string;
    fields: LstmContextField[];
  } | null;
}

function formatLstmValue(field: LstmContextField): string {
  const v = field.value;
  if (field.unit === "%") return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  if (field.unit === "0–1") return v.toFixed(2);
  if (field.key === "vix_level") return v.toFixed(1);
  return v.toFixed(3);
}

export function LstmContextTable({ context }: LstmProps) {
  if (!context?.available || !context.fields.length) {
    return null;
  }

  return (
    <div>
      <p className="text-sm font-medium mb-2">Price model context (same day)</p>
      {context.note && (
        <p className="text-xs text-muted-foreground mb-2">{context.note}</p>
      )}
      <div className="rounded-lg border overflow-hidden">
        <table className="w-full text-sm">
          <tbody>
            {context.fields.map((f) => (
              <tr key={f.key} className="border-b last:border-0">
                <td className="px-3 py-2 text-muted-foreground">{f.label}</td>
                <td className="px-3 py-2 text-right tabular-nums font-mono text-xs">
                  {formatLstmValue(f)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
