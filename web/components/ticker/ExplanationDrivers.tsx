"use client";
import { cn } from "@/lib/utils";
import type { EnsembleExplanation } from "@/lib/types";

type Driver = EnsembleExplanation["drivers"][number];

const BAR_UP = "#22c55e";
const BAR_DOWN = "#ef4444";
const BAR_NEUTRAL = "#d4d4d8";

function DriverRow({ driver, maxAbs }: { driver: Driver; maxAbs: number }) {
  const neutral = driver.direction === "neutral";
  const isUp = driver.direction === "up";
  const barColor = neutral ? BAR_NEUTRAL : isUp ? BAR_UP : BAR_DOWN;
  const pts = Math.round(driver.effect * 100);
  const pct = neutral ? 6 : Math.min(100, Math.max(10, (Math.abs(driver.effect) / maxAbs) * 100));

  return (
    <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 items-center text-sm">
      <span className="text-muted-foreground truncate">{driver.label}</span>
      <span
        className={cn(
          "text-[11px] font-mono tabular-nums shrink-0",
          neutral ? "text-muted-foreground" : isUp ? "text-up" : "text-down"
        )}
      >
        {neutral ? "0 pts" : `${pts >= 0 ? "+" : ""}${pts} pts`}
      </span>
      <div className="col-span-2 h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={cn("h-full rounded-full", neutral && "opacity-60")}
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}

function Anchor({ label, proba }: { label: string; proba: number }) {
  const pct = Math.round(proba * 100);
  const up = pct >= 50;
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="font-medium">{label}</span>
      <span className={cn("font-semibold tabular-nums", up ? "text-up" : "text-down")}>
        {pct}% UP ({up ? "UP" : "DOWN"})
      </span>
    </div>
  );
}

interface Props {
  drivers: Driver[];
  baselineProba?: number | null;
  finalProba?: number | null;
  newsNote?: string | null;
  routeNote?: string | null;
}

export function ExplanationDrivers({
  drivers,
  baselineProba,
  finalProba,
  newsNote,
  routeNote,
}: Props) {
  if (drivers.length === 0 && !newsNote && !routeNote) {
    return (
      <p className="text-sm text-muted-foreground">
        Signal breakdown unavailable for this date.
      </p>
    );
  }

  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.effect)), 0.01);

  return (
    <div className="space-y-3">
      {routeNote && (
        <p className="text-xs text-muted-foreground border-l-2 border-muted pl-3">{routeNote}</p>
      )}
      {drivers.length > 0 && (
        <>
          <p className="text-xs text-muted-foreground">
            Shapley contributions (probability points). The combiner starts from a
            typical-call baseline; each signal&apos;s bar adds to it, summing to the
            final call. So the bars reconcile with the call direction.
          </p>
          <div className="space-y-3 rounded-md border p-3">
            {baselineProba != null && (
              <>
                <Anchor label="Typical call (this setup)" proba={baselineProba} />
                <div className="border-t" />
              </>
            )}
            {drivers.map((d) => (
              <DriverRow key={d.feature} driver={d} maxAbs={maxAbs} />
            ))}
            {finalProba != null && (
              <>
                <div className="border-t" />
                <Anchor label="Final call" proba={finalProba} />
              </>
            )}
          </div>
        </>
      )}
      {newsNote && (
        <p className="text-xs text-muted-foreground border-l-2 border-muted pl-3">{newsNote}</p>
      )}
    </div>
  );
}
