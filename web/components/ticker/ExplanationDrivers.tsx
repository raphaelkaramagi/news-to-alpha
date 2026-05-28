"use client";
import { cn } from "@/lib/utils";
import type { EnsembleExplanation } from "@/lib/types";

type Driver = EnsembleExplanation["drivers"][number];

const BAR_UP = "#22c55e";
const BAR_DOWN = "#ef4444";
const BAR_NEUTRAL = "#d4d4d8";

function DriverRow({ driver }: { driver: Driver }) {
  const neutral = driver.direction === "neutral";
  const isUp = driver.direction === "up";
  const barColor = neutral ? BAR_NEUTRAL : isUp ? BAR_UP : BAR_DOWN;
  const pct = neutral ? 8 : Math.min(100, Math.max(14, Math.abs(driver.effect) * 400));
  const tag = neutral ? "No impact" : isUp ? "Pushed UP" : "Pushed DOWN";

  return (
    <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 items-center text-sm">
      <span className="text-muted-foreground truncate">{driver.label}</span>
      <span
        className={cn(
          "text-[11px] font-medium shrink-0",
          neutral ? "text-muted-foreground" : isUp ? "text-up" : "text-down"
        )}
      >
        {tag}
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

interface Props {
  drivers: Driver[];
  newsNote?: string | null;
  routeNote?: string | null;
}

export function ExplanationDrivers({ drivers, newsNote, routeNote }: Props) {
  if (drivers.length === 0 && !newsNote && !routeNote) {
    return (
      <p className="text-sm text-muted-foreground">
        Counterfactual breakdown unavailable — restart Flask after retraining.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {routeNote && (
        <p className="text-xs text-muted-foreground border-l-2 border-muted pl-3">{routeNote}</p>
      )}
      {drivers.length > 0 && (
        <>
          <p className="text-sm font-medium">What moved the final score</p>
          <p className="text-xs text-muted-foreground -mt-1">
            Counterfactual test: if we neutralize each factor, how much does the ensemble probability shift?
          </p>
          <div className="space-y-3">
            {drivers.map((d) => (
              <DriverRow key={d.feature} driver={d} />
            ))}
          </div>
        </>
      )}
      {newsNote && (
        <p className="text-xs text-muted-foreground border-l-2 border-muted pl-3">{newsNote}</p>
      )}
    </div>
  );
}
