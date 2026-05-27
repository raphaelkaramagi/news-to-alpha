"use client";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { DataStatus } from "@/lib/types";

type Ctx = {
  selectedDate: string | null;
  latestDate: string | null;
  setSelectedDate: (d: string) => void;
  goLatest: () => void;
};

const SelectedDateContext = createContext<Ctx | null>(null);

export function SelectedDateProvider({
  children,
  initialLatest,
}: {
  children: React.ReactNode;
  initialLatest: string | null;
}) {
  const [latestDate, setLatestDate] = useState(initialLatest);
  const [selectedDate, setSelectedDateState] = useState<string | null>(initialLatest);

  useEffect(() => {
    fetch("/api/data-status", { cache: "no-store" })
      .then((r) => r.json())
      .then((s: DataStatus) => {
        const latest = s.latest_prediction_date ?? null;
        setLatestDate((oldLatest) => {
          setSelectedDateState((prev) => {
            if (!prev) return latest;
            if (!latest) return prev;
            // Advance sitewide date when data was refreshed past SSR snapshot
            if (oldLatest && prev === oldLatest && latest > prev) return latest;
            return prev;
          });
          return latest;
        });
      })
      .catch(() => undefined);
  }, []);

  const setSelectedDate = useCallback((d: string) => {
    setSelectedDateState(d);
  }, []);

  const goLatest = useCallback(() => {
    if (latestDate) setSelectedDateState(latestDate);
  }, [latestDate]);

  return (
    <SelectedDateContext.Provider
      value={{ selectedDate, latestDate, setSelectedDate, goLatest }}
    >
      {children}
    </SelectedDateContext.Provider>
  );
}

export function useSelectedDate() {
  const ctx = useContext(SelectedDateContext);
  if (!ctx) {
    throw new Error("useSelectedDate must be used within SelectedDateProvider");
  }
  return ctx;
}
