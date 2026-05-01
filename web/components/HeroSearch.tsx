"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { TICKERS } from "@/lib/tickers";

export function HeroSearch() {
  const router = useRouter();
  const [q, setQ] = useState("");

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const sym = q.trim().toUpperCase();
    if (TICKERS.includes(sym)) {
      router.push(`/ticker/${sym}`);
    }
  }

  return (
    <form
      onSubmit={onSubmit}
      className="mx-auto flex max-w-md gap-2"
      role="search"
    >
      <input
        type="search"
        name="q"
        value={q}
        onChange={(e) => setQ(e.target.value.toUpperCase())}
        placeholder="Ticker e.g. NVDA"
        className="flex-1 rounded-xl border border-border bg-surface px-4 py-3 text-foreground placeholder:text-muted outline-none focus:border-accent"
        autoComplete="off"
        aria-label="Search ticker"
      />
      <button
        type="submit"
        className="rounded-xl bg-surface-2 px-5 py-3 text-sm font-semibold text-foreground ring-1 ring-border transition hover:ring-accent"
      >
        Go
      </button>
    </form>
  );
}
