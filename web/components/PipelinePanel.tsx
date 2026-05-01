"use client";

import { useCallback, useEffect, useState } from "react";
import type { JobsResponse } from "@/lib/types";

export function PipelinePanel() {
  const [jobs, setJobs] = useState<JobsResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [preset, setPreset] = useState("balanced");
  const [refreshMode, setRefreshMode] = useState<"quality" | "fast">("quality");

  const pollJobs = useCallback(async () => {
    try {
      const res = await fetch("/api/jobs", { cache: "no-store" });
      const json = (await res.json()) as JobsResponse;
      if (res.ok) setJobs(json);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    void pollJobs();
  }, [pollJobs]);

  useEffect(() => {
    const id = setInterval(() => {
      if (jobs?.current?.status === "running" || jobs?.current?.status === "pending") {
        void pollJobs();
      }
    }, 2500);
    return () => clearInterval(id);
  }, [jobs?.current?.status, pollJobs]);

  async function runRefresh() {
    setBusy(true);
    setMessage(null);
    try {
      const res = await fetch("/api/data/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          days: 30,
          include_news: true,
          mode: refreshMode,
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setMessage((json as { error?: string }).error ?? `HTTP ${res.status}`);
      } else {
        setMessage(
          (json as { accepted?: boolean }).accepted
            ? "Refresh job accepted."
            : "A job is already running."
        );
      }
      await pollJobs();
    } catch {
      setMessage("Network error");
    } finally {
      setBusy(false);
    }
  }

  async function runPipeline() {
    setBusy(true);
    setMessage(null);
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset, config: {} }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setMessage((json as { error?: string }).error ?? `HTTP ${res.status}`);
      } else {
        setMessage(
          (json as { accepted?: boolean }).accepted
            ? "Pipeline job accepted."
            : "A job is already running."
        );
      }
      await pollJobs();
    } catch {
      setMessage("Network error");
    } finally {
      setBusy(false);
    }
  }

  const current = jobs?.current;

  return (
    <div className="mt-8 space-y-8">
      {message && (
        <p className="rounded-lg border border-border bg-surface px-4 py-3 text-sm text-muted">
          {message}
        </p>
      )}

      <div className="rounded-2xl border border-border bg-surface p-6">
        <h2 className="text-sm font-semibold text-foreground">Fast refresh</h2>
        <p className="mt-1 text-xs text-muted">
          Collect recent prices/news and rebuild ensemble CSV (no full retrain).
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-muted">
            Mode
            <select
              value={refreshMode}
              onChange={(e) =>
                setRefreshMode(e.target.value as "quality" | "fast")
              }
              className="rounded-lg border border-border bg-surface-2 px-3 py-2 text-foreground"
            >
              <option value="quality">Quality</option>
              <option value="fast">Fast</option>
            </select>
          </label>
          <button
            type="button"
            disabled={busy}
            onClick={() => void runRefresh()}
            className="rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-50"
          >
            Run data refresh
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-surface p-6">
        <h2 className="text-sm font-semibold text-foreground">Full pipeline</h2>
        <p className="mt-1 text-xs text-muted">
          Triggers <code className="font-mono text-[0.8rem]">scripts/run_pipeline.py</code>{" "}
          on the server (long-running).
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-muted">
            Preset
            <input
              value={preset}
              onChange={(e) => setPreset(e.target.value)}
              className="rounded-lg border border-border bg-surface-2 px-3 py-2 text-foreground"
              placeholder="balanced"
            />
          </label>
          <button
            type="button"
            disabled={busy}
            onClick={() => void runPipeline()}
            className="rounded-xl border border-border bg-surface-2 px-5 py-2.5 text-sm font-semibold text-foreground disabled:opacity-50"
          >
            Run pipeline
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-surface p-6">
        <h2 className="text-sm font-semibold text-foreground">Job status</h2>
        {!current ? (
          <p className="mt-3 text-sm text-muted">No job running.</p>
        ) : (
          <div className="mt-3 space-y-2 text-sm">
            <p>
              <span className="font-medium text-foreground">{current.label}</span>{" "}
              <span className="text-muted">({current.status})</span>
            </p>
            {current.progress && (
              <p className="text-xs text-muted">{current.progress}</p>
            )}
            {current.error && <p className="text-xs text-down">{current.error}</p>}
            {current.log.length > 0 && (
              <pre className="mt-2 max-h-48 overflow-auto rounded-lg bg-background p-3 text-[0.7rem] leading-relaxed text-muted">
                {current.log.slice(-40).join("\n")}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
