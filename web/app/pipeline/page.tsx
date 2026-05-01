import Link from "next/link";
import { PipelinePanel } from "@/components/PipelinePanel";

export const metadata = {
  title: "Pipeline · News-to-Alpha",
  description: "Refresh data and run training jobs against the Flask API.",
};

export default function PipelinePage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
      <div className="mb-8 flex flex-wrap items-center gap-4">
        <Link href="/" className="text-sm font-medium text-muted hover:text-accent">
          ← Markets
        </Link>
      </div>
      <h1 className="text-2xl font-bold tracking-tight">Pipeline</h1>
      <p className="mt-2 text-sm text-muted">
        Fast refresh (collect + rebuild ensemble) and full pipeline runs execute on
        your Flask host. This page proxies requests — no secrets in the browser.
      </p>
      <PipelinePanel />
    </div>
  );
}
