import { NextResponse } from "next/server";
import { getApiBaseUrl } from "@/lib/env";

export const dynamic = "force-dynamic";

export async function GET() {
  const base = getApiBaseUrl();
  if (!base) {
    return NextResponse.json(
      { error: "API_BASE_URL is not configured on the server" },
      { status: 503 }
    );
  }
  const res = await fetch(`${base}/api/data-status`, { cache: "no-store" });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: {
      "content-type":
        res.headers.get("content-type") || "application/json; charset=utf-8",
    },
  });
}
