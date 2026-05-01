import { NextRequest, NextResponse } from "next/server";
import { getApiBaseUrl } from "@/lib/env";

export async function POST(req: NextRequest) {
  const base = getApiBaseUrl();
  if (!base) {
    return NextResponse.json(
      { error: "API_BASE_URL is not configured on the server" },
      { status: 503 }
    );
  }
  const body = await req.text();
  const res = await fetch(`${base}/api/run`, {
    method: "POST",
    headers: {
      "content-type":
        req.headers.get("content-type") || "application/json; charset=utf-8",
    },
    body: body || "{}",
    cache: "no-store",
  });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: {
      "content-type":
        res.headers.get("content-type") || "application/json; charset=utf-8",
    },
  });
}
