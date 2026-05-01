import { NextRequest, NextResponse } from "next/server";
import { forwardToFlask } from "@/lib/backend";
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
  return forwardToFlask(`${base}/api/data/refresh`, {
    method: "POST",
    headers: {
      "content-type":
        req.headers.get("content-type") || "application/json; charset=utf-8",
    },
    body: body || "{}",
  });
}
