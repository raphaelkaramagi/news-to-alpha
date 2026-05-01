import { NextResponse } from "next/server";
import { forwardToFlask } from "@/lib/backend";
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
  return forwardToFlask(`${base}/api/data-status`);
}
