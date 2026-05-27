import { NextRequest, NextResponse } from "next/server";
import { backendUrl, forwardToFlask } from "@/lib/backend";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  // Flask serves GET /healthz (not /api/healthz). This route is the Vercel-side proxy.
  const target = backendUrl("/healthz", req.nextUrl.searchParams);
  if (!target)
    return NextResponse.json({ error: "API_BASE_URL not configured" }, { status: 503 });
  return forwardToFlask(target);
}
