import { NextRequest, NextResponse } from "next/server";
import { backendUrl, forwardToFlask } from "@/lib/backend";

export async function GET(req: NextRequest) {
  const target = backendUrl("/api/ticker", req.nextUrl.searchParams);
  if (!target) {
    return NextResponse.json(
      { error: "API_BASE_URL is not configured on the server" },
      { status: 503 }
    );
  }
  return forwardToFlask(target);
}
