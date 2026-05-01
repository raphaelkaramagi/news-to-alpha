import { NextRequest, NextResponse } from "next/server";
import { backendUrl } from "@/lib/backend";

export async function GET(req: NextRequest) {
  const target = backendUrl("/api/ticker", req.nextUrl.searchParams);
  if (!target) {
    return NextResponse.json(
      { error: "API_BASE_URL is not configured on the server" },
      { status: 503 }
    );
  }
  const res = await fetch(target, { cache: "no-store" });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: {
      "content-type":
        res.headers.get("content-type") || "application/json; charset=utf-8",
    },
  });
}
