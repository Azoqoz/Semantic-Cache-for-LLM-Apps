import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 300;
const allowed: Record<string, string[]> = {
  health: ["GET"], ready: ["GET"], capabilities: ["GET"], query: ["POST"], cache: ["GET", "DELETE"], evaluation: ["POST"],
};
const safeError = (status: number, code: string) => Response.json({ error: { code } }, { status });

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  if (path.length !== 1 || !allowed[path[0]]?.includes(request.method)) return safeError(404, "http_error");
  try {
    const base = new URL(process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000");
    if (!["http:", "https:"].includes(base.protocol) || base.username || base.password || base.search || base.hash) {
      return safeError(503, "internal_error");
    }
    const target = new URL(`${base.toString().replace(/\/$/, "")}/${path[0]}`);
    if (path[0] === "cache" && request.method === "GET") {
      target.searchParams.set("limit", request.nextUrl.searchParams.get("limit") || "1000");
    }
    const response = await fetch(target, {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body: request.method === "POST" ? await request.text() : undefined,
      cache: "no-store", redirect: "error", signal: AbortSignal.timeout(180_000),
    });
    const payload = await response.json();
    if (!response.ok) return safeError(response.status, payload?.error?.code || "internal_error");
    return Response.json(payload, { headers: { "Cache-Control": "no-store" } });
  } catch {
    // Never log upstream exceptions, request content, headers, or environment data.
    return safeError(502, "offline");
  }
}
export { proxy as GET, proxy as POST, proxy as DELETE };
