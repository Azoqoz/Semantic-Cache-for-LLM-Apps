import type { Capabilities, CacheSnapshot, QueryRequest, QueryResult, EvaluationReport } from "./types";

export class ApiError extends Error {
  constructor(public code: string, public status: number, message: string) { super(message); }
}

const messages: Record<string, string> = {
  invalid_request: "Check your prompt and settings, then try again.",
  provider_configuration: "This provider is disabled or needs configuration on the backend. Try Demo, or check the server environment.",
  operation_failed: "The backend could not finish this operation. Check the model or provider configuration before retrying.",
  internal_error: "The service encountered an error. Please try again.",
  offline: "The API is unreachable. Start the backend or check its configured address, then reconnect.",
};

async function request<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/backend${path}`, {
      method, cache: "no-store", credentials: "omit",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(185_000),
    });
  } catch {
    throw new ApiError("offline", 0, "The request did not complete. It may still finish on the server; refresh the ledger before retrying.");
  }
  let data: unknown;
  try { data = await response.json(); }
  catch { throw new ApiError("offline", response.status, messages.offline); }
  if (!response.ok) {
    const code = (data as { error?: { code?: string } })?.error?.code ?? "internal_error";
    throw new ApiError(code, response.status, messages[code] ?? messages.internal_error);
  }
  return data as T;
}

export const api = {
  health: () => request<{ status: string; check: string }>("/health"),
  capabilities: () => request<Capabilities>("/capabilities"),
  query: (body: QueryRequest) => request<QueryResult>("/query", "POST", body),
  cache: () => request<CacheSnapshot>("/cache?limit=1000"),
  clear: () => request<{ cleared: boolean }>("/cache", "DELETE"),
  evaluate: (threshold: number) => request<EvaluationReport>("/evaluation", "POST", { threshold }),
};
export function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : messages.internal_error;
}
