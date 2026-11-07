import type { ChatRequest, ChatResponse, Meta } from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly kind: "network" | "rate_limit" | "validation" | "server",
    readonly status?: number,
    readonly retryAfterSeconds?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError("Cannot reach the server", "network");
  }
  if (res.ok) return (await res.json()) as T;
  if (res.status === 429) {
    const retry = Number(res.headers.get("Retry-After") ?? "");
    throw new ApiError("Too many requests", "rate_limit", 429, Number.isFinite(retry) ? retry : undefined);
  }
  if (res.status === 422) throw new ApiError("Invalid request", "validation", 422);
  throw new ApiError("Server error", "server", res.status);
}

export function postChat(body: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

export function getMeta(): Promise<Meta> {
  return request<Meta>("/api/meta");
}
