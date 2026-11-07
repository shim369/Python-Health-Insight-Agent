import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, postChat } from "./client";

const body = { question: "q", locale: "ja" as const, history: [] };

afterEach(() => vi.restoreAllMocks());

function mockFetch(res: Partial<Response> | Error) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      if (res instanceof Error) throw res;
      return res as Response;
    }),
  );
}

describe("postChat", () => {
  it("returns parsed JSON on success", async () => {
    mockFetch({ ok: true, status: 200, json: async () => ({ answer: "a" }) });
    await expect(postChat(body)).resolves.toEqual({ answer: "a" });
  });
  it("maps 429 to rate_limit with Retry-After", async () => {
    mockFetch({ ok: false, status: 429, headers: new Headers({ "Retry-After": "12" }) });
    await expect(postChat(body)).rejects.toMatchObject({ kind: "rate_limit", retryAfterSeconds: 12 });
  });
  it("maps 422 / 500 / network failures", async () => {
    mockFetch({ ok: false, status: 422, headers: new Headers() });
    await expect(postChat(body)).rejects.toMatchObject({ kind: "validation" });
    mockFetch({ ok: false, status: 500, headers: new Headers() });
    await expect(postChat(body)).rejects.toMatchObject({ kind: "server", status: 500 });
    mockFetch(new TypeError("Failed to fetch"));
    const err = await postChat(body).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.kind).toBe("network");
  });
});
