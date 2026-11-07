import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ChatResponse } from "../api/types";
import { ChatView } from "./ChatView";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const response: ChatResponse = {
  answer: "遅い食事の夜は睡眠が短い傾向です。",
  status: "ok",
  tool_trace: [
    {
      name: "compare_days",
      arguments: { start_date: "2026-08-01" },
      ok: true,
      rows_used: 84,
      duration_ms: 12,
      result: { ok: true },
    },
  ],
  charts: [
    {
      kind: "bar",
      title: "睡眠時間(分)",
      y_label: "分",
      series: [
        {
          name: "分",
          points: [
            { x: "条件に該当 n=30", y: 360 },
            { x: "それ以外 n=54", y: 420 },
          ],
        },
      ],
    },
    {
      kind: "value",
      title: "明日の睡眠(予測)",
      series: [],
      value: 6.8,
      unit: "時間",
      label: "モデルによる参考値",
    },
  ],
  usage: {
    input_tokens: 100,
    output_tokens: 50,
    total_tokens: 150,
    llm_calls: 2,
    tool_calls: 1,
    duration_ms: 800,
    model: "fake-rule",
  },
};

function stubFetch(impl: () => Promise<Partial<Response>>) {
  const f = vi.fn(impl);
  vi.stubGlobal("fetch", f);
  return f;
}

describe("ChatView", () => {
  it("sends a question and shows answer, charts, reference label, tools and usage", async () => {
    const f = stubFetch(async () => ({ ok: true, status: 200, json: async () => response }));
    render(<ChatView locale="ja" />);
    await userEvent.type(screen.getByLabelText(/質問を入力/), "睡眠について");
    await userEvent.click(screen.getByRole("button", { name: "送信" }));

    expect(await screen.findByText(response.answer)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "睡眠時間(分)" })).toBeInTheDocument();
    expect(screen.getByText("モデルによる参考値")).toBeInTheDocument();
    expect(screen.getByText(/使ったツールと結果 \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/150 トークン/)).toBeInTheDocument();
    const sent = JSON.parse((f.mock.calls[0] as unknown as [string, RequestInit])[1].body as string);
    expect(sent).toMatchObject({ question: "睡眠について", locale: "ja", history: [] });
  });

  it("shows a loading indicator and disables input while waiting", async () => {
    let resolve!: (r: Partial<Response>) => void;
    stubFetch(() => new Promise((r) => (resolve = r)));
    render(<ChatView locale="ja" />);
    await userEvent.type(screen.getByLabelText(/質問を入力/), "q");
    await userEvent.click(screen.getByRole("button", { name: "送信" }));
    expect(screen.getByRole("status")).toHaveTextContent("データを調べています");
    expect(screen.getByLabelText(/質問を入力/)).toBeDisabled();
    resolve({ ok: true, status: 200, json: async () => response });
    await screen.findByText(response.answer);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("shows an error with a retry button that resends the same question", async () => {
    const f = stubFetch(async () => {
      throw new TypeError("Failed to fetch");
    });
    render(<ChatView locale="ja" />);
    await userEvent.type(screen.getByLabelText(/質問を入力/), "再送テスト");
    await userEvent.click(screen.getByRole("button", { name: "送信" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("サーバーに接続できませんでした");

    f.mockImplementation(async () => ({ ok: true, status: 200, json: async () => response }));
    await userEvent.click(screen.getByRole("button", { name: "もう一度送る" }));
    await waitFor(() => expect(screen.getByText(response.answer)).toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(JSON.parse((f.mock.calls[1] as unknown as [string, RequestInit])[1].body as string).question).toBe("再送テスト");
  });

  it("sends previous Q&A as history on the next question", async () => {
    const f = stubFetch(async () => ({ ok: true, status: 200, json: async () => response }));
    render(<ChatView locale="en" />);
    const box = screen.getByRole("textbox");
    await userEvent.type(box, "first");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText(response.answer);
    await userEvent.type(box, "second");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(f).toHaveBeenCalledTimes(2));
    const second = JSON.parse((f.mock.calls[1] as unknown as [string, RequestInit])[1].body as string);
    expect(second.history).toEqual([{ question: "first", answer: response.answer }]);
  });

  it("suggested questions send immediately", async () => {
    const f = stubFetch(async () => ({ ok: true, status: 200, json: async () => response }));
    render(<ChatView locale="ja" />);
    await userEvent.click(screen.getByRole("button", { name: "先週、睡眠が悪かった日の共通点は?" }));
    await waitFor(() => expect(f).toHaveBeenCalledTimes(1));
  });
});
