import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const reply = {
  answer: "回答です",
  status: "ok",
  tool_trace: [],
  charts: [],
  findings: [],
  usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2, llm_calls: 1, tool_calls: 0, duration_ms: 5, model: "m" },
};

async function ask() {
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200, json: async () => reply })));
  await userEvent.type(screen.getByLabelText(/質問を入力/), "質問");
  await userEvent.click(screen.getByRole("button", { name: "送信" }));
  await screen.findByText("回答です");
}

describe("back to top", () => {
  it("the 'New chat' button clears the conversation and shows the start screen again", async () => {
    render(<App />);
    await ask();
    expect(screen.queryByText("今日は何を調べますか?")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "新しい会話" }));
    expect(screen.queryByText("回答です")).not.toBeInTheDocument();
    expect(screen.getByText("今日は何を調べますか?")).toBeInTheDocument();
  });

  it("clicking the logo/title also goes back to the top, in English too", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "English" }));
    await ask2();
    await userEvent.click(screen.getByRole("button", { name: /Back to top/ }));
    expect(screen.queryByText("回答です")).not.toBeInTheDocument();
    expect(screen.getByText("What would you like to look into?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New chat" })).toBeInTheDocument();
  });
});

async function ask2() {
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200, json: async () => reply })));
  await userEvent.type(screen.getByLabelText(/Ask a question/), "q");
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  await screen.findByText("回答です");
}
