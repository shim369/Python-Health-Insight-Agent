import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { Finding } from "../api/types";
import { FindingCard } from "./FindingCard";

afterEach(cleanup);

const correlation: Finding = {
  kind: "correlation",
  title: "遅い食事(就寝3時間以内) × 睡眠時間",
  verdict_text: "強い負の相関: 「遅い食事(就寝3時間以内)」の日は「睡眠時間」が短い傾向",
  tone: "negative",
  strength: 0.75,
  r: -0.754,
  n: 85,
  reliability: "ok",
  reliability_text: "信頼度: 高",
};

describe("FindingCard", () => {
  it("shows the plain-language verdict, r, n and reliability instead of raw codes", () => {
    render(<FindingCard finding={correlation} locale="ja" />);
    expect(screen.getByText(correlation.verdict_text!)).toBeInTheDocument();
    expect(screen.getByText("-0.75")).toBeInTheDocument();
    expect(screen.getByText("85")).toBeInTheDocument();
    expect(screen.getByText("信頼度: 高")).toBeInTheDocument();
    expect(screen.queryByText(/strong_negative|prev_day/)).not.toBeInTheDocument();
    expect(screen.getByText("相関や差は因果を意味しません")).toBeInTheDocument();
  });

  it("draws a meter whose accessible name carries r, filled on the negative side", () => {
    const { container } = render(<FindingCard finding={correlation} locale="en" />);
    expect(screen.getByRole("img", { name: "Correlation r = -0.75" })).toBeInTheDocument();
    const fill = container.querySelector(".meter-fill") as HTMLElement;
    expect(fill.className).toContain("tone-negative");
    expect(parseFloat(fill.style.width)).toBeCloseTo(37.7, 0); // |r|*50%
    expect(parseFloat(fill.style.left)).toBeCloseTo(12.3, 0); // 負は中央(50%)から左へ伸びる
  });

  it("has no fill when the relationship cannot be judged", () => {
    const f: Finding = { ...correlation, tone: "unknown", r: null, strength: null, verdict_text: "判断できません" };
    const { container } = render(<FindingCard finding={f} locale="ja" />);
    expect(container.querySelector(".meter-fill")).toBeNull();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("renders a group comparison table with units and verdict chips", () => {
    const f: Finding = {
      kind: "comparison",
      title: "「遅い食事」の日 vs それ以外の日",
      n_a: 30,
      n_b: 54,
      rows: [
        { label: "睡眠時間", mean_a: 6.2, mean_b: 7.2, n_a: 30, n_b: 54, unit: "時間", verdict_text: "グループAの方が短い", tone: "negative" },
        { label: "朝食カロリー", mean_a: 400, mean_b: 410, n_a: 30, n_b: 54, unit: "kcal", verdict_text: "差は見られません", tone: "none" },
      ],
    };
    render(<FindingCard finding={f} locale="ja" />);
    const row = screen.getByRole("row", { name: /睡眠時間/ });
    expect(within(row).getByText("6.2 時間")).toBeInTheDocument();
    expect(within(row).getByText("7.2 時間")).toBeInTheDocument();
    expect(within(row).getByText("グループAの方が短い").className).toContain("tone-negative");
    expect(screen.getByText("差は見られません")).toBeInTheDocument();
    expect(screen.getByText("30 日")).toBeInTheDocument();
  });
});
