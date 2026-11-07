import { describe, expect, it } from "vitest";
import { splitDisclaimer, splitUnknown } from "./MessageBubble";

describe("splitUnknown", () => {
  it("moves the unknown paragraph (with its bullet list) out of the body", () => {
    const text = "結果A\nわからないこと:\n- 因果は断定できない\n- 件数が少ない\nまとめ";
    const r = splitUnknown(text);
    expect(r.unknown).toBe("わからないこと:\n- 因果は断定できない\n- 件数が少ない");
    expect(r.body).toBe("結果A\nまとめ");
  });
  it("works for English and leaves other text alone", () => {
    expect(splitUnknown("Result\nWhat is unknown: causation.").unknown).toBe(
      "What is unknown: causation.",
    );
    expect(splitUnknown("ただの回答")).toEqual({ body: "ただの回答", unknown: null });
  });
  it("uses the last matching paragraph only", () => {
    const r = splitUnknown("わからないこと: 1\n本文\nわからないこと: 2");
    expect(r.unknown).toBe("わからないこと: 2");
    expect(r.body).toContain("わからないこと: 1");
  });
});

describe("splitDisclaimer", () => {
  it("separates the fixed disclaimer", () => {
    expect(splitDisclaimer("本文\n\n※ 注意")).toEqual({ body: "本文", disclaimer: "注意" });
    expect(splitDisclaimer("本文").disclaimer).toBeNull();
  });
});

import { parseBlocks, renderInline } from "./MessageBubble";

describe("markdown-ish answers from a real LLM", () => {
  it("turns headings, rules, numbered and bullet lists into blocks", () => {
    const blocks = parseBlocks("### 1. 結果\n---\n1. 一つ目\n2. 二つ目\n* 箇条書き\n本文");
    expect(blocks.map((b) => b.kind)).toEqual(["heading", "ol", "ul", "p"]);
    expect(blocks[0]).toEqual({ kind: "heading", text: "1. 結果" });
    expect(blocks[1]).toMatchObject({ items: ["一つ目", "二つ目"] });
  });

  it("renders **bold** and `code` as elements and strips LaTeX dollars, never raw HTML", () => {
    const out = renderInline("相関は **強い** で `r` は $r = -0.67$ です <b>x</b>");
    const kinds = out.map((n) => (typeof n === "string" ? "text" : (n as { type: string }).type));
    expect(kinds).toEqual(["text", "strong", "text", "code", "text"]);
    expect(out.filter((n) => typeof n === "string").join("")).toContain("r = -0.67");
    expect(out.filter((n) => typeof n === "string").join("")).not.toContain("$");
    expect(out.filter((n) => typeof n === "string").join("")).toContain("<b>x</b>"); // タグは文字として表示
  });
});
