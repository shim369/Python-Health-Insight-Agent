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
