import { describe, expect, it } from "vitest";
import { extent, formatNumber, niceTicks, scaleLinear } from "./chartScale";

describe("niceTicks", () => {
  it("returns round steps covering the range", () => {
    const t = niceTicks(0, 430, 5);
    expect(t[0]).toBeLessThanOrEqual(0);
    expect(t[t.length - 1]).toBeGreaterThanOrEqual(430);
    expect(t.slice(1).map((v, i) => v - t[i]).every((d) => d === t[1] - t[0])).toBe(true);
  });
  it("handles a flat range and non-finite input", () => {
    expect(niceTicks(5, 5)).toEqual([4, 5, 6]);
    expect(niceTicks(NaN, 1)).toEqual([0, 1]);
  });
});

describe("scaleLinear / extent / formatNumber", () => {
  it("maps domain to range (inverted y works)", () => {
    const y = scaleLinear({ min: 0, max: 100 }, [200, 0]);
    expect(y(0)).toBe(200);
    expect(y(100)).toBe(0);
    expect(y(50)).toBe(100);
  });
  it("includes zero for bars and tolerates empty data", () => {
    expect(extent([3, 5], true)).toEqual({ min: 0, max: 5 });
    expect(extent([])).toEqual({ min: 0, max: 1 });
  });
  it("formats compactly", () => {
    expect(formatNumber(1234.5)).toBe("1235");
    expect(formatNumber(12.0)).toBe("12");
    expect(formatNumber(0.5)).toBe("0.5");
  });
});
