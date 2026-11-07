// グラフ用の純粋関数(目盛り・スケール)。SVG描画から分離してテストしやすくする。

export interface Domain {
  min: number;
  max: number;
}

/** 値の範囲に「きりのよい」目盛りを返す(例: 0,100,200,300)。 */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, min, min + 1];
  const rawStep = (max - min) / Math.max(1, count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const start = Math.floor(min / step) * step;
  const ticks: number[] = [];
  // 最後の目盛りが必ず最大値以上になるまで進める(上限は無限ループ防止)。
  for (let i = 0, v = start; i < 100; i++, v += step) {
    ticks.push(Number(v.toPrecision(12)));
    if (v >= max) break;
  }
  return ticks;
}

/** 値→ピクセル座標の線形変換。 */
export function scaleLinear(domain: Domain, range: [number, number]): (v: number) => number {
  const span = domain.max - domain.min || 1;
  return (v) => range[0] + ((v - domain.min) / span) * (range[1] - range[0]);
}

export function extent(values: number[], includeZero = false): Domain {
  if (values.length === 0) return { min: 0, max: 1 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (includeZero) {
    min = Math.min(min, 0);
    max = Math.max(max, 0);
  }
  return { min, max };
}

export function formatNumber(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1).replace(/\.0$/, "");
  return v.toFixed(2).replace(/\.?0+$/, "");
}
