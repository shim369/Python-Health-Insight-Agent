import type { Chart } from "../api/types";
import { extent, formatNumber, niceTicks, scaleLinear } from "../lib/chartScale";
import { t } from "../lib/strings";
import type { Locale } from "../api/types";

const W = 480;
const H = 220;
const M = { top: 12, right: 16, bottom: 44, left: 52 };

function Axes({ yTicks, y, yLabel }: { yTicks: number[]; y: (v: number) => number; yLabel?: string | null }) {
  return (
    <g className="axes">
      {yTicks.map((tv) => (
        <g key={tv}>
          <line x1={M.left} x2={W - M.right} y1={y(tv)} y2={y(tv)} className="grid" />
          <text x={M.left - 6} y={y(tv) + 4} textAnchor="end" className="tick">
            {formatNumber(tv)}
          </text>
        </g>
      ))}
      {yLabel ? (
        <text x={12} y={H / 2} transform={`rotate(-90 12 ${H / 2})`} textAnchor="middle" className="axis-label">
          {yLabel}
        </text>
      ) : null}
    </g>
  );
}

function shortLabel(x: string | number): string {
  const s = String(x);
  return /^\d{4}-\d{2}-\d{2}/.test(s) ? s.slice(5, 10) : s;
}

function BarChart({ chart }: { chart: Chart }) {
  const pts = chart.series[0]?.points ?? [];
  const ys = pts.map((p) => p.y);
  const dom = extent(ys, true);
  const ticks = niceTicks(dom.min, dom.max, 5);
  const y = scaleLinear({ min: Math.min(dom.min, ticks[0]), max: Math.max(dom.max, ticks[ticks.length - 1]) }, [H - M.bottom, M.top]);
  const slot = (W - M.left - M.right) / Math.max(1, pts.length);
  const barW = Math.min(48, slot * 0.7);
  const showEvery = Math.ceil(pts.length / 12);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={chart.title} className="chart-svg">
      <Axes yTicks={ticks} y={y} yLabel={chart.y_label} />
      {pts.map((p, i) => {
        const cx = M.left + slot * i + slot / 2;
        return (
          <g key={i}>
            <rect x={cx - barW / 2} y={Math.min(y(p.y), y(0))} width={barW} height={Math.abs(y(0) - y(p.y))} className="bar" />
            {pts.length <= 8 ? (
              <text x={cx} y={Math.min(y(p.y), y(0)) - 4} textAnchor="middle" className="value-label">
                {formatNumber(p.y)}
              </text>
            ) : null}
            {i % showEvery === 0 ? (
              <text x={cx} y={H - M.bottom + 14} textAnchor="middle" className="tick">
                {shortLabel(p.x).slice(0, 14)}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

function LineChart({ chart }: { chart: Chart }) {
  const pts = chart.series[0]?.points ?? [];
  const ys = pts.map((p) => p.y);
  const dom = extent(ys);
  const ticks = niceTicks(dom.min, dom.max, 5);
  const y = scaleLinear({ min: Math.min(dom.min, ticks[0]), max: Math.max(dom.max, ticks[ticks.length - 1]) }, [H - M.bottom, M.top]);
  const x = scaleLinear({ min: 0, max: Math.max(1, pts.length - 1) }, [M.left, W - M.right]);
  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.y).toFixed(1)}`).join(" ");
  const showEvery = Math.ceil(pts.length / 6);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={chart.title} className="chart-svg">
      <Axes yTicks={ticks} y={y} yLabel={chart.y_label} />
      <path d={d} className="line" fill="none" />
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={x(i)} cy={y(p.y)} r={2.5} className="dot" />
          {i % showEvery === 0 ? (
            <text x={x(i)} y={H - M.bottom + 14} textAnchor="middle" className="tick">
              {shortLabel(p.x)}
            </text>
          ) : null}
        </g>
      ))}
    </svg>
  );
}

function ScatterChart({ chart }: { chart: Chart }) {
  const pts = (chart.series[0]?.points ?? []).map((p) => ({ x: Number(p.x), y: p.y }));
  const xd = extent(pts.map((p) => p.x));
  const yd = extent(pts.map((p) => p.y));
  const xt = niceTicks(xd.min, xd.max, 5);
  const yt = niceTicks(yd.min, yd.max, 5);
  const x = scaleLinear({ min: Math.min(xd.min, xt[0]), max: Math.max(xd.max, xt[xt.length - 1]) }, [M.left, W - M.right]);
  const y = scaleLinear({ min: Math.min(yd.min, yt[0]), max: Math.max(yd.max, yt[yt.length - 1]) }, [H - M.bottom, M.top]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={chart.title} className="chart-svg">
      <Axes yTicks={yt} y={y} yLabel={chart.y_label} />
      {xt.map((tv) => (
        <text key={tv} x={x(tv)} y={H - M.bottom + 14} textAnchor="middle" className="tick">
          {formatNumber(tv)}
        </text>
      ))}
      <text x={(M.left + W - M.right) / 2} y={H - 8} textAnchor="middle" className="axis-label">
        {chart.x_label}
      </text>
      {pts.map((p, i) => (
        <circle key={i} cx={x(p.x)} cy={y(p.y)} r={3} className="dot scatter" />
      ))}
    </svg>
  );
}

function ValueCard({ chart, locale }: { chart: Chart; locale: Locale }) {
  return (
    <div className="value-card">
      <span className="badge reference">{chart.label ?? t(locale).referenceValue}</span>
      <div className="value">
        {chart.value != null ? formatNumber(chart.value) : "—"}
        <span className="unit">{chart.unit}</span>
      </div>
      {chart.note ? <p className="muted small">{chart.note}</p> : null}
    </div>
  );
}

export function ChartView({ chart, locale }: { chart: Chart; locale: Locale }) {
  return (
    <figure className="chart">
      <figcaption>{chart.title}</figcaption>
      {chart.kind === "bar" && <BarChart chart={chart} />}
      {chart.kind === "line" && <LineChart chart={chart} />}
      {chart.kind === "scatter" && <ScatterChart chart={chart} />}
      {chart.kind === "value" && <ValueCard chart={chart} locale={locale} />}
    </figure>
  );
}
