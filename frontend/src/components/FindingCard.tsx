import type { Finding, FindingRow, Locale, Tone } from "../api/types";
import { formatNumber } from "../lib/chartScale";
import { t } from "../lib/strings";

const TONE_CLASS: Record<Tone, string> = {
  negative: "tone-negative",
  positive: "tone-positive",
  none: "tone-none",
  unknown: "tone-unknown",
};

/** 相関係数 r(-1〜+1)を、中央を0とする横メーターで示す。強さが一目で分かるようにする。 */
function Meter({ r, tone, label }: { r: number | null; tone: Tone; label: string }) {
  const value = r == null ? 0 : Math.max(-1, Math.min(1, r));
  const width = Math.abs(value) * 50; // 片側が最大50%
  const left = value >= 0 ? 50 : 50 - width;
  return (
    <div className="meter" role="img" aria-label={label}>
      <div className="meter-track">
        <span className="meter-center" />
        {r != null ? (
          <span className={`meter-fill ${TONE_CLASS[tone]}`} style={{ left: `${left}%`, width: `${width}%` }} />
        ) : null}
      </div>
    </div>
  );
}

function ComparisonTable({ rows, locale, nA, nB }: { rows: FindingRow[]; locale: Locale; nA: number; nB: number }) {
  const s = t(locale).finding;
  const unit = (r: FindingRow) => (r.unit === "%" ? "%" : r.unit ? ` ${r.unit}` : "");
  const value = (v: number | null, r: FindingRow) => (v == null ? "—" : `${formatNumber(v)}${unit(r)}`);
  return (
    <table className="finding-table">
      <thead>
        <tr>
          <th scope="col" />
          <th scope="col">
            {s.groupA}
            <small>
              {nA} {s.days}
            </small>
          </th>
          <th scope="col">
            {s.groupB}
            <small>
              {nB} {s.days}
            </small>
          </th>
          <th scope="col" />
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label}>
            <th scope="row">{r.label}</th>
            <td>{value(r.mean_a, r)}</td>
            <td>{value(r.mean_b, r)}</td>
            <td>
              <span className={`chip ${TONE_CLASS[r.tone]}`}>{r.verdict_text}</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function FindingCard({ finding, locale }: { finding: Finding; locale: Locale }) {
  const s = t(locale).finding;
  if (finding.kind === "comparison") {
    return (
      <section className="finding" aria-label={s.comparisonTitle}>
        <header>
          <span className="finding-kicker">{s.comparisonTitle}</span>
          <h3>{finding.title}</h3>
        </header>
        <ComparisonTable rows={finding.rows ?? []} locale={locale} nA={finding.n_a ?? 0} nB={finding.n_b ?? 0} />
      </section>
    );
  }
  const tone: Tone = finding.tone ?? "unknown";
  return (
    <section className="finding" aria-label={s.correlationTitle}>
      <header>
        <span className="finding-kicker">{s.correlationTitle}</span>
        <h3>{finding.title}</h3>
      </header>
      <p className={`finding-verdict ${TONE_CLASS[tone]}`}>{finding.verdict_text}</p>
      <Meter r={finding.r ?? null} tone={tone} label={`${s.r} = ${finding.r == null ? "—" : finding.r.toFixed(2)}`} />
      <div className="meter-scale" aria-hidden="true">
        <span>−1 · {s.meterLeft}</span>
        <span>0</span>
        <span>{s.meterRight} · +1</span>
      </div>
      <dl className="finding-stats">
        <div>
          <dt>{s.r}</dt>
          <dd>{finding.r == null ? "—" : finding.r.toFixed(2)}</dd>
        </div>
        <div>
          <dt>{s.n}</dt>
          <dd>{finding.n ?? "—"}</dd>
        </div>
        <div>
          <dd className="chip">{finding.reliability_text}</dd>
        </div>
      </dl>
      <p className="finding-note">{s.note}</p>
    </section>
  );
}
