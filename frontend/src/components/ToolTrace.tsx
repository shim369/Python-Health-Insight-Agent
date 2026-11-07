import type { Locale, ToolTraceItem } from "../api/types";
import { t } from "../lib/strings";

export function ToolTrace({ trace, locale }: { trace: ToolTraceItem[]; locale: Locale }) {
  const s = t(locale);
  if (trace.length === 0) return null;
  return (
    <details className="tool-trace">
      <summary>
        {s.tools} ({trace.length})
      </summary>
      <ol>
        {trace.map((item, i) => (
          <li key={i}>
            <div className="tool-head">
              <code>{item.name}</code>
              <span className={`badge ${item.ok ? "ok" : "ng"}`}>{item.ok ? "ok" : "error"}</span>
              <span className="muted small">
                {item.rows_used} {s.rows} · {Math.round(item.duration_ms)}ms
              </span>
            </div>
            <pre aria-label="arguments">{JSON.stringify(item.arguments, null, 2)}</pre>
            <details>
              <summary className="small">result</summary>
              <pre aria-label="result">{JSON.stringify(item.result, null, 2)}</pre>
            </details>
          </li>
        ))}
      </ol>
    </details>
  );
}
