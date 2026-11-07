import type { ChatResponse, Locale } from "../api/types";
import { t } from "../lib/strings";
import { ChartView } from "./Charts";
import { ToolTrace } from "./ToolTrace";

export interface UiMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  response?: ChatResponse;
  error?: string;
}

export function MessageBubble({
  message,
  locale,
  onRetry,
}: {
  message: UiMessage;
  locale: Locale;
  onRetry?: () => void;
}) {
  const s = t(locale);
  const isUser = message.role === "user";
  const r = message.response;
  return (
    <article className={`bubble ${isUser ? "user" : "assistant"}`} aria-label={isUser ? s.you : s.agent}>
      <div className="who">{isUser ? s.you : s.agent}</div>
      {message.error ? (
        <div role="alert" className="error">
          <p>{message.error}</p>
          {onRetry ? (
            <button type="button" onClick={onRetry}>
              {s.retry}
            </button>
          ) : null}
        </div>
      ) : (
        <div className="text">{message.text}</div>
      )}
      {r && r.status !== "ok" ? <p className="status-note">{s.statusNotes[r.status]}</p> : null}
      {r && r.charts.length > 0 ? (
        <div className="charts">
          {r.charts.map((c, i) => (
            <ChartView key={i} chart={c} locale={locale} />
          ))}
        </div>
      ) : null}
      {r ? <ToolTrace trace={r.tool_trace} locale={locale} /> : null}
      {r ? (
        <footer className="usage muted small">
          {r.usage.model} · {r.usage.total_tokens} {s.tokens} · {r.usage.tool_calls} {s.toolCalls} ·{" "}
          {Math.round(r.usage.duration_ms)}ms
        </footer>
      ) : null}
    </article>
  );
}
