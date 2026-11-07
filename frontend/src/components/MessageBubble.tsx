import type { ChatResponse, Locale } from "../api/types";
import { t } from "../lib/strings";
import { ChartView } from "./Charts";
import { IconHelp, IconInfo, IconSpark, IconUser } from "./Icons";
import { ToolTrace } from "./ToolTrace";

export interface UiMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  response?: ChatResponse;
  error?: string;
}

/** 回答本文と、コードが末尾に固定で付ける注意書き(「※ …」)を分ける。注意書きは控えめに表示する。 */
export function splitDisclaimer(text: string): { body: string; disclaimer: string | null } {
  const i = text.lastIndexOf("\n※");
  if (i === -1) return { body: text, disclaimer: null };
  return { body: text.slice(0, i).trimEnd(), disclaimer: text.slice(i).replace(/^\s*※\s*/, "") };
}

const UNKNOWN_HEAD = /^\s*(わからないこと|分からないこと|What is unknown)\s*[:：]/i;

/** 「わからないこと: …」の段落(続く箇条書きを含む)を本文から切り出す。吹き出しの下に別枠で出すため。 */
export function splitUnknown(text: string): { body: string; unknown: string | null } {
  const lines = text.split("\n");
  let start = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (UNKNOWN_HEAD.test(lines[i])) {
      start = i;
      break;
    }
  }
  if (start === -1) return { body: text, unknown: null };
  let end = start + 1;
  while (end < lines.length && /^\s*[-・]\s+/.test(lines[end])) end++;
  const unknown = lines.slice(start, end).join("\n").trim();
  const body = [...lines.slice(0, start), ...lines.slice(end)].join("\n").trimEnd();
  return { body, unknown };
}

/** 「- 」で始まる行は箇条書き、それ以外は段落として描画する(最小限の整形)。 */
function AnswerText({ text }: { text: string }) {
  const blocks: { list: boolean; lines: string[] }[] = [];
  for (const line of text.split("\n")) {
    if (line.trim() === "") continue;
    const isItem = /^\s*[-・]\s+/.test(line);
    const cleaned = isItem ? line.replace(/^\s*[-・]\s+/, "") : line;
    const last = blocks[blocks.length - 1];
    if (isItem && last?.list) last.lines.push(cleaned);
    else blocks.push({ list: isItem, lines: [cleaned] });
  }
  return (
    <div className="text">
      {blocks.map((b, i) =>
        b.list ? (
          <ul key={i}>
            {b.lines.map((l, j) => (
              <li key={j}>{l}</li>
            ))}
          </ul>
        ) : (
          <p key={i}>{b.lines.join("\n")}</p>
        ),
      )}
    </div>
  );
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
  const { body: withUnknown, disclaimer } = splitDisclaimer(message.text);
  const { body, unknown } = isUser ? { body: withUnknown, unknown: null } : splitUnknown(withUnknown);
  return (
    <article className={`message ${isUser ? "user" : "assistant"}`} aria-label={isUser ? s.you : s.agent}>
      <div className="avatar" aria-hidden="true">
        {isUser ? <IconUser /> : <IconSpark />}
      </div>
      <div className="message-main">
      <div className="bubble">
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
          <AnswerText text={body} />
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
        {disclaimer ? (
          <p className="disclaimer">
            <IconInfo />
            <span>{disclaimer}</span>
          </p>
        ) : null}
        {r ? (
          <footer className="usage">
            <span>{r.usage.model}</span>
            <span>
              {r.usage.total_tokens} {s.tokens}
            </span>
            <span>
              {r.usage.tool_calls} {s.toolCalls}
            </span>
            <span>{Math.round(r.usage.duration_ms)}ms</span>
          </footer>
        ) : null}
      </div>
      {unknown ? (
        <aside className="unknown-note" aria-label={s.unknownLabel}>
          <IconHelp />
          <AnswerText text={unknown} />
        </aside>
      ) : null}
      </div>
    </article>
  );
}
