import type { ReactNode } from "react";
import type { ChatResponse, Locale } from "../api/types";
import { t } from "../lib/strings";
import { ChartView } from "./Charts";
import { FindingCard } from "./FindingCard";
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

/** インライン記法(**太字** と `コード`)を、HTML文字列ではなくReact要素に変換する(XSS対策)。 */
export function renderInline(text: string): ReactNode[] {
  // LaTeX風の $...$ はそのまま見せると読みにくいので、囲みの$だけ外す
  const plain = text.replace(/\$([^$\n]{1,60})\$/g, "$1");
  const out: ReactNode[] = [];
  const re = /\*\*([^*\n]+)\*\*|`([^`\n]+)`/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(plain)) !== null) {
    if (m.index > last) out.push(plain.slice(last, m.index));
    out.push(m[1] !== undefined ? <strong key={key++}>{m[1]}</strong> : <code key={key++}>{m[2]}</code>);
    last = m.index + m[0].length;
  }
  if (last < plain.length) out.push(plain.slice(last));
  return out;
}

type Block =
  | { kind: "p"; lines: string[] }
  | { kind: "heading"; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] };

const BULLET = /^\s*[-*・]\s+/;
const ORDERED = /^\s*\d+[.)]\s+/;
const HEADING = /^\s*#{1,6}\s+/;
const RULE = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/;

/** 最小限のMarkdown整形: 見出し・箇条書き・番号付きリスト・区切り線・太字・コード。 */
export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  for (const raw of text.split("\n")) {
    if (raw.trim() === "" || RULE.test(raw)) continue;
    const last = blocks[blocks.length - 1];
    if (HEADING.test(raw)) {
      blocks.push({ kind: "heading", text: raw.replace(HEADING, "") });
    } else if (BULLET.test(raw)) {
      const item = raw.replace(BULLET, "");
      if (last?.kind === "ul") last.items.push(item);
      else blocks.push({ kind: "ul", items: [item] });
    } else if (ORDERED.test(raw)) {
      const item = raw.replace(ORDERED, "");
      if (last?.kind === "ol") last.items.push(item);
      else blocks.push({ kind: "ol", items: [item] });
    } else if (last?.kind === "p") {
      last.lines.push(raw);
    } else {
      blocks.push({ kind: "p", lines: [raw] });
    }
  }
  return blocks;
}

function AnswerText({ text }: { text: string }) {
  return (
    <div className="text">
      {parseBlocks(text).map((b, i) => {
        if (b.kind === "heading") return <p key={i} className="answer-heading">{renderInline(b.text)}</p>;
        if (b.kind === "ul" || b.kind === "ol") {
          const List = b.kind;
          return (
            <List key={i}>
              {b.items.map((item, j) => (
                <li key={j}>{renderInline(item)}</li>
              ))}
            </List>
          );
        }
        return <p key={i}>{renderInline(b.lines.join("\n"))}</p>;
      })}
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
        {r && r.findings.length > 0 ? (
          <div className="findings">
            {r.findings.map((f, i) => (
              <FindingCard key={i} finding={f} locale={locale} />
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
