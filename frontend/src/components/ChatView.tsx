import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, postChat } from "../api/client";
import type { HistoryItem, Locale } from "../api/types";
import { t } from "../lib/strings";
import { MessageBubble, type UiMessage } from "./MessageBubble";

export function ChatView({ locale }: { locale: Locale }) {
  const s = t(locale);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const nextId = useRef(1);
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastQuestion = useRef<string>("");

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const errorText = useCallback(
    (e: unknown): string => {
      if (e instanceof ApiError) {
        const base = s.errors[e.kind];
        return e.retryAfterSeconds ? `${base} (${e.retryAfterSeconds}s)` : base;
      }
      return s.errors.server;
    },
    [s],
  );

  const send = useCallback(
    async (question: string, history: HistoryItem[]) => {
      setLoading(true);
      try {
        const res = await postChat({ question, locale, history });
        setMessages((m) => [
          ...m,
          { id: nextId.current++, role: "assistant", text: res.answer, response: res },
        ]);
      } catch (e) {
        setMessages((m) => [
          ...m,
          { id: nextId.current++, role: "assistant", text: "", error: errorText(e) },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [locale, errorText],
  );

  // 履歴は「質問→回答」のテキストだけ。エラーの回答は含めない。
  const buildHistory = (msgs: UiMessage[]): HistoryItem[] => {
    const items: HistoryItem[] = [];
    for (let i = 0; i < msgs.length - 1; i++) {
      const a = msgs[i];
      const b = msgs[i + 1];
      if (a.role === "user" && b.role === "assistant" && !b.error) {
        items.push({ question: a.text, answer: b.text });
      }
    }
    return items.slice(-6);
  };

  const submit = (question: string) => {
    const q = question.trim();
    if (!q || loading) return;
    lastQuestion.current = q;
    const history = buildHistory(messages);
    setMessages((m) => [...m, { id: nextId.current++, role: "user", text: q }]);
    setInput("");
    void send(q, history);
  };

  const retry = () => {
    // 直前のエラー回答を取り除いて、同じ質問を再送する。
    const q = lastQuestion.current;
    if (!q || loading) return;
    const base = messages.slice(0, -2); // [..., user(q), assistant(error)]
    setMessages((m) => m.slice(0, -1));
    void send(q, buildHistory(base));
  };

  return (
    <main className="chat">
      <section className="messages" aria-live="polite">
        {messages.length === 0 ? (
          <div className="empty">
            <ul className="suggestions">
              {s.suggestions.map((q) => (
                <li key={q}>
                  <button type="button" onClick={() => submit(q)}>
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {messages.map((m, i) => (
          <MessageBubble
            key={m.id}
            message={m}
            locale={locale}
            onRetry={m.error && i === messages.length - 1 ? retry : undefined}
          />
        ))}
        {loading ? (
          <div className="loading" role="status">
            <span className="spinner" aria-hidden="true" /> {s.thinking}
          </div>
        ) : null}
        <div ref={bottomRef} />
      </section>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          submit(input);
        }}
      >
        <textarea
          value={input}
          rows={2}
          maxLength={2000}
          placeholder={s.placeholder}
          aria-label={s.placeholder}
          disabled={loading}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // 日本語入力の変換確定のEnterでは送信しない(isComposing)。
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit(input);
            }
          }}
        />
        <button type="submit" disabled={loading || input.trim() === ""}>
          {s.send}
        </button>
      </form>
    </main>
  );
}
