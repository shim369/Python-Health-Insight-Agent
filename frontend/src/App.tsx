import { useEffect, useState } from "react";
import type { Locale } from "./api/types";
import { ChatView } from "./components/ChatView";
import { IconPlus, LogoMark } from "./components/Icons";
import { t } from "./lib/strings";

const LOCALES: { value: Locale; label: string }[] = [
  { value: "ja", label: "日本語" },
  { value: "en", label: "English" },
];

export function App() {
  const [locale, setLocale] = useState<Locale>("ja");
  // ChatViewのkeyを変えると作り直され、会話がクリアされてトップ(最初の画面)に戻る。
  const [chatKey, setChatKey] = useState(0);
  const s = t(locale);
  // 読み上げ・ハイフネーション・フォント選択のため、文書の言語も切り替える
  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const backToTop = () => {
    setChatKey((k) => k + 1);
    window.scrollTo?.({ top: 0 });
  };

  return (
    <div className="shell">
      <header className="app-header">
        <button type="button" className="brand" onClick={backToTop} aria-label={s.backToTop} title={s.backToTop}>
          <LogoMark />
          <span className="brand-text">
            <h1>{s.title}</h1>
            <span className="muted">{s.subtitle}</span>
          </span>
        </button>
        <div className="header-actions">
          <button type="button" className="new-chat" onClick={backToTop}>
            <IconPlus />
            {s.newChat}
          </button>
          <span className="badge demo">{s.demoBadge}</span>
          <div className="segmented" role="group" aria-label="Language">
            {LOCALES.map((l) => (
              <button
                key={l.value}
                type="button"
                aria-pressed={locale === l.value}
                onClick={() => setLocale(l.value)}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>
      </header>
      <div className="app">
        <ChatView key={chatKey} locale={locale} />
      </div>
    </div>
  );
}
