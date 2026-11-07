import { useEffect, useState } from "react";
import type { Locale } from "./api/types";
import { ChatView } from "./components/ChatView";
import { LogoMark } from "./components/Icons";
import { t } from "./lib/strings";

const LOCALES: { value: Locale; label: string }[] = [
  { value: "ja", label: "日本語" },
  { value: "en", label: "English" },
];

export function App() {
  const [locale, setLocale] = useState<Locale>("ja");
  const s = t(locale);
  // 読み上げ・ハイフネーション・フォント選択のため、文書の言語も切り替える
  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);
  return (
    <div className="shell">
      <header className="app-header">
        <div className="brand">
          <LogoMark />
          <div>
            <h1>{s.title}</h1>
            <p className="muted">{s.subtitle}</p>
          </div>
        </div>
        <div className="header-actions">
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
        <ChatView locale={locale} />
      </div>
    </div>
  );
}
