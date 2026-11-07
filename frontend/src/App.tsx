import { useState } from "react";
import type { Locale } from "./api/types";
import { ChatView } from "./components/ChatView";
import { t } from "./lib/strings";

export function App() {
  const [locale, setLocale] = useState<Locale>("ja");
  const s = t(locale);
  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>{s.title}</h1>
          <p className="muted">{s.subtitle}</p>
        </div>
        <label className="lang">
          <span className="visually-hidden">Language</span>
          <select value={locale} onChange={(e) => setLocale(e.target.value as Locale)}>
            <option value="ja">日本語</option>
            <option value="en">English</option>
          </select>
        </label>
      </header>
      <ChatView locale={locale} />
    </div>
  );
}
