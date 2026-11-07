import type { Locale } from "../api/types";

const ja = {
  title: "Health Insight Agent",
  subtitle: "食事・運動・睡眠ログを調べて、根拠つきで答えます(デモ: 合成データ)",
  placeholder: "質問を入力(Enterで送信、Shift+Enterで改行)",
  send: "送信",
  thinking: "データを調べています…",
  tools: "使ったツールと結果",
  usage: "使用量",
  retry: "もう一度送る",
  you: "あなた",
  agent: "エージェント",
  suggestions: [
    "先週、睡眠が悪かった日の共通点は?",
    "夜遅い食事は睡眠時間に影響している?",
    "朝食のカロリーと睡眠に関係はある?",
    "今週の健康スコアとおすすめは?",
  ],
  errors: {
    network: "サーバーに接続できませんでした。バックエンドが起動しているか確認してください。",
    rate_limit: "リクエストが多すぎます。少し待ってからもう一度お試しください。",
    validation: "入力内容が正しくありません。質問は1〜2000文字で入力してください。",
    server: "サーバーでエラーが発生しました。しばらくしてからもう一度お試しください。",
  },
  statusNotes: {
    ok: "",
    max_tool_calls: "ツールの呼び出し上限に達したため、途中までの結果で回答しています。",
    llm_error: "AIサービスに接続できませんでした。",
    empty: "回答を生成できませんでした。",
  },
  tokens: "トークン",
  toolCalls: "ツール呼び出し",
  rows: "件",
  referenceValue: "モデルによる参考値",
};

const en: typeof ja = {
  title: "Health Insight Agent",
  subtitle: "Investigates your meal/workout/sleep logs and answers with evidence (demo: synthetic data)",
  placeholder: "Ask a question (Enter to send, Shift+Enter for newline)",
  send: "Send",
  thinking: "Looking into your data…",
  tools: "Tools used and results",
  usage: "Usage",
  retry: "Send again",
  you: "You",
  agent: "Agent",
  suggestions: [
    "What did my bad-sleep days last week have in common?",
    "Does eating late affect how long I sleep?",
    "Is breakfast size related to my sleep?",
    "What is my health score and what should I improve?",
  ],
  errors: {
    network: "Could not reach the server. Check that the backend is running.",
    rate_limit: "Too many requests. Please wait a moment and try again.",
    validation: "Invalid input. Questions must be 1 to 2000 characters.",
    server: "A server error occurred. Please try again shortly.",
  },
  statusNotes: {
    ok: "",
    max_tool_calls: "Reached the tool-call limit, so this answer uses partial results.",
    llm_error: "Could not reach the AI service.",
    empty: "No answer was produced.",
  },
  tokens: "tokens",
  toolCalls: "tool calls",
  rows: "rows",
  referenceValue: "Model-based reference value",
};

export function t(locale: Locale): typeof ja {
  return locale === "ja" ? ja : en;
}
