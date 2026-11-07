// バックエンド(src/health_insight_agent/api/schemas.py)のレスポンス型。変更時は両方そろえる。

export type Locale = "ja" | "en";

export interface HistoryItem {
  question: string;
  answer: string;
}

export interface ChatRequest {
  question: string;
  locale: Locale;
  history: HistoryItem[];
}

export interface ToolTraceItem {
  name: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  rows_used: number;
  duration_ms: number;
  result: Record<string, unknown>;
}

export interface ChartPoint {
  x: number | string;
  y: number;
}

export interface ChartSeries {
  name: string;
  points: ChartPoint[];
}

export interface Chart {
  kind: "line" | "bar" | "scatter" | "value";
  title: string;
  x_label?: string | null;
  y_label?: string | null;
  series: ChartSeries[];
  // kind === "value"(予測値など)
  value?: number | null;
  unit?: string | null;
  label?: string | null;
  note?: string | null;
}

export type Tone = "negative" | "positive" | "none" | "unknown";

export interface FindingRow {
  label: string;
  mean_a: number | null;
  mean_b: number | null;
  n_a: number;
  n_b: number;
  unit: string;
  verdict_text: string;
  tone: Tone;
}

/** 統計ツールの結果を人が読める形にした所見(バックエンド services/findings.py)。 */
export interface Finding {
  kind: "correlation" | "comparison";
  title: string;
  // correlation
  verdict_text?: string | null;
  tone?: Tone | null;
  strength?: number | null;
  r?: number | null;
  n?: number | null;
  reliability?: string | null;
  reliability_text?: string | null;
  // comparison
  n_a?: number | null;
  n_b?: number | null;
  rows?: FindingRow[];
}

export interface UsageInfo {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  llm_calls: number;
  tool_calls: number;
  duration_ms: number;
  model: string;
}

export type AnswerStatus = "ok" | "max_tool_calls" | "llm_error" | "empty";

export interface ChatResponse {
  answer: string;
  status: AnswerStatus;
  tool_trace: ToolTraceItem[];
  charts: Chart[];
  findings: Finding[];
  usage: UsageInfo;
}

export interface Meta {
  provider: string;
  max_tool_calls: number;
  demo_user: boolean;
}
