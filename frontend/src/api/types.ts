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
  usage: UsageInfo;
}

export interface Meta {
  provider: string;
  max_tool_calls: number;
  demo_user: boolean;
}
