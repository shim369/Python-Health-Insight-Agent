"""APIの入出力スキーマ(外部境界はPydantic)。フロントエンドのTypeScript型(frontend/src/api/types.ts)と対応させる。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_QUESTION_CHARS = 2000
MAX_HISTORY_ITEMS = 6
MAX_HISTORY_CHARS = 1500
"""履歴の1件あたりの最大文字数。長い回答を丸ごと送り返してトークンが膨らむのを防ぐ。"""


class HistoryItem(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    answer: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    locale: Literal["ja", "en"] = "ja"
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)


class ToolTraceItem(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool
    rows_used: int
    duration_ms: float
    result: dict[str, Any]


class ChartPoint(BaseModel):
    x: float | str
    y: float


class ChartSeries(BaseModel):
    name: str
    points: list[ChartPoint]


class Chart(BaseModel):
    kind: Literal["line", "bar", "scatter", "value"]
    title: str
    x_label: str | None = None
    y_label: str | None = None
    series: list[ChartSeries] = Field(default_factory=list)
    # kind="value"(単一の値。予測値など)
    value: float | None = None
    unit: str | None = None
    label: str | None = None
    note: str | None = None


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int
    duration_ms: float
    model: str


class ChatResponse(BaseModel):
    answer: str
    status: Literal["ok", "max_tool_calls", "llm_error", "empty"]
    tool_trace: list[ToolTraceItem]
    charts: list[Chart]
    usage: UsageInfo


class MetaResponse(BaseModel):
    provider: str
    max_tool_calls: int
    demo_user: bool = True
    """認証は未実装で固定のデモユーザーを使っている(統合時にOAuthへ置換)。"""


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
