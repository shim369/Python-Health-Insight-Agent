"""run_readonly_sql: ガード付きの自由SQLツール。

型付きツールで答えられない自由集計のための最後の手段。実行前に readonly_sql/guard.py が
検証・ユーザー絞り込みの強制・LIMIT注入を行い、executor が READ ONLY・タイムアウトで実行する。
"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import Field

from health_insight_agent.readonly_sql.executor import ReadOnlySqlExecutor, SqlExecutionError
from health_insight_agent.readonly_sql.guard import MAX_SQL_CHARS, SqlGuardError, guard_sql
from health_insight_agent.readonly_sql.schema_doc import build_schema_doc
from health_insight_agent.services.insight_data_service import InvalidRequestError
from health_insight_agent.tools import charts
from health_insight_agent.tools.base import Tool, ToolArgs, ToolOutput, iso_jst

MAX_ROWS_LIMIT = 500
MAX_RESULT_CHARS = 30_000
"""LLMに返すJSONの文字数上限(トークン・コスト対策)。超えたら行を切り詰める。"""


class SqlArgs(ToolArgs):
    sql: str = Field(min_length=1, max_length=MAX_SQL_CHARS, description="One SELECT statement")
    max_rows: int = Field(default=100, ge=1, le=MAX_ROWS_LIMIT, description="Max rows returned")


def to_json_value(v: Any) -> Any:
    """DBの値をJSONにできる形へ。日時はJSTに直す(LLMがUTCで日付を取り違えないように)。"""
    if v is None or isinstance(v, bool | int | str):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 6)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, datetime):
        return iso_jst(v) if v.tzinfo else v.isoformat(timespec="seconds")
    if isinstance(v, date | time):
        return v.isoformat()
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, list | tuple):
        return [to_json_value(x) for x in v]
    return str(v)


def _maybe_chart(columns: list[str], rows: list[list[Any]]) -> list[dict[str, Any]]:
    """「ラベル列+数値列」の小さな結果なら棒グラフにする(決定的に、LLMを介さず)。"""
    if len(columns) < 2 or not (1 < len(rows) <= 40):
        return []
    ys = [r[1] for r in rows]
    if not all(isinstance(y, int | float) and not isinstance(y, bool) for y in ys):
        return []
    return [
        charts.bar_chart(
            f"SQL: {columns[1]} by {columns[0]}",
            [(str(r[0]), float(r[1])) for r in rows],
            y_label=columns[1],
        )
    ]


def build_sql_tool(executor: ReadOnlySqlExecutor) -> Tool:
    def handler(user_id: UUID, a: SqlArgs) -> ToolOutput:
        try:
            guarded = guard_sql(a.sql, user_id, max_rows=a.max_rows)
        except SqlGuardError as e:
            raise InvalidRequestError(f"SQL rejected by the safety guard: {e}") from e
        try:
            result = executor.run(guarded.sql, max_rows=a.max_rows)
        except SqlExecutionError as e:
            msg = {
                "timeout": f"query timed out after {executor.timeout_seconds:g}s; narrow it",
                "permission": "permission denied for a table",
                "read_only": "writes are not allowed",
            }.get(e.kind, f"database error: {e}")
            raise InvalidRequestError(msg) from e
        rows = [[to_json_value(v) for v in r] for r in result.rows]
        truncated = result.truncated
        while rows and len(str(rows)) > MAX_RESULT_CHARS:
            rows = rows[: max(1, len(rows) // 2)]
            truncated = True
            if len(rows) == 1:
                break
        notes = ["read-only query; your data was automatically scoped to the current user"]
        if truncated:
            notes.append(f"result truncated to {len(rows)} rows; aggregate or narrow the query")
        return ToolOutput(
            data={
                "columns": result.columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
            },
            rows_used=len(rows),
            notes=notes,
            charts=_maybe_chart(result.columns, rows),
        )

    return Tool(
        "run_readonly_sql",
        "Run ONE read-only SELECT over the user's own logs for custom aggregation that the typed "
        f"tools cannot do. {build_schema_doc()}",
        SqlArgs,
        handler,
    )
