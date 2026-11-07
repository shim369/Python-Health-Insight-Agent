"""ツールの基盤: 定義・登録・実行。

セキュリティ上の約束:
- `user_id`はLLMの引数に含めない。Registry.execute が呼び出し側(認証済みコンテキスト)から
  渡す。引数モデルは extra="forbid" なので、LLMが user_id を渡そうとしても拒否される。
- 引数は必ずPydanticで検証してから実行する。検証失敗・不正な値は、例外ではなく
  「エラー結果」としてLLMに返し、自分で修正して再試行できるようにする。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from health_insight_agent.ai.protocols import ToolSpec
from health_insight_agent.local_time import APP_TIMEZONE
from health_insight_agent.logging_config import get_logger
from health_insight_agent.services.insight_data_service import InvalidRequestError

logger = get_logger(__name__)


class ToolArgs(BaseModel):
    """全ツール引数モデルの基底。未知のフィールド(user_id含む)は拒否する。"""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """ツールの成功結果。`evidence`は「何をどれだけ使ったか」(UIの根拠表示・回答の件数明示用)。"""

    data: dict[str, Any]
    rows_used: int
    period: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    args_model: type[ToolArgs]
    handler: Callable[[UUID, Any], ToolOutput]
    enrich_schema: Callable[[dict[str, Any]], None] | None = None
    """JSON Schemaに追加の制約(指標名のenum等)を足すフック。"""


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """1回のツール実行の記録(ログ・UIの根拠表示・評価に使う)。"""

    name: str
    arguments: dict[str, Any]
    ok: bool
    content: dict[str, Any]
    """LLMに返す内容(成功時は{"ok":true,"data":...,"evidence":...}、失敗時は{"ok":false,"error":...})。"""
    duration_ms: float
    rows_used: int = 0


def iso_jst(dt: datetime) -> str:
    """JSTのISO文字列(秒まで)。LLMにUTCを見せると日付を取り違えやすいため。"""
    return dt.astimezone(APP_TIMEZONE).isoformat(timespec="minutes")


def _resolve_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            return _resolve_refs(defs[node["$ref"].split("/")[-1]], defs)
        out = {k: _resolve_refs(v, defs) for k, v in node.items() if k not in ("title", "$defs")}
        # Optional[X] は anyOf[X, null]。モデルには「任意項目」として単純なXだけを見せる。
        any_of = out.get("anyOf")
        if isinstance(any_of, list):
            non_null = [a for a in any_of if a.get("type") != "null"]
            if len(non_null) == 1:
                out.pop("anyOf")
                out = {**non_null[0], **out}
        return out
    if isinstance(node, list):
        return [_resolve_refs(v, defs) for v in node]
    return node


def tool_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydanticモデル→LLM向けの単純なJSON Schema($ref/$defs/titleを展開・除去)。"""
    schema = model.model_json_schema()
    resolved = _resolve_refs(schema, schema.get("$defs", {}))
    assert isinstance(resolved, dict)
    return resolved


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        specs = []
        for t in self._tools.values():
            schema = tool_json_schema(t.args_model)
            if t.enrich_schema is not None:
                t.enrich_schema(schema)
            specs.append(ToolSpec(name=t.name, description=t.description, parameters_schema=schema))
        return specs

    def execute(self, user_id: UUID, name: str, arguments: dict[str, Any]) -> ToolExecution:
        started = time.perf_counter()

        def done(ok: bool, content: dict[str, Any], rows: int = 0) -> ToolExecution:
            return ToolExecution(
                name=name,
                arguments=arguments,
                ok=ok,
                content=content,
                duration_ms=(time.perf_counter() - started) * 1000,
                rows_used=rows,
            )

        tool = self._tools.get(name)
        if tool is None:
            return done(False, {"ok": False, "error": f"unknown tool {name!r}"})
        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in e.errors()
            )
            return done(False, {"ok": False, "error": f"invalid arguments: {problems}"})
        try:
            out = tool.handler(user_id, args)
        except InvalidRequestError as e:
            return done(False, {"ok": False, "error": str(e)})
        except Exception:
            # 内部エラーの詳細(SQL・スタック等)はLLM/ユーザーに出さず、ログにだけ残す。
            logger.exception("tool %s failed", name)
            return done(False, {"ok": False, "error": "internal error while running the tool"})
        content = {
            "ok": True,
            "data": out.data,
            "evidence": {"rows_used": out.rows_used, "period": out.period, "notes": out.notes},
        }
        try:
            json.dumps(content)  # JSONにできない値はLLMに渡せない。実装ミスとして即エラーにする
        except TypeError:
            logger.exception("tool %s returned non-JSON data", name)
            return done(False, {"ok": False, "error": "internal error while running the tool"})
        return done(True, content, out.rows_used)
