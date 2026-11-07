from __future__ import annotations

from typing import Any
from uuid import UUID

from health_insight_agent.services.health_insight_service import HealthReport
from health_insight_agent.services.metrics import METRIC_NAMES
from health_insight_agent.tools.base import Tool, ToolArgs, ToolOutput, ToolRegistry
from health_insight_agent.tools.builtin import build_default_registry

USER = UUID(int=1)


class _Args(ToolArgs):
    n: int


def _registry(handler: Any) -> ToolRegistry:
    return ToolRegistry([Tool("t", "d", _Args, handler)])


def test_user_id_is_never_accepted_from_the_llm() -> None:
    seen: list[UUID] = []

    def handler(user_id: UUID, a: _Args) -> ToolOutput:
        seen.append(user_id)
        return ToolOutput(data={}, rows_used=0)

    reg = _registry(handler)
    bad = reg.execute(USER, "t", {"n": 1, "user_id": str(UUID(int=2))})
    assert not bad.ok and "user_id" in bad.content["error"]
    assert seen == []  # 実行されない
    ok = reg.execute(USER, "t", {"n": 1})
    assert ok.ok and seen == [USER]  # 認証済みコンテキストのuser_idだけが渡る


def test_validation_unknown_tool_and_internal_errors_are_returned_not_raised() -> None:
    def boom(user_id: UUID, a: _Args) -> ToolOutput:
        raise RuntimeError("secret SQL detail")

    reg = _registry(boom)
    assert "n:" in reg.execute(USER, "t", {"n": "x"}).content["error"]
    assert "unknown tool" in reg.execute(USER, "nope", {}).content["error"]
    err = reg.execute(USER, "t", {"n": 1})
    assert not err.ok and "secret" not in str(err.content)  # 内部詳細は漏らさない


def test_non_json_output_is_caught_immediately() -> None:
    def bad(user_id: UUID, a: _Args) -> ToolOutput:
        return ToolOutput(data={"x": object()}, rows_used=0)

    assert not _registry(bad).execute(USER, "t", {"n": 1}).ok


def test_default_tool_schemas_are_flat_and_constrain_metric_names() -> None:
    reg = build_default_registry(None, None)  # type: ignore[arg-type]
    specs = {s.name: s for s in reg.specs()}
    assert {
        "get_data_overview",
        "get_sleep_logs",
        "get_meal_logs",
        "get_workout_logs",
        "compare_days",
        "correlate",
        "get_health_score",
        "get_recommendations",
    } <= set(specs)
    for s in specs.values():
        text = str(s.parameters_schema)
        assert "$ref" not in text and "$defs" not in text and "anyOf" not in text
        assert "user_id" not in s.parameters_schema.get("properties", {})
    corr = specs["correlate"].parameters_schema["properties"]
    assert corr["x_metric"]["enum"] == list(METRIC_NAMES)
    cmp_props = specs["compare_days"].parameters_schema["properties"]
    assert cmp_props["condition"]["properties"]["op"]["enum"]
    assert cmp_props["compare_metrics"]["items"]["enum"] == list(METRIC_NAMES)
    assert "start_date" in specs["get_sleep_logs"].parameters_schema["required"]


def test_report_dataclass_is_importable() -> None:
    assert HealthReport.__name__ == "HealthReport"
