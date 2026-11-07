"""直接呼び出しとMCP経由(インメモリ/stdio子プロセス)で、結果が同じであることの検証。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest

from health_insight_agent.agent.loop import HealthInsightAgent
from health_insight_agent.ai.rule_based_fake import RuleBasedFakeLLM
from health_insight_agent.mcp_server.client import McpToolExecutor, stdio_params
from health_insight_agent.mcp_server.server import build_mcp_server
from health_insight_agent.services.health_insight_service import LocalHealthInsightService
from health_insight_agent.services.insight_data_service import InsightDataService
from health_insight_agent.tools.base import ToolRegistry
from health_insight_agent.tools.builtin import build_default_registry
from tests.conftest import FIXED_END_DATE

pytestmark = pytest.mark.integration

ARGS = {
    "start_date": (FIXED_END_DATE - timedelta(days=89)).isoformat(),
    "end_date": FIXED_END_DATE.isoformat(),
    "condition": {"metric": "prev_day_late_meal", "op": "==", "value": 1},
    "compare_metrics": ["sleep_duration_min", "prev_day_breakfast_kcal"],
}


@pytest.fixture
def registry(seeded: Any) -> ToolRegistry:
    factory, _, _ = seeded
    return build_default_registry(
        InsightDataService(factory, clock=lambda: FIXED_END_DATE),
        LocalHealthInsightService(factory),
    )


@pytest.fixture
def in_memory(registry: ToolRegistry, seeded: Any) -> Iterator[McpToolExecutor]:
    ex = McpToolExecutor(build_mcp_server(registry, seeded[1]), seeded[1])
    yield ex
    ex.close()


def test_in_memory_mcp_equals_direct_for_every_tool(
    in_memory: McpToolExecutor, registry: ToolRegistry, seeded: Any
) -> None:
    uid = seeded[1]
    assert in_memory.specs() == registry.specs()
    period = {"start_date": ARGS["start_date"], "end_date": ARGS["end_date"]}
    calls: list[tuple[str, dict[str, Any]]] = [
        ("compare_days", ARGS),
        (
            "correlate",
            {**period, "x_metric": "sleep_duration_min", "y_metric": "same_day_fatigue_score"},
        ),
        ("get_sleep_logs", {**period, "limit": 20}),
        ("get_meal_logs", period),
        ("get_workout_logs", period),
        ("get_data_overview", {}),
        ("get_health_score", {}),
        ("get_recommendations", {}),
    ]
    for name, args in calls:
        a = in_memory.execute(uid, name, args)
        b = registry.execute(uid, name, args)
        assert a.ok and b.ok, name
        assert a.content == b.content, name
        assert a.charts == b.charts and a.rows_used == b.rows_used, name


def test_agent_answers_identically_over_mcp_and_direct(
    in_memory: McpToolExecutor, registry: ToolRegistry, seeded: Any
) -> None:
    uid = seeded[1]
    q = "睡眠が悪かった日の共通点は?"

    def ask(executor: Any) -> str:
        agent = HealthInsightAgent(
            RuleBasedFakeLLM(), executor, today=lambda: FIXED_END_DATE, sink=lambda _l: None
        )
        return agent.ask(uid, q).text

    assert ask(in_memory) == ask(registry)


def test_stdio_subprocess_server_returns_the_same_result(
    registry: ToolRegistry, seeded: Any
) -> None:
    """本物のMCPサーバーを子プロセス(stdio)で起動して呼ぶ。"""
    uid: UUID = seeded[1]
    ex = McpToolExecutor(
        stdio_params(extra_env={"MCP_USER_ID": str(uid), "MCP_TODAY": FIXED_END_DATE.isoformat()}),
        uid,
        connect_timeout_seconds=60,
    )
    try:
        # 子プロセスは本番構成(RAG・SQL・自作モデルあり)なので、テスト用レジストリの全ツール+それら
        extra = {
            "search_notes",
            "run_readonly_sql",
            "predict_sleep_duration",
            "explain_sleep_factors",
        }
        assert set(ex.names) == {*registry.names, *extra}
        via = ex.execute(uid, "compare_days", ARGS)
        direct = registry.execute(uid, "compare_days", ARGS)
        assert via.ok and via.content == direct.content and via.charts == direct.charts
    finally:
        ex.close()
