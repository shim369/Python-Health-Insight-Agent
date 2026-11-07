"""キー無しFake LLM + 実DB(合成データ)で、質問→ツール→回答を通しで確認する。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest

from health_insight_agent.agent.loop import HealthInsightAgent
from health_insight_agent.ai.rule_based_fake import RuleBasedFakeLLM
from health_insight_agent.services.health_insight_service import LocalHealthInsightService
from health_insight_agent.services.insight_data_service import InsightDataService
from health_insight_agent.tools.builtin import build_default_registry
from tests.conftest import FIXED_END_DATE

pytestmark = pytest.mark.integration


@pytest.fixture
def agent_env(seeded: tuple[Any, UUID, UUID]) -> tuple[HealthInsightAgent, UUID]:
    factory, uid, _ = seeded
    reg = build_default_registry(
        InsightDataService(factory, clock=lambda: FIXED_END_DATE),
        LocalHealthInsightService(factory),
    )
    agent = HealthInsightAgent(
        RuleBasedFakeLLM(), reg, today=lambda: FIXED_END_DATE, sink=lambda _log: None
    )
    return agent, uid


def test_why_bad_sleep_question_runs_compare_days(agent_env: Any) -> None:
    agent, uid = agent_env
    ans = agent.ask(uid, "睡眠が悪かった日の共通点は?")
    assert ans.status == "ok" and [t.name for t in ans.tool_trace] == ["compare_days"]
    assert ans.tool_trace[0].ok and ans.tool_trace[0].rows_used > 50
    assert "判定=" in ans.text and "医療診断" in ans.text
    assert ans.usage.total_tokens > 0 and ans.duration_ms > 0


def test_null_pattern_is_answered_as_no_relationship(agent_env: Any) -> None:
    agent, uid = agent_env
    ans = agent.ask(uid, "朝食の量と睡眠時間に影響はある?")
    assert [t.name for t in ans.tool_trace] == ["correlate"]
    assert "no_clear_relationship" in ans.text


def test_score_question_uses_both_wrapped_services(agent_env: Any) -> None:
    agent, uid = agent_env
    ans = agent.ask(uid, "健康スコアとおすすめは?")
    assert [t.name for t in ans.tool_trace] == ["get_health_score", "get_recommendations"]


def test_unknown_user_gets_insufficient_data_not_fabrication(agent_env: Any) -> None:
    agent, _ = agent_env
    ans = agent.ask(UUID(int=12345), "睡眠が悪かった日の共通点は?")
    assert "insufficient_data" in ans.text
    assert (FIXED_END_DATE - timedelta(days=1)).isoformat()  # 日付はテスト固定
