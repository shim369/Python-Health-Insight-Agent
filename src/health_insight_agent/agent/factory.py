"""設定からエージェントを組み立てる(CLI・API・評価で共通)。"""

from __future__ import annotations

from datetime import date

from health_insight_agent.agent.loop import AgentConfig, HealthInsightAgent
from health_insight_agent.agent.usage_log import JsonlSink, QuerySink, logging_sink
from health_insight_agent.ai.factory import build_llm_provider
from health_insight_agent.ai.protocols import LLMProvider
from health_insight_agent.config import PROJECT_ROOT, Settings
from health_insight_agent.local_time import local_today
from health_insight_agent.repositories.sqlalchemy.factory import build_uow_factory
from health_insight_agent.services.health_insight_service import LocalHealthInsightService
from health_insight_agent.services.insight_data_service import InsightDataService
from health_insight_agent.tools.base import ToolRegistry
from health_insight_agent.tools.builtin import build_default_registry


def build_registry(settings: Settings, *, today: date | None = None) -> ToolRegistry:
    # エージェントは読み取り専用。既存アプリのDBを読むときは READONLY_DATABASE_URL を使う
    # (書き込み権限の無い専用ユーザー)。無ければ開発DBを、トランザクションREAD ONLYで読む。
    url = settings.readonly_database_url or settings.database_url
    uow_factory = build_uow_factory(url, read_only=True)
    clock = (lambda: today) if today else local_today
    return build_default_registry(
        InsightDataService(uow_factory, clock=clock), LocalHealthInsightService(uow_factory)
    )


def build_agent(
    settings: Settings,
    *,
    llm: LLMProvider | None = None,
    config: AgentConfig | None = None,
    today: date | None = None,
    sink: QuerySink | None = None,
) -> HealthInsightAgent:
    return HealthInsightAgent(
        llm or build_llm_provider(settings.ai_provider, model=settings.gemini_model),
        build_registry(settings, today=today),
        config=config or AgentConfig(),
        today=(lambda: today) if today else local_today,
        sink=sink or JsonlSink(PROJECT_ROOT / "logs" / "agent_queries.jsonl"),
    )


__all__ = ["build_agent", "build_registry", "logging_sink"]
