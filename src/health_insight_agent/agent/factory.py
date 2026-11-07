"""設定からエージェントを組み立てる(CLI・API・評価で共通)。"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from health_insight_agent.agent.loop import AgentConfig, HealthInsightAgent
from health_insight_agent.agent.prompts import MODEL_RULE, NOTES_RULE, SQL_RULE
from health_insight_agent.agent.usage_log import JsonlSink, QuerySink, logging_sink
from health_insight_agent.ai.embeddings import OUTPUT_DIMENSIONS
from health_insight_agent.ai.factory import build_embedding_provider, build_llm_provider
from health_insight_agent.ai.protocols import LLMProvider
from health_insight_agent.config import PROJECT_ROOT, Settings
from health_insight_agent.local_time import local_today
from health_insight_agent.ml.model_registry import ModelRegistry
from health_insight_agent.ml.service import SleepModelService
from health_insight_agent.ml.tracking import MlflowTracker, NullTracker
from health_insight_agent.rag.note_search import EmbeddingCache, NoteSearchService
from health_insight_agent.rag.vector_store import NumpyVectorStore
from health_insight_agent.readonly_sql.executor import ReadOnlySqlExecutor
from health_insight_agent.repositories.sqlalchemy.factory import build_uow_factory
from health_insight_agent.services.health_insight_service import LocalHealthInsightService
from health_insight_agent.services.insight_data_service import InsightDataService
from health_insight_agent.tools.base import ToolRegistry
from health_insight_agent.tools.builtin import build_default_registry
from health_insight_agent.tools.executor import ToolExecutor


def build_registry(
    settings: Settings,
    *,
    today: date | None = None,
    enable_notes: bool = True,
    enable_sql: bool | None = None,
    enable_model: bool | None = None,
) -> ToolRegistry:
    # エージェントは読み取り専用。既存アプリのDBを読むときは READONLY_DATABASE_URL を使う
    # (書き込み権限の無い専用ユーザー)。無ければ開発DBを、トランザクションREAD ONLYで読む。
    url = settings.readonly_database_url or settings.database_url
    uow_factory = build_uow_factory(url, read_only=True)
    clock = (lambda: today) if today else local_today
    embedder = build_embedding_provider(settings.ai_provider)
    notes = NoteSearchService(
        uow_factory,
        embedder,
        NumpyVectorStore(),
        # 同じメモを再度埋め込まないためのキャッシュ(コスト管理)。モデルごとに名前空間を分ける。
        cache=EmbeddingCache(
            # モデル名と次元数も名前空間に含め、設定を変えたときに古いベクトルを使い回さない
            f"{settings.ai_provider}:{type(embedder).__name__}:"
            f"{getattr(embedder, 'model', '')}:{OUTPUT_DIMENSIONS}",
            PROJECT_ROOT / ".cache" / f"embeddings-{settings.ai_provider}.json",
        ),
        clock=clock,
    )
    data = InsightDataService(uow_factory, clock=clock)
    use_model = settings.model_tool_enabled if enable_model is None else enable_model
    model = (
        SleepModelService(
            data,
            # ML_ARTIFACT_DIR: 採用判断JSONの保存先(テストで一時ディレクトリに向けるため)
            ModelRegistry(Path(os.environ.get("ML_ARTIFACT_DIR") or PROJECT_ROOT / "ml_artifacts")),
            tracker=MlflowTracker(PROJECT_ROOT / "mlruns")
            if settings.ml_tracking_enabled
            else NullTracker(),
        )
        if use_model
        else None
    )
    return build_default_registry(
        data,
        LocalHealthInsightService(uow_factory),
        notes if enable_notes else None,
        ReadOnlySqlExecutor(url)
        if (settings.sql_tool_enabled if enable_sql is None else enable_sql)
        else None,
        model,
    )


def build_tool_executor(settings: Settings, *, today: date | None = None) -> ToolExecutor:
    """TOOL_TRANSPORT=direct|mcp でツールの呼び方を切り替える(エージェント側のコードは同じ)。"""
    if settings.tool_transport == "mcp":
        from health_insight_agent.mcp_server.client import McpToolExecutor, stdio_params

        extra = {"MCP_TODAY": today.isoformat()} if today else {}
        return McpToolExecutor(
            stdio_params(extra_env={**extra, "MCP_USER_ID": str(settings.demo_user_id)}),
            settings.demo_user_id,
        )
    return build_registry(settings, today=today)


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
        build_tool_executor(settings, today=today),
        config=config or AgentConfig(extra_rules=(NOTES_RULE, SQL_RULE, MODEL_RULE)),
        today=(lambda: today) if today else local_today,
        sink=sink or JsonlSink(PROJECT_ROOT / "logs" / "agent_queries.jsonl"),
    )


__all__ = ["build_agent", "build_registry", "logging_sink"]
