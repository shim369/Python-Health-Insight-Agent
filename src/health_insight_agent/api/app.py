"""FastAPIアプリ。Presentation層: 入力検証・認証の代用・エージェント呼び出し・整形のみを行う。

起動: uv run uvicorn health_insight_agent.api.app:create_app --factory --port 8000
"""

# 注意: ここでは `from __future__ import annotations` を使わない。FastAPIがローカル関数の
# 依存(Depends)の型注釈を実行時に解決できなくなるため。

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from health_insight_agent.agent.factory import build_agent
from health_insight_agent.agent.loop import AgentAnswer, HealthInsightAgent
from health_insight_agent.api import schemas
from health_insight_agent.api.rate_limit import RateLimiter
from health_insight_agent.config import Settings, load_settings
from health_insight_agent.database.connection import create_db_engine, wait_for_database
from health_insight_agent.domain.enums import Locale
from health_insight_agent.logging_config import get_logger
from health_insight_agent.services.findings import (
    ComparisonFinding,
    CorrelationFinding,
    build_findings,
)
from health_insight_agent.tools import charts as chart_texts

logger = get_logger(__name__)


def _finding(f: CorrelationFinding | ComparisonFinding) -> schemas.Finding:
    if isinstance(f, CorrelationFinding):
        return schemas.Finding(
            kind="correlation",
            title=f.title,
            verdict_text=f.verdict_text,
            tone=f.tone,  # type: ignore[arg-type]
            strength=f.strength,
            r=f.r,
            n=f.n,
            reliability=f.reliability,
            reliability_text=f.reliability_text,
        )
    return schemas.Finding(
        kind="comparison",
        title=f.title,
        n_a=f.n_a,
        n_b=f.n_b,
        rows=[
            schemas.FindingRow(
                label=r.label,
                mean_a=r.mean_a,
                mean_b=r.mean_b,
                n_a=r.n_a,
                n_b=r.n_b,
                unit=r.unit,
                verdict_text=r.verdict_text,
                tone=r.tone,  # type: ignore[arg-type]
            )
            for r in f.rows
        ],
    )


def _to_response(answer: AgentAnswer, locale: str) -> schemas.ChatResponse:
    return schemas.ChatResponse(
        answer=answer.text,
        status=answer.status,  # type: ignore[arg-type]
        tool_trace=[
            schemas.ToolTraceItem(
                name=t.name,
                arguments=t.arguments,
                ok=t.ok,
                rows_used=t.rows_used,
                duration_ms=round(t.duration_ms, 1),
                result=t.content,
            )
            for t in answer.tool_trace
        ],
        # グラフの文言は{ja,en}の両方を持っているので、利用者の言語で解決して返す
        charts=[
            schemas.Chart.model_validate(chart_texts.localize(c, locale))
            for t in answer.tool_trace
            for c in t.charts
        ],
        findings=[_finding(f) for f in build_findings(answer.tool_trace, locale)],
        usage=schemas.UsageInfo(
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            total_tokens=answer.usage.total_tokens,
            llm_calls=answer.llm_calls,
            tool_calls=answer.tool_calls,
            duration_ms=round(answer.duration_ms, 1),
            model=answer.model,
        ),
    )


def create_app(
    *,
    settings: Settings | None = None,
    agent: HealthInsightAgent | None = None,
    rate_limit_per_min: int | None = None,
    check_db_on_startup: bool = True,
) -> FastAPI:
    settings = settings or load_settings()
    limiter = RateLimiter(
        rate_limit_per_min or int(os.environ.get("CHAT_RATE_LIMIT_PER_MIN", "20"))
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if check_db_on_startup:
            # Neonのscale-to-zero復帰待ち。失敗しても起動は続け、リクエスト時にエラーを返す。
            try:
                await run_in_threadpool(
                    wait_for_database,
                    create_db_engine(settings.readonly_database_url or settings.database_url),
                )
            except Exception:
                logger.warning("database not reachable at startup", exc_info=True)
        yield
        the_agent.close()  # MCP経由のときは子プロセス・接続を閉じる

    app = FastAPI(title="Health Insight Agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    the_agent = agent or build_agent(settings)

    def current_user_id() -> UUID:
        # 認証の代用: 固定のデモユーザー。統合時はここだけをOAuthセッションから取得する
        # 実装に差し替える(ツール・サービスは user_id を引数で受けるので影響しない)。
        return settings.demo_user_id

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.get("/api/health", response_model=schemas.HealthResponse)
    def health() -> schemas.HealthResponse:
        return schemas.HealthResponse()

    @app.get("/api/meta", response_model=schemas.MetaResponse)
    def meta() -> schemas.MetaResponse:
        return schemas.MetaResponse(
            provider=settings.ai_provider, max_tool_calls=the_agent.max_tool_calls
        )

    @app.post(
        "/api/chat",
        response_model=schemas.ChatResponse,
        dependencies=[Depends(limiter.dependency())],
    )
    async def chat(
        req: schemas.ChatRequest, user_id: Annotated[UUID, Depends(current_user_id)]
    ) -> schemas.ChatResponse:
        history = [
            (h.question, h.answer[: schemas.MAX_HISTORY_CHARS])
            for h in req.history[-schemas.MAX_HISTORY_ITEMS :]
        ]
        # エージェントは同期処理(DB・LLMの待ち)なので、イベントループを塞がないようスレッドで実行。
        answer = await run_in_threadpool(
            the_agent.ask, user_id, req.question, locale=Locale(req.locale), history=history
        )
        return _to_response(answer, req.locale)

    return app
