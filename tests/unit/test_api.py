from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient

from health_insight_agent.agent.loop import AgentConfig, HealthInsightAgent
from health_insight_agent.ai.fake_provider import (
    ScriptedLLMProvider,
    text_response,
    tool_call_response,
)
from health_insight_agent.ai.protocols import ToolCall
from health_insight_agent.api.app import create_app
from health_insight_agent.config import DEFAULT_DEMO_USER_ID, Settings
from health_insight_agent.tools import charts
from health_insight_agent.tools.base import Tool, ToolArgs, ToolOutput, ToolRegistry


class _Args(ToolArgs):
    pass


SETTINGS = Settings(
    database_url="postgresql+psycopg://x/y_test",
    readonly_database_url=None,
    demo_user_id=DEFAULT_DEMO_USER_ID,
    app_env="test",
    ai_provider="fake",
    gemini_model=None,
    tool_transport="direct",
    sql_tool_enabled=True,
    model_tool_enabled=True,
    ml_tracking_enabled=False,
    log_level="INFO",
)


def _client(script: list, limit: int = 100, seen: list[UUID] | None = None) -> TestClient:  # type: ignore[type-arg]
    def handler(user_id: UUID, a: _Args) -> ToolOutput:
        if seen is not None:
            seen.append(user_id)
        return ToolOutput(
            data={"v": 1},
            rows_used=7,
            period="p",
            charts=[charts.bar_chart("t", [("a", 1.0), ("b", 2.0)], y_label="kcal")],
        )

    agent = HealthInsightAgent(
        ScriptedLLMProvider(script),
        ToolRegistry([Tool("t", "d", _Args, handler)]),
        config=AgentConfig(max_tool_calls=3),
        sink=lambda _l: None,
    )
    app = create_app(
        settings=SETTINGS, agent=agent, rate_limit_per_min=limit, check_db_on_startup=False
    )
    return TestClient(app)


def test_chat_returns_answer_trace_charts_and_usage() -> None:
    seen: list[UUID] = []
    c = _client([tool_call_response(ToolCall("c1", "t", {})), text_response("回答です")], seen=seen)
    r = c.post("/api/chat", json={"question": "質問", "locale": "ja"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("回答です") and "医療診断" in body["answer"]
    assert body["status"] == "ok"
    [t] = body["tool_trace"]
    assert (t["name"], t["ok"], t["rows_used"]) == ("t", True, 7)
    assert t["result"]["data"] == {"v": 1}
    assert body["charts"][0]["kind"] == "bar" and len(body["charts"][0]["series"][0]["points"]) == 2
    assert body["usage"]["tool_calls"] == 1 and body["usage"]["total_tokens"] > 0
    assert seen == [DEFAULT_DEMO_USER_ID]  # 認証の代用(固定デモユーザー)が使われる


def test_request_validation() -> None:
    c = _client([text_response("x")])
    assert c.post("/api/chat", json={"question": ""}).status_code == 422
    assert c.post("/api/chat", json={"question": "a" * 2001}).status_code == 422
    assert c.post("/api/chat", json={"question": "q", "locale": "fr"}).status_code == 422
    assert c.post("/api/chat", json={"question": "q", "user_id": "x"}).status_code == 200  # 無視


def test_client_cannot_choose_the_user() -> None:
    seen: list[UUID] = []
    c = _client([tool_call_response(ToolCall("c1", "t", {})), text_response("x")], seen=seen)
    c.post("/api/chat", json={"question": "q", "user_id": str(UUID(int=99))})
    assert seen == [DEFAULT_DEMO_USER_ID]


def test_rate_limit_returns_429_with_retry_after() -> None:
    c = _client([text_response("1"), text_response("2"), text_response("3")], limit=2)
    assert c.post("/api/chat", json={"question": "q"}).status_code == 200
    assert c.post("/api/chat", json={"question": "q"}).status_code == 200
    r = c.post("/api/chat", json={"question": "q"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_llm_failure_is_a_200_with_status_not_a_crash() -> None:
    c = _client([])  # スクリプト切れ=LLMエラー
    r = c.post("/api/chat", json={"question": "q"})
    assert r.status_code == 200 and r.json()["status"] == "llm_error"


def test_health_and_meta() -> None:
    c = _client([])
    assert c.get("/api/health").json() == {"status": "ok"}
    assert c.get("/api/meta").json() == {
        "provider": "fake",
        "max_tool_calls": 3,
        "demo_user": True,
    }


def test_chart_texts_are_returned_in_the_requested_language() -> None:
    def handler(user_id: UUID, a: _Args) -> ToolOutput:
        return ToolOutput(
            data={},
            rows_used=1,
            charts=[
                charts.bar_chart(
                    charts.tr("睡眠時間", "Sleep duration"),
                    [(charts.tr("条件に該当", "Matching"), 1.0), ("2026-11-07", 2.0)],
                    y_label=charts.tr("分", "min"),
                ),
                charts.value_card(
                    charts.tr("予測", "Prediction"),
                    6.5,
                    unit=charts.tr("時間", "h"),
                    label=charts.tr("モデルによる参考値", "Model-based reference value"),
                ),
            ],
        )

    def client_for(script_calls: int) -> TestClient:
        script = []
        for _ in range(script_calls):
            script += [tool_call_response(ToolCall("c1", "t", {})), text_response("x")]
        agent = HealthInsightAgent(
            ScriptedLLMProvider(script),
            ToolRegistry([Tool("t", "d", _Args, handler)]),
            sink=lambda _l: None,
        )
        return TestClient(create_app(settings=SETTINGS, agent=agent, check_db_on_startup=False))

    c = client_for(2)
    ja = c.post("/api/chat", json={"question": "q", "locale": "ja"}).json()["charts"]
    en = c.post("/api/chat", json={"question": "q", "locale": "en"}).json()["charts"]
    assert ja[0]["title"] == "睡眠時間" and en[0]["title"] == "Sleep duration"
    assert ja[0]["y_label"] == "分" and en[0]["y_label"] == "min"
    assert [p["x"] for p in ja[0]["series"][0]["points"]] == ["条件に該当", "2026-11-07"]
    assert [p["x"] for p in en[0]["series"][0]["points"]] == ["Matching", "2026-11-07"]
    assert (
        ja[1]["label"] == "モデルによる参考値" and en[1]["label"] == "Model-based reference value"
    )
    assert en[1]["unit"] == "h" and "ja" not in str(en)  # 言語キーの辞書は残らない
