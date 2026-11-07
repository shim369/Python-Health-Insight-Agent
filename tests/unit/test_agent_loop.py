from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from health_insight_agent.agent.loop import AgentConfig, HealthInsightAgent
from health_insight_agent.agent.prompts import DISCLAIMERS
from health_insight_agent.agent.usage_log import QueryLog
from health_insight_agent.ai.errors import AIServiceUnavailableError
from health_insight_agent.ai.fake_provider import (
    CallbackLLMProvider,
    ScriptedLLMProvider,
    text_response,
    tool_call_response,
)
from health_insight_agent.ai.protocols import TokenUsage, ToolCall
from health_insight_agent.domain.enums import Locale
from health_insight_agent.tools.base import Tool, ToolArgs, ToolOutput, ToolRegistry

USER = UUID("11111111-2222-4333-8444-555555555555")


class _Args(ToolArgs):
    n: int = 1


def _registry(calls: list[UUID] | None = None) -> ToolRegistry:
    def handler(user_id: UUID, a: _Args) -> ToolOutput:
        if calls is not None:
            calls.append(user_id)
        return ToolOutput(data={"value": a.n * 2}, rows_used=a.n, period="p")

    return ToolRegistry([Tool("double", "doubles", _Args, handler)])


def _agent(llm: Any, *, max_calls: int = 8, sink: list[QueryLog] | None = None, **kw: Any):  # type: ignore[no-untyped-def]
    out = sink if sink is not None else []
    return HealthInsightAgent(
        llm,
        kw.pop("registry", None) or _registry(),
        config=AgentConfig(max_tool_calls=max_calls),
        today=lambda: date(2026, 11, 7),
        sink=out.append,
    ), out


def test_tool_then_answer_records_trace_usage_and_log() -> None:
    llm = ScriptedLLMProvider(
        [
            tool_call_response(ToolCall("c1", "double", {"n": 3}), usage=TokenUsage(100, 10)),
            text_response("結果は6です", usage=TokenUsage(150, 20)),
        ]
    )
    seen: list[UUID] = []
    agent, logs = _agent(llm, registry=_registry(seen))
    ans = agent.ask(USER, "二倍して")
    assert ans.status == "ok" and ans.llm_calls == 2 and ans.tool_calls == 1
    assert ans.usage == TokenUsage(250, 30)
    assert ans.tool_trace[0].content["data"] == {"value": 6} and ans.tool_trace[0].rows_used == 3
    assert ans.text.startswith("結果は6です") and ans.text.endswith(DISCLAIMERS[Locale.JA])
    assert seen == [USER]  # 認証済みのuser_idがツールに渡る
    [log] = logs
    assert (log.tool_calls, log.llm_calls, log.total_tokens, log.tools) == (1, 2, 280, ["double"])
    assert log.duration_ms >= 0 and log.question_chars == 4


def test_english_disclaimer_and_history_are_sent() -> None:
    llm = ScriptedLLMProvider([text_response("ok")])
    agent, _ = _agent(llm)
    ans = agent.ask(USER, "q2", locale=Locale.EN, history=[("q1", "a1")])
    assert ans.text.endswith(DISCLAIMERS[Locale.EN])
    system, messages, _ = llm.requests[0]
    assert [m.text for m in messages] == ["q1", "a1", "q2"]
    assert "2026-11-07" in system and "English" in system


def test_tool_budget_is_enforced_and_final_answer_is_forced_without_tools() -> None:
    def always_call(system: str, messages: Any, tools: Any):  # type: ignore[no-untyped-def]
        if not tools:
            return text_response("途中までの結果です")
        return tool_call_response(ToolCall(f"c{len(messages)}", "double", {"n": 1}))

    llm = CallbackLLMProvider(always_call)
    agent, logs = _agent(llm, max_calls=3)
    ans = agent.ask(USER, "ずっと呼ぶ")
    assert ans.tool_calls == 3  # 上限を超えて実行されない
    assert ans.status == "max_tool_calls" and "途中までの結果" in ans.text
    assert llm.calls == 4  # 3回のツール要求 + ツール無しの最終回答
    assert logs[0].status == "max_tool_calls"


def test_multiple_calls_in_one_turn_count_individually_and_overflow_gets_error() -> None:
    calls = tuple(ToolCall(f"c{i}", "double", {"n": 1}) for i in range(4))
    llm = ScriptedLLMProvider([tool_call_response(*calls), text_response("done")])
    agent, _ = _agent(llm, max_calls=2)
    ans = agent.ask(USER, "q")
    assert ans.tool_calls == 2
    results = llm.requests[1][1][-1].tool_results
    assert len(results) == 4 and results[2].content["ok"] is False  # 超過分はエラーで返す


def test_tool_errors_are_fed_back_so_the_llm_can_correct_itself() -> None:
    llm = ScriptedLLMProvider(
        [
            tool_call_response(ToolCall("c1", "double", {"n": "x"})),
            tool_call_response(ToolCall("c2", "double", {"n": 2})),
            text_response("修正して取得できました"),
        ]
    )
    agent, _ = _agent(llm)
    ans = agent.ask(USER, "q")
    assert [t.ok for t in ans.tool_trace] == [False, True]
    fed_back = llm.requests[1][1][-1].tool_results[0].content
    assert fed_back["ok"] is False and "invalid arguments" in fed_back["error"]
    assert ans.status == "ok"


def test_llm_failure_becomes_a_polite_answer_not_an_exception() -> None:
    def fail(system: str, messages: Any, tools: Any):  # type: ignore[no-untyped-def]
        raise AIServiceUnavailableError("503")

    agent, logs = _agent(CallbackLLMProvider(fail))
    ans = agent.ask(USER, "q")
    assert ans.status == "llm_error" and "AIサービス" in ans.text
    assert ans.text.endswith(DISCLAIMERS[Locale.JA]) and logs[0].status == "llm_error"


def test_empty_model_answer_is_reported() -> None:
    agent, _ = _agent(ScriptedLLMProvider([text_response("")]))
    assert agent.ask(USER, "q").status == "empty"


def test_nothing_identifying_is_sent_to_the_llm() -> None:
    """LLMに送る内容にuser_idやメールアドレスが含まれない(データ最小化)。"""
    llm = ScriptedLLMProvider(
        [tool_call_response(ToolCall("c1", "double", {"n": 1})), text_response("ok")]
    )
    agent, _ = _agent(llm)
    agent.ask(USER, "睡眠について教えて")
    sent = repr(llm.requests)
    assert str(USER) not in sent and "@" not in sent


def test_query_log_does_not_contain_the_question_text() -> None:
    agent, logs = _agent(ScriptedLLMProvider([text_response("ok")]))
    agent.ask(USER, "私のひみつの質問")
    assert "ひみつ" not in repr(logs[0].to_json())
