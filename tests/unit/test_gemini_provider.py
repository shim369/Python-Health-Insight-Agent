"""GeminiProviderを実APIなしで検証する(クライアントを差し込む)。"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors, types

from health_insight_agent.ai.errors import (
    AIProviderError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
    AIValidationError,
)
from health_insight_agent.ai.gemini_provider import GeminiProvider
from health_insight_agent.ai.protocols import Message, ToolCall, ToolResultPart, ToolSpec
from health_insight_agent.ai.retry_policy import DEFAULT_RETRY_POLICY, RetryPolicy


def _ok_response(text: str = "hello", *, call: tuple[str, dict[str, Any]] | None = None) -> Any:
    parts = [types.Part(text=text)] if text else []
    if call:
        parts.append(types.Part(function_call=types.FunctionCall(name=call[0], args=call[1])))
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5
        ),
    )


class _Models:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def _provider(outcomes: list[Any]) -> tuple[GeminiProvider, _Models, list[float]]:
    models = _Models(outcomes)
    sleeps: list[float] = []
    p = GeminiProvider(
        client=SimpleNamespace(models=models),
        sleep=sleeps.append,
        rng=random.Random(0),
        model="test-model",
    )
    return p, models, sleeps


def _gen(p: GeminiProvider) -> Any:
    return p.generate(system="sys", messages=[Message("user", "q")], tools=[])


def test_retry_policy_matches_existing_app_except_timeout() -> None:
    p = DEFAULT_RETRY_POLICY
    assert (p.max_attempts, p.base_delay_seconds, p.backoff_multiplier, p.jitter_ratio) == (
        3,
        1.0,
        2.0,
        0.2,
    )
    assert p.request_timeout_seconds == 20.0  # 既存は60秒。エージェント用に短縮(決定3)
    assert [p.delay_seconds(1), p.delay_seconds(2)] == [1.0, 2.0]
    with pytest.raises(ValueError):
        p.delay_seconds(0)


def test_retries_429_then_succeeds_with_jittered_backoff() -> None:
    p, models, sleeps = _provider(
        [errors.APIError(429, {}), errors.APIError(503, {}), _ok_response()]
    )
    resp = _gen(p)
    assert resp.message.text == "hello"
    assert len(models.calls) == 3  # 初回+リトライ2回
    assert 0.8 <= sleeps[0] <= 1.2 and 1.6 <= sleeps[1] <= 2.4  # 1秒×2倍、±20%


@pytest.mark.parametrize(
    ("code", "exc"), [(429, AIRateLimitError), (503, AIServiceUnavailableError)]
)
def test_gives_up_after_three_attempts(code: int, exc: type[Exception]) -> None:
    p, models, sleeps = _provider([errors.APIError(code, {})] * 3)
    with pytest.raises(exc):
        _gen(p)
    assert len(models.calls) == 3 and len(sleeps) == 2


def test_other_errors_are_not_retried() -> None:
    p, models, sleeps = _provider([errors.APIError(400, {})])
    with pytest.raises(AIProviderError) as ei:
        _gen(p)
    assert not isinstance(ei.value, AIRateLimitError | AIServiceUnavailableError)
    assert len(models.calls) == 1 and sleeps == []


def test_timeout_is_not_retried() -> None:
    class ReadTimeout(Exception):  # httpx/httpx2どちらの例外でも名前で判定できること
        pass

    p, models, _ = _provider([ReadTimeout("slow")])
    with pytest.raises(AITimeoutError):
        _gen(p)
    assert len(models.calls) == 1


def test_empty_response_is_validation_error() -> None:
    p, _, _ = _provider([SimpleNamespace(candidates=[], usage_metadata=None)])
    with pytest.raises(AIValidationError):
        _gen(p)


def test_parses_text_tool_calls_and_usage_including_thoughts() -> None:
    p, _, _ = _provider([_ok_response("", call=("correlate", {"x_metric": "a"}))])
    resp = _gen(p)
    assert resp.message.tool_calls[0].name == "correlate"
    assert resp.message.tool_calls[0].arguments == {"x_metric": "a"}
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (100, 25)  # 思考トークンを含む
    assert isinstance(resp.message.provider_payload, types.Content)


def test_request_disables_auto_function_calling_and_replays_payload() -> None:
    p, models, _ = _provider([_ok_response("done")])
    payload = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="t", args={}), thought_signature=b"sig"
            )
        ],
    )
    msgs = [
        Message("user", "q"),
        Message("assistant", tool_calls=(ToolCall("c1", "t", {}),), provider_payload=payload),
        Message("tool", tool_results=(ToolResultPart("c1", "t", {"ok": True}),)),
    ]
    spec = ToolSpec("t", "desc", {"type": "object", "properties": {}})
    p.generate(system="sys", messages=msgs, tools=[spec])
    call = models.calls[0]
    assert call["config"].automatic_function_calling.disable is True  # ループ側で実行する
    assert call["contents"][1] is payload  # thought signatureごと再送される
    assert call["contents"][2].parts[0].function_response.name == "t"
    decl = call["config"].tools[0].function_declarations[0]
    assert decl.name == "t" and decl.parameters_json_schema == spec.parameters_schema


def test_missing_api_key_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiProvider(retry_policy=RetryPolicy())
