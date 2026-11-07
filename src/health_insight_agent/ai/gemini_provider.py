"""Gemini API(google-genai SDK)のLLMProvider実装。

- Retry/Timeoutは RetryPolicy に従う(最大3回・1秒×2倍・±20%・1回20秒・429/503のみ)。
  SDK内蔵のリトライは無効化(attempts=1)し、自前ループだけにする(二重リトライ防止)。
- 関数呼び出しは自動実行を切り(AFC disable)、ツール実行はエージェントループが行う。
  ツール実行・回数上限・ログを自前で制御するため。
- Gemini 3は関数呼び出しのthought signatureを履歴に戻す必要がある。assistantのMessageには
  生のContentを `provider_payload` として持たせ、そのまま再送する。
- SDKの例外の分類は `APIError.code` に基づく。実キーでの疎通はREADMEの手順で確認すること。
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Sequence
from typing import Any, cast

from google import genai
from google.genai import errors, types

from health_insight_agent.ai.errors import (
    AIProviderError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
    AIValidationError,
)
from health_insight_agent.ai.protocols import (
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from health_insight_agent.ai.retry import call_with_retry
from health_insight_agent.ai.retry_policy import DEFAULT_RETRY_POLICY, RetryPolicy
from health_insight_agent.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "gemini-3.6-flash"
"""安定版(stable)のFlash。公式モデル一覧で確認済み(2026-11)。GEMINI_MODEL環境変数で変更可。"""


def _is_timeout(e: BaseException) -> bool:
    """SDKが使うHTTPライブラリ(httpx/httpx2等)に依存せずタイムアウトを判定する。

    google-genai 3.xはhttpx2を使うため、特定ライブラリの例外型をimportせず、
    継承階層のクラス名で判定する。
    """
    return isinstance(e, TimeoutError) or any("Timeout" in cls.__name__ for cls in type(e).__mro__)


def _to_contents(messages: Sequence[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for m in messages:
        if m.role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m.text)]))
        elif m.role == "assistant":
            if isinstance(m.provider_payload, types.Content):
                contents.append(m.provider_payload)  # thought signatureごと再送
                continue
            parts: list[types.Part] = []
            if m.text:
                parts.append(types.Part(text=m.text))
            parts.extend(
                types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.arguments))
                for c in m.tool_calls
            )
            contents.append(types.Content(role="model", parts=parts))
        else:  # tool
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=r.call_id, name=r.name, response=r.content
                            )
                        )
                        for r in m.tool_results
                    ],
                )
            )
    return contents


def _to_tools(tools: Sequence[ToolSpec]) -> list[types.Tool]:
    if not tools:
        return []
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters_schema,
                )
                for t in tools
            ]
        )
    ]


def _parse_response(response: Any, model: str) -> LLMResponse:
    candidates = getattr(response, "candidates", None) or []
    if not candidates or candidates[0].content is None:
        raise AIValidationError("Gemini returned no content (blocked or empty response)")
    content: types.Content = candidates[0].content
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for i, part in enumerate(content.parts or []):
        if part.function_call is not None:
            fc = part.function_call
            calls.append(
                ToolCall(id=fc.id or f"call_{i}", name=fc.name or "", arguments=dict(fc.args or {}))
            )
        elif part.text and not part.thought:
            text_parts.append(part.text)
    usage_md = getattr(response, "usage_metadata", None)
    usage = TokenUsage(
        input_tokens=int(getattr(usage_md, "prompt_token_count", 0) or 0),
        # 思考トークンも課金対象のため出力側に含めて、コストを過小評価しない。
        output_tokens=int(getattr(usage_md, "candidates_token_count", 0) or 0)
        + int(getattr(usage_md, "thoughts_token_count", 0) or 0),
    )
    message = Message(
        role="assistant",
        text="".join(text_parts),
        tool_calls=tuple(calls),
        provider_payload=content,
    )
    return LLMResponse(message=message, usage=usage, model=model)


class GeminiProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        temperature: float = 0.2,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        """`client`/`sleep`/`rng`はテスト用の差し込み口(実APIを呼ばずにリトライを検証する)。"""
        self._model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self._policy = retry_policy
        self._temperature = temperature
        self._sleep = sleep
        self._rng = rng or random.Random()
        if client is not None:
            self._client = client
            return
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY が未設定です(.env.example を参照)。")
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1),
                timeout=int(retry_policy.request_timeout_seconds * 1000),
            ),
        )

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=cast(Any, _to_tools(tools) or None),
            temperature=self._temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = _to_contents(messages)
        return call_with_retry(
            lambda: self._call_once(contents, config),
            policy=self._policy,
            sleep=self._sleep,
            rng=self._rng,
            label="Gemini generate",
        )

    def _call_once(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> LLMResponse:
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except errors.APIError as e:
            if e.code == 429:
                raise AIRateLimitError(str(e)) from e
            if e.code == 503:
                raise AIServiceUnavailableError(str(e)) from e
            raise AIProviderError(f"Gemini API error {e.code}: {e}") from e
        except Exception as e:
            if _is_timeout(e):
                raise AITimeoutError(str(e)) from e
            raise AIProviderError(f"Gemini call failed: {e}") from e
        return _parse_response(response, self._model)
