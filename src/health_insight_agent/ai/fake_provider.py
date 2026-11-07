"""テスト・キー無しの動作確認用LLM。実APIは一切呼ばない。

ScriptedLLMProvider: あらかじめ決めた応答を順に返す(エージェントループの単体テスト用)。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from health_insight_agent.ai.errors import AIProviderError
from health_insight_agent.ai.protocols import (
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

Responder = Callable[[str, Sequence[Message], Sequence[ToolSpec]], LLMResponse]


def text_response(text: str, *, usage: TokenUsage | None = None) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", text=text),
        usage=usage or TokenUsage(10, max(1, len(text) // 4)),
        model="fake",
    )


def tool_call_response(*calls: ToolCall, usage: TokenUsage | None = None) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", tool_calls=tuple(calls)),
        usage=usage or TokenUsage(10, 5),
        model="fake",
    )


@dataclass
class ScriptedLLMProvider:
    """`script`の応答を呼び出し順に返す。足りなくなったらエラー(テストの想定ズレを検知)。

    `requests`に受け取った入力を記録するので、「何をLLMに送ったか」(例: 個人識別情報が
    含まれていないか)をテストで検査できる。
    """

    script: list[LLMResponse]
    requests: list[tuple[str, list[Message], list[ToolSpec]]] = field(default_factory=list)
    _index: int = 0

    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        self.requests.append((system, list(messages), list(tools)))
        if self._index >= len(self.script):
            raise AIProviderError("ScriptedLLMProvider: script exhausted")
        response = self.script[self._index]
        self._index += 1
        return response


@dataclass
class CallbackLLMProvider:
    """呼び出しごとに関数で応答を決める版(入力に応じて分岐したいテスト用)。"""

    responder: Responder
    calls: int = 0

    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        self.calls += 1
        return self.responder(system, messages, tools)
