"""LLMProviderのProtocolと、プロバイダ非依存のメッセージ型。

エージェントループはこの型だけに依存する。Gemini固有の型(google.genai.types)は
gemini_provider.py の内側に閉じ込め、Fake/将来の別プロバイダと差し替えられるようにする。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """LLMに見せるツール定義(名前・説明・引数のJSON Schema)。"""

    name: str
    description: str
    parameters_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultPart:
    call_id: str
    name: str
    content: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResultPart, ...] = ()
    provider_payload: Any = None
    """プロバイダ固有の生データ。Gemini 3は関数呼び出しの「thought signature」を含む
    モデルの出力をそのまま履歴に戻す必要があるため、assistantメッセージでは生のContentを保持して
    再送する。ループ側は中身を解釈しない。"""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    message: Message
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""


class LLMProvider(Protocol):
    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        """1ターン分の生成。ツール呼び出しを要求する場合は message.tool_calls が入る。

        リトライ(429/503)とタイムアウトは実装側の責務。失敗時は ai/errors.py の例外を送出。
        """
        ...


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """テキストごとの埋め込みベクトル(Phase 5のRAGで使用)。"""
        ...
