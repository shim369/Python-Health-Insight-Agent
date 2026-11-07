"""エージェントループ: LLM ⇄ ツールを、上限回数つきで回す。

設計:
- ツール呼び出しの最大回数は設定値(AgentConfig.max_tool_calls)。超えたら以降のツール要求には
  「予算超過」エラーを返し、最後にツール無しで1回だけ答えさせる(無限ループ・コスト暴走の防止)。
- ツールの失敗は例外にせず、結果としてLLMに返して自己修正させる(Registry側で実施)。
- LLM自体の失敗(429/503の再試行後、タイムアウト等)は、利用者向けの丁寧なエラー回答にする。
- 回答末尾の注意書きはコードが付ける(prompts.DISCLAIMERS)。
- LLMには質問文と(必要なら)直近の会話テキストしか送らない。メールアドレス等の個人識別情報や
  user_id自体は送らない(user_idはツール実行側が保持する)。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from uuid import UUID

from health_insight_agent.agent.prompts import DISCLAIMERS, build_system_prompt
from health_insight_agent.agent.usage_log import QueryLog, QuerySink, logging_sink
from health_insight_agent.ai.errors import AIProviderError
from health_insight_agent.ai.protocols import LLMProvider, Message, TokenUsage, ToolResultPart
from health_insight_agent.domain.enums import Locale
from health_insight_agent.local_time import local_today
from health_insight_agent.logging_config import get_logger
from health_insight_agent.tools.base import ToolExecution, ToolRegistry

logger = get_logger(__name__)

_LLM_ERROR_TEXT = {
    Locale.JA: "AIサービスに接続できず、回答を生成できませんでした。"
    "しばらくしてからもう一度お試しください。",
    Locale.EN: "I could not reach the AI service, so I could not produce an answer. "
    "Please try again in a moment.",
}
_EMPTY_TEXT = {
    Locale.JA: "回答を生成できませんでした。質問を言い換えてもう一度お試しください。",
    Locale.EN: "I could not produce an answer. Please rephrase and try again.",
}


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_tool_calls: int = 8
    """1質問あたりのツール呼び出し回数の上限(コスト・レイテンシ管理)。"""
    extra_rules: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    text: str
    """注意書き込みの最終回答。"""
    status: str
    """ok | max_tool_calls | llm_error | empty"""
    tool_trace: list[ToolExecution] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0
    duration_ms: float = 0.0
    model: str = ""

    @property
    def tool_calls(self) -> int:
        return len(self.tool_trace)


class HealthInsightAgent:
    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        *,
        config: AgentConfig | None = None,
        today: Callable[[], date] = local_today,
        sink: QuerySink = logging_sink,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._config = config or AgentConfig()
        self._today = today
        self._sink = sink

    @property
    def max_tool_calls(self) -> int:
        return self._config.max_tool_calls

    def ask(
        self,
        user_id: UUID,
        question: str,
        *,
        locale: Locale = Locale.JA,
        history: Sequence[tuple[str, str]] = (),
    ) -> AgentAnswer:
        """`history`は直近の(質問, 回答)テキストのみ。ツール結果は含めない(トークン節約)。"""
        started = time.perf_counter()
        system = build_system_prompt(
            today=self._today(), locale=locale, extra_rules=self._config.extra_rules
        )
        messages: list[Message] = []
        for q, a in history:
            messages.append(Message("user", q))
            messages.append(Message("assistant", a))
        messages.append(Message("user", question))
        specs = self._registry.specs()

        trace: list[ToolExecution] = []
        usage = TokenUsage()
        llm_calls = 0
        model = ""
        status = "ok"
        text = ""
        budget_exhausted = False

        while True:
            offered = [] if budget_exhausted else specs
            try:
                resp = self._llm.generate(system=system, messages=messages, tools=offered)
            except AIProviderError:
                logger.exception("LLM call failed")
                status = "llm_error"
                text = _LLM_ERROR_TEXT[locale]
                break
            llm_calls += 1
            usage = usage + resp.usage
            model = resp.model or model
            msg = resp.message
            messages.append(msg)
            if not msg.tool_calls or budget_exhausted:
                text = msg.text.strip()
                break

            results: list[ToolResultPart] = []
            for call in msg.tool_calls:
                if len(trace) >= self._config.max_tool_calls:
                    budget_exhausted = True
                    content: dict[str, Any] = {
                        "ok": False,
                        "error": "tool call budget exceeded; answer now using results so far",
                    }
                    results.append(ToolResultPart(call.id, call.name, content))
                    continue
                execution = self._registry.execute(user_id, call.name, call.arguments)
                trace.append(execution)
                results.append(ToolResultPart(call.id, call.name, execution.content))
            messages.append(Message("tool", tool_results=tuple(results)))
            if len(trace) >= self._config.max_tool_calls:
                budget_exhausted = True  # 次の1回はツール無しで答えさせる

        if budget_exhausted and status == "ok":
            status = "max_tool_calls"
        if not text and status in ("ok", "max_tool_calls"):
            status = "empty"
            text = _EMPTY_TEXT[locale]
        full_text = f"{text}\n\n{DISCLAIMERS[locale]}"
        duration_ms = (time.perf_counter() - started) * 1000
        self._sink(
            QueryLog(
                model=model,
                status=status,
                llm_calls=llm_calls,
                tool_calls=len(trace),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                duration_ms=round(duration_ms, 1),
                tools=[t.name for t in trace],
                question_chars=len(question),
            )
        )
        return AgentAnswer(
            text=full_text,
            status=status,
            tool_trace=trace,
            usage=usage,
            llm_calls=llm_calls,
            duration_ms=duration_ms,
            model=model,
        )
