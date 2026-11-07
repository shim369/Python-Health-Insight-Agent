"""1質問ごとの使用量ログ(トークン数・ツール呼び出し回数・所要時間)。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from health_insight_agent.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QueryLog:
    model: str
    status: str
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    duration_ms: float
    tools: list[str] = field(default_factory=list)
    question_chars: int = 0
    """質問の文字数のみ記録する(質問本文は個人情報になり得るためログに残さない)。"""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "total_tokens": self.total_tokens}


QuerySink = Callable[[QueryLog], None]


def logging_sink(log: QueryLog) -> None:
    logger.info("agent_query %s", json.dumps(log.to_json(), ensure_ascii=False))


class JsonlSink:
    """追記型のJSONLファイル(評価・コスト集計用)。logs/はgit管理外。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, log: QueryLog) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log.to_json(), ensure_ascii=False) + "\n")
