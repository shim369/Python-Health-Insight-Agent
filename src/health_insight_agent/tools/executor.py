"""ツール実行の抽象(直接呼び出し / MCP経由を切り替えるための境界)。"""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from health_insight_agent.ai.protocols import ToolSpec
from health_insight_agent.tools.base import ToolExecution


class ToolExecutor(Protocol):
    """エージェントループが依存するインターフェース。ToolRegistry(直接)とMcpToolExecutor(MCP)が実装。"""

    @property
    def names(self) -> list[str]: ...

    def specs(self) -> list[ToolSpec]: ...

    def execute(
        self, user_id: UUID, name: str, arguments: dict[str, Any], locale: str = "ja"
    ) -> ToolExecution: ...

    def close(self) -> None: ...
