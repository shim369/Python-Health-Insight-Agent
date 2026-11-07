"""MCPサーバー/クライアントの単体テスト(DB不要・インメモリ接続)。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest

from health_insight_agent.mcp_server.client import McpToolExecutor
from health_insight_agent.mcp_server.server import build_mcp_server
from health_insight_agent.tools import charts
from health_insight_agent.tools.base import Tool, ToolArgs, ToolOutput, ToolRegistry

BOUND = UUID(int=1)


class _Args(ToolArgs):
    n: int = 1


def _registry(seen: list[UUID]) -> ToolRegistry:
    def handler(user_id: UUID, a: _Args) -> ToolOutput:
        seen.append(user_id)
        return ToolOutput(
            data={"v": a.n * 2},
            rows_used=a.n,
            period="p",
            charts=[charts.bar_chart("t", [("a", 1.0)], y_label="y")],
        )

    return ToolRegistry([Tool("double", "doubles n", _Args, handler)])


@pytest.fixture
def pair() -> Iterator[tuple[McpToolExecutor, ToolRegistry, list[UUID]]]:
    seen: list[UUID] = []
    reg = _registry(seen)
    ex = McpToolExecutor(build_mcp_server(reg, BOUND), BOUND)
    yield ex, reg, seen
    ex.close()


def test_specs_match_the_direct_registry(pair: Any) -> None:
    ex, reg, _ = pair
    assert ex.names == reg.names == ["double"]
    assert ex.specs() == reg.specs()  # LLMに見せる定義が直接呼び出しと完全に同じ


def test_execute_matches_direct_result_including_charts(pair: Any) -> None:
    ex, reg, seen = pair
    via_mcp = ex.execute(BOUND, "double", {"n": 3})
    direct = reg.execute(BOUND, "double", {"n": 3})
    assert via_mcp.ok and via_mcp.content == direct.content
    assert via_mcp.rows_used == direct.rows_used == 3
    assert via_mcp.charts == direct.charts  # グラフはLLMとは別経路でUIに届く
    assert seen == [BOUND, BOUND]


def test_user_is_bound_server_side_and_mismatch_is_refused(pair: Any) -> None:
    ex, _, seen = pair
    res = ex.execute(UUID(int=2), "double", {"n": 1})
    assert not res.ok and "user mismatch" in res.content["error"]
    assert seen == []  # サーバーには届いていない
    # クライアント引数でuser_idを渡そうとしても、サーバー側の検証で拒否される
    bad = ex.execute(BOUND, "double", {"n": 1, "user_id": str(UUID(int=2))})
    assert not bad.ok and "user_id" in bad.content["error"] and seen == []


def test_tool_errors_are_returned_as_errors(pair: Any) -> None:
    ex, _, _ = pair
    res = ex.execute(BOUND, "double", {"n": "x"})
    assert not res.ok and "invalid arguments" in res.content["error"]
    assert not ex.execute(BOUND, "nope", {}).ok


def test_close_stops_the_background_thread(pair: Any) -> None:
    ex, _, _ = pair
    ex.specs()
    thread = ex._thread
    assert thread is not None and thread.is_alive()
    ex.close()
    assert not thread.is_alive()


def test_connection_failure_is_reported_not_hung() -> None:
    from mcp.client.stdio import StdioServerParameters

    ex = McpToolExecutor(
        StdioServerParameters(command="/nonexistent/binary", args=[]),
        BOUND,
        connect_timeout_seconds=5,
    )
    res = ex.execute(BOUND, "double", {})
    assert not res.ok
    ex.close()
