"""MCP経由のToolExecutor(エージェント側のクライアント)。

エージェントループは同期、MCP SDKは非同期なので、専用スレッドでイベントループを回し、
そこにClientセッションを1本保持する。`execute()`は同期呼び出しをそのループへ投げて結果を待つ。

セキュリティ: サーバーは起動時に1ユーザーへ束縛されている。このクライアントも同じ
`user_id`にだけ使えるようにし、別のuser_idでの呼び出しはサーバーに送らず拒否する。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future
from typing import Any, cast
from uuid import UUID

from mcp import types
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

from health_insight_agent.ai.protocols import ToolSpec
from health_insight_agent.logging_config import get_logger
from health_insight_agent.tools.base import ToolExecution

logger = get_logger(__name__)
CALL_TIMEOUT_SECONDS = 60.0
"""1回のツール呼び出し(DB問い合わせ+MCP往復)の最大待ち時間。"""


class McpToolExecutor:
    def __init__(
        self,
        server: Any,
        user_id: UUID,
        *,
        connect_timeout_seconds: float = 30.0,
    ) -> None:
        """`server`はMCP Clientに渡せるもの(StdioServerParameters、テスト用のServer等)。"""
        self._server = server
        self._user_id = user_id
        self._connect_timeout = connect_timeout_seconds
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Client | None = None
        self._specs: list[ToolSpec] = []
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    # ---- 接続の管理(専用スレッドのイベントループ) ----
    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run_loop, name="mcp-client", daemon=True)
            self._thread.start()
        if not self._ready.wait(self._connect_timeout):
            raise RuntimeError("MCP server did not become ready in time")
        if self._error is not None:
            raise RuntimeError(f"MCP connection failed: {self._error}") from self._error

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session())
        except BaseException as e:  # 接続失敗を呼び出し側スレッドへ伝える
            self._error = e
            self._ready.set()
        finally:
            self._loop.close()

    async def _session(self) -> None:
        # Clientのasync with は、入ったタスクと同じタスクで抜ける必要があるため、
        # 1つのタスク内で「接続→停止待ち→切断」をまとめて行う。
        self._stop = asyncio.Event()
        async with Client(self._server) as client:
            self._client = client
            tools = await client.list_tools()
            self._specs = [
                ToolSpec(t.name, t.description or "", dict(t.input_schema)) for t in tools.tools
            ]
            self._ready.set()
            await self._stop.wait()

    def close(self) -> None:
        loop, stop, thread = self._loop, self._stop, self._thread
        if loop is not None and stop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop.set)
        if thread is not None:
            thread.join(timeout=10)
        self._thread = None

    # ---- ToolExecutor ----
    @property
    def names(self) -> list[str]:
        self._ensure_started()
        return [s.name for s in self._specs]

    def specs(self) -> list[ToolSpec]:
        self._ensure_started()
        return list(self._specs)

    def execute(
        self, user_id: UUID, name: str, arguments: dict[str, Any], locale: str = "ja"
    ) -> ToolExecution:
        started = time.perf_counter()

        def fail(msg: str) -> ToolExecution:
            return ToolExecution(
                name=name,
                arguments=arguments,
                ok=False,
                content={"ok": False, "error": msg},
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        if user_id != self._user_id:
            # サーバーが束縛しているユーザー以外のデータは、そもそも要求しない。
            return fail("user mismatch: this MCP connection is bound to a different user")
        try:
            self._ensure_started()
            assert self._loop is not None and self._client is not None
            future: Future[types.CallToolResult] = asyncio.run_coroutine_threadsafe(
                self._client.call_tool(name, arguments, meta=cast(Any, {"locale": locale})),
                self._loop,
            )
            result = future.result(timeout=CALL_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("MCP tool call failed: %s", name)
            return fail("internal error while calling the tool over MCP")
        return self._to_execution(name, arguments, result, started)

    @staticmethod
    def _to_execution(
        name: str, arguments: dict[str, Any], result: types.CallToolResult, started: float
    ) -> ToolExecution:
        sc = result.structured_content or {}
        content = sc.get("content")
        if not isinstance(content, dict):  # 構造化結果が無いサーバーでもテキストから復元する
            text = next((c.text for c in result.content if isinstance(c, types.TextContent)), "{}")
            try:
                content = json.loads(text)
            except json.JSONDecodeError:
                content = {"ok": False, "error": text[:200]}
        return ToolExecution(
            name=name,
            arguments=arguments,
            ok=not result.is_error,
            content=content,
            duration_ms=float(sc.get("duration_ms", (time.perf_counter() - started) * 1000)),
            rows_used=int(sc.get("rows_used", 0)),
            charts=list(sc.get("charts", [])),
        )


def stdio_params(*, extra_env: dict[str, str] | None = None) -> StdioServerParameters:
    """自分自身のPython(同じ仮想環境)でMCPサーバーを子プロセスとして起動する設定。"""
    import os
    import sys

    env = {**os.environ, **(extra_env or {})}
    return StdioServerParameters(
        command=sys.executable, args=["-m", "health_insight_agent.mcp_server"], env=env
    )
