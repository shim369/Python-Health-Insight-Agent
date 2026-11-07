"""MCPサーバー: ToolRegistryの全ツールをMCPで公開する。

セキュリティ上の約束(直接呼び出しと同じ):
- `user_id`はクライアントの引数からは受け取らない。サーバー起動時に束縛したユーザー
  (MCP_USER_ID、既定はデモユーザー)のデータだけを返す。ツールの引数スキーマにもuser_idは無い。
- 読み取り専用。書き込み系のツールは存在しない。
結果は content にLLM向けのJSONテキスト、structured_content に件数・グラフ等のメタ情報を載せる
(グラフはLLMに渡さずUIにだけ届けるため、別枠で運ぶ)。
"""

from __future__ import annotations

import json
from uuid import UUID

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from health_insight_agent.tools.executor import ToolExecutor

SERVER_NAME = "health-insight-agent"


def build_mcp_server(executor: ToolExecutor, user_id: UUID) -> Server:
    async def on_list_tools(
        ctx: object, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=s.name, description=s.description, input_schema=s.parameters_schema)
                for s in executor.specs()
            ]
        )

    async def on_call_tool(
        ctx: object, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        # ツールは同期(DB待ち)。イベントループを塞がないようスレッドで実行する。
        execution = await anyio.to_thread.run_sync(
            executor.execute, user_id, params.name, dict(params.arguments or {})
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(execution.content))],
            structured_content={
                "content": execution.content,
                "rows_used": execution.rows_used,
                "duration_ms": execution.duration_ms,
                "charts": execution.charts,
            },
            is_error=not execution.ok,
        )

    return Server(SERVER_NAME, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve_stdio(executor: ToolExecutor, user_id: UUID) -> None:
    server = build_mcp_server(executor, user_id)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
