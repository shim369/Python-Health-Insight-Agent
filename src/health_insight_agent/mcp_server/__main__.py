"""MCPサーバー(stdio)の起動: uv run python -m health_insight_agent.mcp_server

stdoutはMCPのプロトコル専用。ログは必ずstderrへ出す(logging_configの既定がstderr)。
環境変数: MCP_USER_ID(束縛するユーザー。既定はDEMO_USER_ID)、MCP_TODAY(YYYY-MM-DD、評価用)。
"""

from __future__ import annotations

import os
from datetime import date
from uuid import UUID

import anyio

from health_insight_agent.agent.factory import build_registry
from health_insight_agent.config import load_settings
from health_insight_agent.mcp_server.server import serve_stdio


def main() -> None:
    settings = load_settings()
    today = date.fromisoformat(os.environ["MCP_TODAY"]) if os.environ.get("MCP_TODAY") else None
    user_id = UUID(os.environ["MCP_USER_ID"]) if os.environ.get("MCP_USER_ID") else None
    anyio.run(serve_stdio, build_registry(settings, today=today), user_id or settings.demo_user_id)


if __name__ == "__main__":
    main()
