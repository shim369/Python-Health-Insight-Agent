# ADR(設計判断の記録)

| # | 判断 |
|---|---|
| [0001](0001-llm-explains-tools-compute.md) | LLMは説明、数値と判定はツール |
| [0002](0002-tools-bind-user-server-side.md) | ツールは`user_id`をLLMから受け取らない |
| [0003](0003-guarded-readonly-sql.md) | 自由SQLは多層防御とユーザー絞り込みの強制 |
| [0004](0004-tool-transport-direct-or-mcp.md) | ツールの呼び方を直接/MCPで切り替える |
| [0005](0005-existing-app-integration.md) | 既存アプリとの統合方針(認証の論点を含む) |
| [0006](0006-rag-vectorstore-protocol.md) | RAGはVectorStoreをProtocolにしてNumPyで実装 |
| [0007](0007-python-3-15-1.md) | Python 3.15.1の採用とmlflowの注意点 |
| [0008](0008-retry-timeout-20s.md) | Geminiのタイムアウトは20秒(既存は60秒) |
