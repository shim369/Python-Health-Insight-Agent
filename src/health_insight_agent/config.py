"""設定の読み込み。環境変数(ローカルは.env)から組み立てる。秘密情報はコードに書かない。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 認証は当面、固定のデモuser_idで代用する(統合時はREST API + Google OAuth。ADRで論点化)。
DEFAULT_DEMO_USER_ID = UUID("00000000-0000-4000-8000-000000000001")


def load_environment() -> None:
    """APP_ENV=test のときは .env.test、それ以外は .env を読む。

    テストが誤って本番/開発の`.env`(実DB)を読まないよう、ファイルを完全に分ける。
    すでに設定済みの環境変数は上書きしない(CIの値を優先するため)。
    """
    name = ".env.test" if os.environ.get("APP_ENV") == "test" else ".env"
    load_dotenv(dotenv_path=PROJECT_ROOT / name, override=False)


@dataclass(frozen=True, kw_only=True, slots=True)
class Settings:
    database_url: str
    readonly_database_url: str | None
    demo_user_id: UUID
    app_env: str
    ai_provider: str
    gemini_model: str | None
    tool_transport: str
    sql_tool_enabled: bool
    """direct=関数を直接呼ぶ / mcp=MCPサーバー(子プロセス)経由で呼ぶ。"""
    log_level: str


def load_settings() -> Settings:
    load_environment()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("環境変数 DATABASE_URL が未設定です(.env.example を参照)。")
    return Settings(
        database_url=database_url,
        # 既存アプリ(Neon)を読むときは、書き込み権限の無い専用ユーザーのURLだけを使う。
        readonly_database_url=os.environ.get("READONLY_DATABASE_URL") or None,
        demo_user_id=UUID(os.environ.get("DEMO_USER_ID", str(DEFAULT_DEMO_USER_ID))),
        app_env=os.environ.get("APP_ENV", "dev"),
        ai_provider=os.environ.get("AI_PROVIDER", "gemini").lower(),
        gemini_model=os.environ.get("GEMINI_MODEL") or None,
        tool_transport=os.environ.get("TOOL_TRANSPORT", "direct").lower(),
        sql_tool_enabled=os.environ.get("SQL_TOOL", "on").lower() not in ("off", "0", "false"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


def assert_test_database(url: str) -> None:
    """DB名が`_test`で終わらなければ例外。テストが実DBに触れる事故を防ぐ最後の砦。

    既存アプリでは test_migrations がテーブルを削除するため、同様の事故を構造的に防ぐ。
    """
    database = make_url(url).database or ""
    if not database.endswith("_test"):
        raise RuntimeError(
            f"テスト用DB名は '_test' で終わる必要があります(指定: {database!r})。"
            " 実DBには接続しません。"
        )
