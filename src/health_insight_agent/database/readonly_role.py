"""読み取り専用のログインユーザーを作る(管理者接続が必要)。

    READONLY_PASSWORD=... uv run python -m health_insight_agent.database.readonly_role

DATABASE_URL(管理者権限)で接続し、`health_insight_reader`(migration 0002)のメンバーとして
ログインユーザーを作る。パスワードは環境変数から読み、出力には含めない。
"""

from __future__ import annotations

import argparse
import os

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url

from health_insight_agent.config import load_settings

GROUP_ROLE = "health_insight_reader"


def _raw_dsn(url: str) -> str:
    return make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)


def create_login_role(admin_url: str, user: str, password: str) -> None:
    """ログインロールを作成(既にあればパスワードだけ更新)。識別子・値はpsycopgで安全に組み立てる。"""
    with psycopg.connect(_raw_dsn(admin_url), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,)).fetchone()
        if exists:
            stmt = sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(user), sql.Literal(password)
            )
        else:
            stmt = sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} IN ROLE {}").format(
                sql.Identifier(user), sql.Literal(password), sql.Identifier(GROUP_ROLE)
            )
        conn.execute(stmt)


def drop_login_role(admin_url: str, user: str) -> None:
    with psycopg.connect(_raw_dsn(admin_url), autocommit=True) as conn:
        conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(user)))


def readonly_url(admin_url: str, user: str, password: str) -> str:
    """管理者URLのユーザー/パスワードだけを読み取り専用ユーザーに差し替えたURL。"""
    return make_url(admin_url).set(username=user, password=password).render_as_string(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="読み取り専用ログインユーザーを作成")
    parser.add_argument("--user", default="agent_ro")
    args = parser.parse_args()
    password = os.environ.get("READONLY_PASSWORD")
    if not password:
        raise SystemExit("READONLY_PASSWORD 環境変数を設定してください。")
    settings = load_settings()
    create_login_role(settings.database_url, args.user, password)
    shown = readonly_url(settings.database_url, args.user, "<password>")
    print(f"created/updated role {args.user!r}.\nREADONLY_DATABASE_URL={shown}")


if __name__ == "__main__":
    main()
