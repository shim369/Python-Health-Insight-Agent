"""read-only reader role for the guarded SQL tool

SQLツール用の読み取り専用グループロール(NOLOGIN)を作り、許可した表だけにSELECTを与える。
usersテーブル(メールアドレス等)は含めない。ログインできるユーザーは、このロールのメンバーとして
別途作成する(`python -m health_insight_agent.database.readonly_role`)。パスワードはmigrationに
書かない。

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

ROLE = "health_insight_reader"
TABLES = (
    "meal_logs, sleep_logs, workout_logs, workout_exercises, workout_sets, "
    "health_logs, weight_logs, daily_journals"
)


def upgrade() -> None:
    # ロール作成権限が無い環境(マネージドDB等)でもmigration全体は失敗させず、通知だけ出す。
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{ROLE}') THEN
                CREATE ROLE {ROLE} NOLOGIN;
            END IF;
            GRANT USAGE ON SCHEMA public TO {ROLE};
            GRANT SELECT ON {TABLES} TO {ROLE};
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'skipped creating {ROLE}: insufficient privilege';
        END
        $$;
        """
    )


def downgrade() -> None:
    # ロールはクラスタ全体で共有され他DBの権限も持つため、DROPせず権限だけ外す。
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{ROLE}') THEN
                REVOKE SELECT ON {TABLES} FROM {ROLE};
            END IF;
        END
        $$;
        """
    )
