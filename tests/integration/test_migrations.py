"""マイグレーションがORMと一致しているか。テーブルは消さない(既存アプリとの違い)。"""

from __future__ import annotations

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect

from health_insight_agent.database.connection import create_db_engine
from health_insight_agent.database.schema import Base

pytestmark = pytest.mark.integration


def test_schema_matches_orm_models(migrated_database: str) -> None:
    engine = create_db_engine(migrated_database)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        tables = set(inspect(conn).get_table_names())
    engine.dispose()
    assert diff == []  # モデルを変えたのにmigrationを作り忘れると失敗する
    assert {"users", "sleep_logs", "meal_logs", "workout_sets", "daily_journals"} <= tables
