"""Alembic環境。接続先は設定(DATABASE_URL)から取得する。"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from health_insight_agent.config import load_settings
from health_insight_agent.database.schema import Base

target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = create_engine(load_settings().database_url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
