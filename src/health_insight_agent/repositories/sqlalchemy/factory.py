"""設定からUnitOfWork factoryを組み立てる。"""

from __future__ import annotations

from collections.abc import Callable

from health_insight_agent.database.connection import create_db_engine, create_session_factory
from health_insight_agent.repositories.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork


def build_uow_factory(url: str, *, read_only: bool = False) -> Callable[[], SqlAlchemyUnitOfWork]:
    """`url`に接続するUoW factoryを返す。エージェントは read_only=True で使う。"""
    session_factory = create_session_factory(create_db_engine(url))
    return lambda: SqlAlchemyUnitOfWork(session_factory, read_only=read_only)
