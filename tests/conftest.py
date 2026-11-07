"""pytest共通設定。

最重要: テストは必ずテスト専用DB(名前が`_test`で終わる)にだけ接続する。
- APP_ENV=test を最初に設定し、config.load_environment が .env(実DB)ではなく
  .env.test を読むようにする。
- DB名が`_test`で終わらなければ、テスト全体を中止する。
(既存アプリでは test_migrations がテーブルを消すため、同種の事故を構造的に防ぐ。)
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from datetime import date, time
from uuid import UUID, uuid4

os.environ["APP_ENV"] = "test"
# テストがリポジトリ直下に学習記録(mlruns/ml_artifacts)を書き散らさないようにする
os.environ["ML_TRACKING"] = "off"
os.environ["ML_ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="hia-ml-")


import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.orm import Session

from health_insight_agent.config import (
    PROJECT_ROOT,
    assert_test_database,
    load_settings,
)
from health_insight_agent.database.connection import create_db_engine, create_session_factory
from health_insight_agent.domain.enums import Locale
from health_insight_agent.domain.models import User
from health_insight_agent.local_time import local_datetime

FIXED_END_DATE = date(2026, 11, 7)
"""合成データの最終日。今日の日付に依存させず、テストを再現可能にする。"""


@pytest.fixture(scope="session")
def database_url() -> str:
    url = load_settings().database_url
    try:
        assert_test_database(url)
    except RuntimeError as e:
        pytest.exit(str(e), returncode=2)
    return url


@pytest.fixture(scope="session")
def migrated_database(database_url: str) -> str:
    """テストDBを最新スキーマにする(downgrade/dropはしない)。"""
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(cfg, "head")
    return database_url


@pytest.fixture
def session(migrated_database: str) -> Iterator[Session]:
    """テストごとに1トランザクション。終了時にrollbackするのでDBは汚れない。"""
    engine = create_db_engine(migrated_database)
    with engine.connect() as conn:
        outer = conn.begin()
        sess = create_session_factory(engine)(bind=conn)
        try:
            yield sess
        finally:
            sess.close()
            outer.rollback()
    engine.dispose()


def make_user(session: Session, *, email: str | None = None) -> UUID:
    from health_insight_agent.repositories.sqlalchemy.repositories import (
        SqlAlchemyUserRepository,
    )

    uid = uuid4()
    SqlAlchemyUserRepository(session).add(
        User(
            user_id=uid,
            email=email or f"{uid}@example.com",
            display_name="t",
            created_at=local_datetime(FIXED_END_DATE, time.min),
            locale=Locale.JA,
        )
    )
    session.flush()
    return uid


@pytest.fixture(scope="module")
def seeded(migrated_database: str) -> Iterator[tuple[object, UUID, UUID]]:
    """実際にcommitした合成データ(2ユーザー)。終了時に削除する(テスト専用DBなので安全)。

    戻り値: (uow_factory, user_id, other_user_id)。other_user_idは少量の別データを持ち、
    ユーザー分離の確認に使う。
    """
    from health_insight_agent.repositories.sqlalchemy.factory import build_uow_factory
    from health_insight_agent.synthetic.generator import generate_dataset

    factory = build_uow_factory(migrated_database)
    ids: list[UUID] = []
    for seed, days in ((42, 90), (7, 20)):
        uid = uuid4()
        ids.append(uid)
        ds = generate_dataset(uid, end_date=FIXED_END_DATE, days=days, seed=seed)
        with factory() as uow:
            uow.users.add(replace(ds.user, email=f"{uid}@example.com"))
            uow.commit()
            uow.sleep_logs.add_many(ds.sleep_logs)
            uow.meal_logs.add_many(ds.meal_logs)
            uow.workout_logs.add_many(ds.workout_logs)
            uow.health_logs.add_many(ds.health_logs)
            uow.weight_logs.add_many(ds.weight_logs)
            uow.daily_journals.add_many(ds.daily_journals)
            uow.commit()
    yield factory, ids[0], ids[1]
    from sqlalchemy import delete

    from health_insight_agent.database import schema as orm

    for uid in ids:
        with factory() as uow:
            uow.purge_user_data(uid)
            uow._session.execute(delete(orm.UserORM).where(orm.UserORM.id == uid))  # type: ignore[union-attr]
            uow.commit()
