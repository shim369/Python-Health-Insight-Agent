"""Repositoryの結合テスト(テスト専用DB。各テストはrollbackされる)。"""

from __future__ import annotations

from datetime import date, time, timedelta
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from health_insight_agent.domain.enums import ExerciseCategory
from health_insight_agent.domain.models import WorkoutExercise, WorkoutLog, WorkoutSet
from health_insight_agent.local_time import local_datetime, local_day_range
from health_insight_agent.repositories.sqlalchemy import repositories as r
from health_insight_agent.repositories.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from health_insight_agent.synthetic.generator import SyntheticDataset, generate_dataset
from tests.conftest import FIXED_END_DATE, make_user

pytestmark = pytest.mark.integration


def _load(session: Session, user_id: UUID, *, seed: int = 1) -> SyntheticDataset:
    ds = generate_dataset(user_id, end_date=FIXED_END_DATE, days=30, seed=seed)
    r.SqlAlchemySleepLogRepository(session).add_many(ds.sleep_logs)
    r.SqlAlchemyMealLogRepository(session).add_many(ds.meal_logs)
    r.SqlAlchemyWorkoutLogRepository(session).add_many(ds.workout_logs)
    r.SqlAlchemyHealthLogRepository(session).add_many(ds.health_logs)
    r.SqlAlchemyWeightLogRepository(session).add_many(ds.weight_logs)
    r.SqlAlchemyDailyJournalRepository(session).add_many(ds.daily_journals)
    session.flush()
    return ds


def test_roundtrip_and_ordering(session: Session) -> None:
    uid = make_user(session)
    ds = _load(session, uid)
    start, end = local_day_range(FIXED_END_DATE - timedelta(days=29), FIXED_END_DATE)

    sleeps = r.SqlAlchemySleepLogRepository(session).list_between(uid, start, end)
    assert [s.sleep_log_id for s in sleeps] == [
        s.sleep_log_id for s in sorted(ds.sleep_logs, key=lambda x: x.wake_time)
    ]
    meals = r.SqlAlchemyMealLogRepository(session).list_between(uid, start, end)
    assert len(meals) == len(ds.meal_logs)
    journals = r.SqlAlchemyDailyJournalRepository(session).list_between_dates(
        uid, FIXED_END_DATE - timedelta(days=29), FIXED_END_DATE
    )
    assert len(journals) == len(ds.daily_journals)


def test_workout_three_levels_roundtrip(session: Session) -> None:
    uid = make_user(session)
    when = local_datetime(date(2026, 11, 1), time(20, 30))
    original = WorkoutLog(
        workout_log_id=UUID(int=1),
        user_id=uid,
        performed_at=when,
        calories_burned=450.0,
        exercises=[
            WorkoutExercise(
                exercise_name="有酸素運動", category=ExerciseCategory.CARDIO, duration_minutes=25
            ),
            WorkoutExercise(
                exercise_name="ラットプルダウン",
                category=ExerciseCategory.STRENGTH,
                # セットごとに重さが違うケース
                sets=[WorkoutSet(reps=10, weight_kg=30), WorkoutSet(reps=10, weight_kg=20)],
            ),
        ],
    )
    repo = r.SqlAlchemyWorkoutLogRepository(session)
    repo.add_many([original])
    session.flush()
    session.expire_all()  # DBから読み直させる
    start, end = local_day_range(date(2026, 11, 1), date(2026, 11, 1))
    [loaded] = repo.list_between(uid, start, end)
    assert loaded == original


def test_range_is_half_open_by_jst_day(session: Session) -> None:
    uid = make_user(session)
    _load(session, uid)
    day = FIXED_END_DATE - timedelta(days=5)
    start, end = local_day_range(day, day)
    meals = r.SqlAlchemyMealLogRepository(session).list_between(uid, start, end)
    assert meals
    assert all(start <= m.consumed_at < end for m in meals)
    # 隣接する日の範囲と重複しない(境界の二重カウントなし)
    n_start, n_end = local_day_range(day + timedelta(days=1), day + timedelta(days=1))
    next_ids = {
        m.meal_log_id
        for m in r.SqlAlchemyMealLogRepository(session).list_between(uid, n_start, n_end)
    }
    assert not next_ids & {m.meal_log_id for m in meals}


def test_user_isolation_across_all_repositories(session: Session) -> None:
    a, b = make_user(session), make_user(session)
    _load(session, a, seed=1)
    _load(session, b, seed=2)
    start, end = local_day_range(FIXED_END_DATE - timedelta(days=29), FIXED_END_DATE)
    d0, d1 = FIXED_END_DATE - timedelta(days=29), FIXED_END_DATE

    for rows in (
        r.SqlAlchemySleepLogRepository(session).list_between(a, start, end),
        r.SqlAlchemyMealLogRepository(session).list_between(a, start, end),
        r.SqlAlchemyWorkoutLogRepository(session).list_between(a, start, end),
        r.SqlAlchemyHealthLogRepository(session).list_between(a, start, end),
        r.SqlAlchemyWeightLogRepository(session).list_between(a, start, end),
        r.SqlAlchemyDailyJournalRepository(session).list_between_dates(a, d0, d1),
    ):
        assert rows
        assert {x.user_id for x in rows} == {a}  # Bの行は一件も見えない

    # 存在しないユーザーには何も返らない
    assert r.SqlAlchemySleepLogRepository(session).list_between(UUID(int=99), start, end) == []


def test_read_only_unit_of_work_rejects_writes(migrated_database: str) -> None:
    from health_insight_agent.database.connection import create_db_engine, create_session_factory

    engine = create_db_engine(migrated_database)
    uow = SqlAlchemyUnitOfWork(create_session_factory(engine), read_only=True)
    with uow:
        with pytest.raises(DBAPIError, match="read-only"):
            uow.users.add(  # 追加→flushでINSERTが走り、READ ONLYトランザクションが拒否する
                __import__("health_insight_agent.domain.models", fromlist=["User"]).User(
                    user_id=UUID(int=5),
                    email="ro@example.com",
                    display_name="x",
                    created_at=local_datetime(date(2026, 1, 1), time.min),
                )
            )
            uow._session.flush()  # type: ignore[union-attr]
    engine.dispose()


def test_purge_user_data_only_removes_target_user(migrated_database: str) -> None:
    """purgeは対象ユーザーのログだけを消し、他ユーザーには触れない。"""
    from health_insight_agent.database.connection import create_db_engine, create_session_factory

    engine = create_db_engine(migrated_database)
    with engine.connect() as conn:
        outer = conn.begin()
        sess = create_session_factory(engine)(bind=conn)
        a, b = make_user(sess), make_user(sess)
        _load(sess, a, seed=1)
        _load(sess, b, seed=2)
        uow = SqlAlchemyUnitOfWork(lambda: sess)
        uow._session = sess
        uow.purge_user_data(a)
        sess.flush()
        start, end = local_day_range(FIXED_END_DATE - timedelta(days=29), FIXED_END_DATE)
        assert r.SqlAlchemyWorkoutLogRepository(sess).list_between(a, start, end) == []
        assert r.SqlAlchemyWorkoutLogRepository(sess).list_between(b, start, end)
        leftover = sess.execute(
            text(
                "SELECT count(*) FROM workout_sets s JOIN workout_exercises e"
                " ON s.workout_exercise_id = e.id JOIN workout_logs w ON e.workout_log_id = w.id"
                " WHERE w.user_id = :u"
            ),
            {"u": a},
        ).scalar_one()
        assert leftover == 0
        sess.close()
        outer.rollback()
    engine.dispose()
