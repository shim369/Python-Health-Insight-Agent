"""SQLAlchemy実装のRepository群。

全てのSELECTに `user_id == :user_id` を付ける。これがユーザー分離の唯一の砦なので、
新しいメソッドを足すときも必ず user_id 条件を入れること(tests/integration/test_repositories.py
に分離テストあり)。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from health_insight_agent.database import schema as orm
from health_insight_agent.domain.models import (
    DailyJournal,
    HealthLog,
    MealLog,
    SleepLog,
    User,
    WeightLog,
    WorkoutLog,
)
from health_insight_agent.repositories.sqlalchemy import mappers as m


class SqlAlchemyUserRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add(self, user: User) -> None:
        self._s.add(m.user_to_orm(user))

    def get(self, user_id: UUID) -> User | None:
        row = self._s.get(orm.UserORM, user_id)
        return m.user_to_domain(row) if row else None


class SqlAlchemySleepLogRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add_many(self, logs: Sequence[SleepLog]) -> None:
        self._s.add_all([m.sleep_to_orm(x) for x in logs])

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[SleepLog]:
        stmt = (
            select(orm.SleepLogORM)
            .where(
                orm.SleepLogORM.user_id == user_id,
                orm.SleepLogORM.wake_time >= start,
                orm.SleepLogORM.wake_time < end,
            )
            .order_by(orm.SleepLogORM.wake_time)
        )
        return [m.sleep_to_domain(r) for r in self._s.scalars(stmt)]


class SqlAlchemyMealLogRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add_many(self, logs: Sequence[MealLog]) -> None:
        self._s.add_all([m.meal_to_orm(x) for x in logs])

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[MealLog]:
        stmt = (
            select(orm.MealLogORM)
            .where(
                orm.MealLogORM.user_id == user_id,
                orm.MealLogORM.consumed_at >= start,
                orm.MealLogORM.consumed_at < end,
            )
            .order_by(orm.MealLogORM.consumed_at)
        )
        return [m.meal_to_domain(r) for r in self._s.scalars(stmt)]


class SqlAlchemyWorkoutLogRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add_many(self, logs: Sequence[WorkoutLog]) -> None:
        self._s.add_all([m.workout_to_orm(x) for x in logs])

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[WorkoutLog]:
        stmt = (
            select(orm.WorkoutLogORM)
            .where(
                orm.WorkoutLogORM.user_id == user_id,
                orm.WorkoutLogORM.performed_at >= start,
                orm.WorkoutLogORM.performed_at < end,
            )
            # 3階層をまとめて読み、種目・セットごとのN+1クエリを避ける。
            .options(
                selectinload(orm.WorkoutLogORM.exercises).selectinload(orm.WorkoutExerciseORM.sets)
            )
            .order_by(orm.WorkoutLogORM.performed_at)
        )
        return [m.workout_to_domain(r) for r in self._s.scalars(stmt)]


class SqlAlchemyHealthLogRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add_many(self, logs: Sequence[HealthLog]) -> None:
        self._s.add_all([m.health_to_orm(x) for x in logs])

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[HealthLog]:
        stmt = (
            select(orm.HealthLogORM)
            .where(
                orm.HealthLogORM.user_id == user_id,
                orm.HealthLogORM.recorded_at >= start,
                orm.HealthLogORM.recorded_at < end,
            )
            .order_by(orm.HealthLogORM.recorded_at)
        )
        return [m.health_to_domain(r) for r in self._s.scalars(stmt)]


class SqlAlchemyWeightLogRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add_many(self, logs: Sequence[WeightLog]) -> None:
        self._s.add_all([m.weight_to_orm(x) for x in logs])

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[WeightLog]:
        stmt = (
            select(orm.WeightLogORM)
            .where(
                orm.WeightLogORM.user_id == user_id,
                orm.WeightLogORM.recorded_at >= start,
                orm.WeightLogORM.recorded_at < end,
            )
            .order_by(orm.WeightLogORM.recorded_at)
        )
        return [m.weight_to_domain(r) for r in self._s.scalars(stmt)]


class SqlAlchemyDailyJournalRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add_many(self, journals: Sequence[DailyJournal]) -> None:
        self._s.add_all([m.journal_to_orm(x) for x in journals])

    def list_between_dates(self, user_id: UUID, start: date, end: date) -> list[DailyJournal]:
        stmt = (
            select(orm.DailyJournalORM)
            .where(
                orm.DailyJournalORM.user_id == user_id,
                orm.DailyJournalORM.journal_date >= start,
                orm.DailyJournalORM.journal_date <= end,
            )
            .order_by(orm.DailyJournalORM.journal_date)
        )
        return [m.journal_to_domain(r) for r in self._s.scalars(stmt)]
