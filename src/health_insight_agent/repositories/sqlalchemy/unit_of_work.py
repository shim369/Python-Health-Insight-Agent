"""SQLAlchemy版 UnitOfWork。1回の利用=1セッション=1トランザクション。"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from health_insight_agent.database import schema as orm
from health_insight_agent.repositories.sqlalchemy.repositories import (
    SqlAlchemyDailyJournalRepository,
    SqlAlchemyHealthLogRepository,
    SqlAlchemyMealLogRepository,
    SqlAlchemySleepLogRepository,
    SqlAlchemyUserRepository,
    SqlAlchemyWeightLogRepository,
    SqlAlchemyWorkoutLogRepository,
)


class SqlAlchemyUnitOfWork:
    # UnitOfWork Protocolを満たすため、各Repositoryを属性として宣言する(中身は__enter__で設定)。
    users: SqlAlchemyUserRepository
    sleep_logs: SqlAlchemySleepLogRepository
    meal_logs: SqlAlchemyMealLogRepository
    workout_logs: SqlAlchemyWorkoutLogRepository
    health_logs: SqlAlchemyHealthLogRepository
    weight_logs: SqlAlchemyWeightLogRepository
    daily_journals: SqlAlchemyDailyJournalRepository

    def __init__(self, session_factory: Callable[[], Session], *, read_only: bool = False) -> None:
        self._session_factory = session_factory
        self._read_only = read_only
        self._session: Session | None = None

    def __enter__(self) -> SqlAlchemyUnitOfWork:
        session = self._session_factory()
        if self._read_only:
            # DBロール側の読み取り専用化が本命だが、念のためトランザクション自体も
            # READ ONLYにしておく(書き込みはDBが拒否する)。最初の文である必要がある。
            session.execute(text("SET TRANSACTION READ ONLY"))
        self._session = session
        self.users = SqlAlchemyUserRepository(session)
        self.sleep_logs = SqlAlchemySleepLogRepository(session)
        self.meal_logs = SqlAlchemyMealLogRepository(session)
        self.workout_logs = SqlAlchemyWorkoutLogRepository(session)
        self.health_logs = SqlAlchemyHealthLogRepository(session)
        self.weight_logs = SqlAlchemyWeightLogRepository(session)
        self.daily_journals = SqlAlchemyDailyJournalRepository(session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._session is not None
        # commitされなかった変更は破棄する(例外時も含む)。
        self._session.rollback()
        self._session.close()
        self._session = None

    def commit(self) -> None:
        assert self._session is not None
        self._session.commit()

    def purge_user_data(self, user_id: UUID) -> None:
        """指定ユーザーのログを全削除(合成データの再生成用。usersは残す)。

        UnitOfWork Protocolには含めない: エージェント経路から呼べないようにするため。
        """
        assert self._session is not None
        s = self._session
        workout_ids = select(orm.WorkoutLogORM.id).where(orm.WorkoutLogORM.user_id == user_id)
        exercise_ids = select(orm.WorkoutExerciseORM.id).where(
            orm.WorkoutExerciseORM.workout_log_id.in_(workout_ids)
        )
        s.execute(
            delete(orm.WorkoutSetORM).where(orm.WorkoutSetORM.workout_exercise_id.in_(exercise_ids))
        )
        s.execute(
            delete(orm.WorkoutExerciseORM).where(
                orm.WorkoutExerciseORM.workout_log_id.in_(workout_ids)
            )
        )
        for table in (
            orm.WorkoutLogORM,
            orm.SleepLogORM,
            orm.MealLogORM,
            orm.HealthLogORM,
            orm.WeightLogORM,
            orm.DailyJournalORM,
        ):
            s.execute(delete(table).where(table.user_id == user_id))
