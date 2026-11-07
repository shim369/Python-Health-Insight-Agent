"""Repositoryのインターフェース(Protocol)。

方針(全てのデータアクセスは user_id を境界にする):
- 取得メソッドは必ず`user_id`を引数に取り、他ユーザーの行は返さない。
- 期間は半開区間[start, end)。日付→UTC範囲の変換はService側(local_time.local_day_range)。
- 本エージェントは基本的に読み取り専用。`add_many`は合成データのseed/テスト用。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import TracebackType
from typing import Protocol
from uuid import UUID

from health_insight_agent.domain.models import (
    DailyJournal,
    HealthLog,
    MealLog,
    SleepLog,
    User,
    WeightLog,
    WorkoutLog,
)


@dataclass(frozen=True, slots=True)
class DataBounds:
    """あるユーザーのログの件数と最初・最後の日時(データ量の把握・「データ不足」の判断用)。"""

    count: int
    first: datetime | date | None
    last: datetime | date | None


class UserRepository(Protocol):
    def add(self, user: User) -> None: ...

    def get(self, user_id: UUID) -> User | None: ...


class SleepLogRepository(Protocol):
    def bounds(self, user_id: UUID) -> DataBounds: ...

    def add_many(self, logs: Sequence[SleepLog]) -> None: ...

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[SleepLog]:
        """起床時刻(wake_time)が[start, end)の睡眠。睡眠は起床日に属する(既存アプリと同じ)。"""
        ...


class MealLogRepository(Protocol):
    def bounds(self, user_id: UUID) -> DataBounds: ...

    def add_many(self, logs: Sequence[MealLog]) -> None: ...

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[MealLog]: ...


class WorkoutLogRepository(Protocol):
    def bounds(self, user_id: UUID) -> DataBounds: ...

    def add_many(self, logs: Sequence[WorkoutLog]) -> None: ...

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[WorkoutLog]:
        """セッション/種目/セットの3階層をまとめて返す。"""
        ...


class HealthLogRepository(Protocol):
    def bounds(self, user_id: UUID) -> DataBounds: ...

    def add_many(self, logs: Sequence[HealthLog]) -> None: ...

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[HealthLog]: ...


class WeightLogRepository(Protocol):
    def bounds(self, user_id: UUID) -> DataBounds: ...

    def add_many(self, logs: Sequence[WeightLog]) -> None: ...

    def list_between(self, user_id: UUID, start: datetime, end: datetime) -> list[WeightLog]: ...


class DailyJournalRepository(Protocol):
    def bounds(self, user_id: UUID) -> DataBounds: ...

    def add_many(self, journals: Sequence[DailyJournal]) -> None: ...

    def list_between_dates(self, user_id: UUID, start: date, end: date) -> list[DailyJournal]:
        """journal_dateが[start, end](両端含む)の振り返り。"""
        ...


class UnitOfWork(Protocol):
    # 読み取り専用プロパティにする(実装側が具体型のRepositoryを持てるように共変にする)。
    @property
    def users(self) -> UserRepository: ...

    @property
    def sleep_logs(self) -> SleepLogRepository: ...

    @property
    def meal_logs(self) -> MealLogRepository: ...

    @property
    def workout_logs(self) -> WorkoutLogRepository: ...

    @property
    def health_logs(self) -> HealthLogRepository: ...

    @property
    def weight_logs(self) -> WeightLogRepository: ...

    @property
    def daily_journals(self) -> DailyJournalRepository: ...

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...
