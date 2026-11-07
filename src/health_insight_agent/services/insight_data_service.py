"""ログの取得・日次テーブル・比較・相関をまとめたService(ツールの土台)。

全メソッドが user_id を取り、Repositoryに渡す(ユーザー分離)。日付はJSTの暦日で受け取り、
半開区間のUTC範囲への変換はここで行う。
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from uuid import UUID

from health_insight_agent.domain.models import MealLog, SleepLog, WorkoutLog
from health_insight_agent.local_time import local_day_range, local_today
from health_insight_agent.repositories.protocols import DataBounds, UnitOfWork
from health_insight_agent.services import stats
from health_insight_agent.services.metrics import METRICS, DayRow, build_day_rows

MAX_PERIOD_DAYS = 366

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}
OPERATORS: tuple[str, ...] = tuple(_OPS)


class InvalidRequestError(ValueError):
    """ツール引数がドメイン的に不正(期間の逆転・未知の指標など)。LLMに返して修正させる。"""


@dataclass(frozen=True, slots=True)
class Condition:
    metric: str
    op: str
    value: float


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric: str
    comparison: stats.Comparison


@dataclass(frozen=True, slots=True)
class CompareDaysResult:
    start: date
    end: date
    condition: Condition
    n_days_in_period: int
    n_days_with_condition_data: int
    n_matching: int
    n_other: int
    matching_dates: list[date]
    metrics: list[MetricComparison]


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    start: date
    end: date
    x: str
    y: str
    n_days_in_period: int
    correlation: stats.Correlation


@dataclass(frozen=True, slots=True)
class DataOverview:
    today: date
    tables: dict[str, DataBounds]


def _check_period(start: date, end: date) -> None:
    if end < start:
        raise InvalidRequestError("end_date must be on or after start_date")
    if (end - start).days + 1 > MAX_PERIOD_DAYS:
        raise InvalidRequestError(f"period must be at most {MAX_PERIOD_DAYS} days")


def _check_metric(name: str) -> None:
    if name not in METRICS:
        raise InvalidRequestError(f"unknown metric {name!r}; valid: {', '.join(METRICS)}")


class InsightDataService:
    def __init__(
        self, uow_factory: Callable[[], UnitOfWork], *, clock: Callable[[], date] = local_today
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock

    def today(self) -> date:
        return self._clock()

    # ---- 生ログ ----
    def sleep_logs(self, user_id: UUID, start: date, end: date) -> list[SleepLog]:
        _check_period(start, end)
        s, e = local_day_range(start, end)
        with self._uow_factory() as uow:
            return uow.sleep_logs.list_between(user_id, s, e)

    def meal_logs(self, user_id: UUID, start: date, end: date) -> list[MealLog]:
        _check_period(start, end)
        s, e = local_day_range(start, end)
        with self._uow_factory() as uow:
            return uow.meal_logs.list_between(user_id, s, e)

    def workout_logs(self, user_id: UUID, start: date, end: date) -> list[WorkoutLog]:
        _check_period(start, end)
        s, e = local_day_range(start, end)
        with self._uow_factory() as uow:
            return uow.workout_logs.list_between(user_id, s, e)

    # ---- 日次テーブル ----
    def day_rows(self, user_id: UUID, start: date, end: date) -> list[DayRow]:
        """起床日[start, end]の日次テーブル。前日の食事・運動のため start-1 から読む。"""
        _check_period(start, end)
        wake_s, wake_e = local_day_range(start, end)
        prev_s, prev_e = local_day_range(start - timedelta(days=1), end - timedelta(days=1))
        with self._uow_factory() as uow:
            return build_day_rows(
                start=start,
                end=end,
                sleeps=uow.sleep_logs.list_between(user_id, wake_s, wake_e),
                meals=uow.meal_logs.list_between(user_id, prev_s, prev_e),
                workouts=uow.workout_logs.list_between(user_id, prev_s, prev_e),
                health_logs=uow.health_logs.list_between(user_id, wake_s, wake_e),
                weights=uow.weight_logs.list_between(user_id, wake_s, wake_e),
            )

    # ---- 分析 ----
    def compare_days(
        self,
        user_id: UUID,
        start: date,
        end: date,
        condition: Condition,
        metric_names: list[str],
    ) -> CompareDaysResult:
        _check_metric(condition.metric)
        for m in metric_names:
            _check_metric(m)
        if condition.op not in _OPS:
            raise InvalidRequestError(f"unknown operator {condition.op!r}")
        rows = self.day_rows(user_id, start, end)
        test = _OPS[condition.op]
        cond_get = METRICS[condition.metric].getter
        matching: list[DayRow] = []
        other: list[DayRow] = []
        for r in rows:
            v = cond_get(r)
            if v is None:  # 判定に使う値が無い日は、どちらのグループにも入れない
                continue
            (matching if test(v, condition.value) else other).append(r)
        comparisons = []
        for name in metric_names:
            get = METRICS[name].getter
            a = [x for r in matching if (x := get(r)) is not None]
            b = [x for r in other if (x := get(r)) is not None]
            comparisons.append(MetricComparison(name, stats.compare_groups(a, b)))
        return CompareDaysResult(
            start=start,
            end=end,
            condition=condition,
            n_days_in_period=len(rows),
            n_days_with_condition_data=len(matching) + len(other),
            n_matching=len(matching),
            n_other=len(other),
            matching_dates=[r.date for r in matching][:31],
            metrics=comparisons,
        )

    def correlate(self, user_id: UUID, start: date, end: date, x: str, y: str) -> CorrelationResult:
        _check_metric(x)
        _check_metric(y)
        rows = self.day_rows(user_id, start, end)
        gx, gy = METRICS[x].getter, METRICS[y].getter
        pairs = [(a, b) for r in rows if (a := gx(r)) is not None and (b := gy(r)) is not None]
        return CorrelationResult(
            start=start,
            end=end,
            x=x,
            y=y,
            n_days_in_period=len(rows),
            correlation=stats.correlate_pairs([p[0] for p in pairs], [p[1] for p in pairs]),
        )

    def overview(self, user_id: UUID) -> DataOverview:
        with self._uow_factory() as uow:
            tables = {
                "sleep_logs": uow.sleep_logs.bounds(user_id),
                "meal_logs": uow.meal_logs.bounds(user_id),
                "workout_logs": uow.workout_logs.bounds(user_id),
                "health_logs": uow.health_logs.bounds(user_id),
                "weight_logs": uow.weight_logs.bounds(user_id),
                "daily_journals": uow.daily_journals.bounds(user_id),
            }
        return DataOverview(today=self._clock(), tables=tables)
