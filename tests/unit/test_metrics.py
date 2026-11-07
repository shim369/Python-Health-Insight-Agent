"""日次テーブルの時間対応(起床日X ⇔ 前日X-1の食事・運動)の検証。"""

from __future__ import annotations

from datetime import date, time
from uuid import uuid4

from health_insight_agent.domain.enums import MealType
from health_insight_agent.domain.models import MealLog, SleepLog, WorkoutLog
from health_insight_agent.local_time import local_datetime
from health_insight_agent.services.metrics import build_day_rows

U = uuid4()
D1, D2 = date(2026, 11, 1), date(2026, 11, 2)


def _meal(d: date, t: time, kcal: float, mt: MealType = MealType.DINNER) -> MealLog:
    return MealLog(
        meal_log_id=uuid4(),
        user_id=U,
        consumed_at=local_datetime(d, t),
        meal_type=mt,
        calories=kcal,
        protein_g=1,
        fat_g=1,
        carbohydrate_g=1,
    )


def _sleep(bed_day: date, bed: time, wake_day: date, wake: time) -> SleepLog:
    return SleepLog(
        sleep_log_id=uuid4(),
        user_id=U,
        bedtime=local_datetime(bed_day, bed),
        wake_time=local_datetime(wake_day, wake),
    )


def _rows(**kw: object) -> dict[date, object]:
    base: dict[str, object] = {
        "start": D2,
        "end": D2,
        "sleeps": [],
        "meals": [],
        "workouts": [],
        "health_logs": [],
        "weights": [],
    }
    base.update(kw)
    return {r.date: r for r in build_day_rows(**base)}  # type: ignore[arg-type]


def test_sleep_belongs_to_wake_date_and_uses_previous_day_meals() -> None:
    rows = _rows(
        sleeps=[_sleep(D1, time(23, 30), D2, time(6, 30))],
        meals=[
            _meal(D1, time(21, 0), 700),  # 就寝2.5時間前 → 遅い食事
            _meal(D1, time(7, 30), 400, MealType.BREAKFAST),
            _meal(D2, time(7, 30), 999, MealType.BREAKFAST),  # 当日朝は前日扱いにならない
        ],
    )
    r = rows[D2]
    assert r.sleep_duration_min == 420  # type: ignore[attr-defined]
    assert r.sleep_bedtime_hour == 23.5  # type: ignore[attr-defined]
    assert r.prev_day_last_meal_hour == 21.0  # type: ignore[attr-defined]
    assert r.prev_day_meal_to_bed_gap_h == 2.5  # type: ignore[attr-defined]
    assert r.prev_day_late_meal == 1.0  # type: ignore[attr-defined]
    assert r.prev_day_breakfast_kcal == 400  # type: ignore[attr-defined]
    assert r.prev_day_total_kcal == 1100  # type: ignore[attr-defined]


def test_midnight_crossing_bedtime_is_above_24() -> None:
    r = _rows(
        sleeps=[_sleep(D2, time(0, 30), D2, time(7, 0))], meals=[_meal(D1, time(20, 0), 600)]
    )[D2]
    assert r.sleep_bedtime_hour == 24.5  # type: ignore[attr-defined]
    assert r.prev_day_meal_to_bed_gap_h == 4.5  # type: ignore[attr-defined]
    assert r.prev_day_late_meal == 0.0  # type: ignore[attr-defined]


def test_no_workout_is_zero_not_missing_but_no_logs_at_all_is_unknown() -> None:
    with_meals = _rows(meals=[_meal(D1, time(19, 0), 600)])[D2]
    assert with_meals.prev_day_workout_kcal == 0.0  # type: ignore[attr-defined]
    assert with_meals.prev_day_evening_high_workout == 0.0  # type: ignore[attr-defined]
    nothing = _rows()[D2]
    assert nothing.prev_day_workout_kcal is None  # type: ignore[attr-defined]
    assert nothing.sleep_duration_min is None  # type: ignore[attr-defined]


def test_evening_high_workout_requires_20h_and_high_intensity() -> None:
    def w(hour: int, kcal: float) -> WorkoutLog:
        return WorkoutLog(
            workout_log_id=uuid4(),
            user_id=U,
            performed_at=local_datetime(D1, time(hour, 0)),
            exercises=[],
            calories_burned=kcal,
        )

    meals = [_meal(D1, time(19, 0), 600)]
    assert _rows(meals=meals, workouts=[w(21, 500)])[D2].prev_day_evening_high_workout == 1.0  # type: ignore[attr-defined]
    assert _rows(meals=meals, workouts=[w(18, 500)])[D2].prev_day_evening_high_workout == 0.0  # type: ignore[attr-defined]
    assert _rows(meals=meals, workouts=[w(21, 300)])[D2].prev_day_evening_high_workout == 0.0  # type: ignore[attr-defined]
