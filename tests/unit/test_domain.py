from datetime import UTC, date, datetime, time
from uuid import uuid4

import pytest

from health_insight_agent.domain.enums import ExerciseCategory, WorkoutIntensity
from health_insight_agent.domain.intensity import (
    cardio_minutes,
    strength_volume_kg,
    workout_intensity,
)
from health_insight_agent.domain.models import (
    SleepLog,
    WorkoutExercise,
    WorkoutLog,
    WorkoutSet,
)
from health_insight_agent.local_time import local_date_of, local_datetime, local_day_range


def _workout(*, kcal: float | None, exercises: list[WorkoutExercise]) -> WorkoutLog:
    return WorkoutLog(
        workout_log_id=uuid4(),
        user_id=uuid4(),
        performed_at=datetime(2026, 1, 1, tzinfo=UTC),
        exercises=exercises,
        calories_burned=kcal,
    )


def test_sleep_duration_is_derived_from_bed_and_wake() -> None:
    log = SleepLog(
        sleep_log_id=uuid4(),
        user_id=uuid4(),
        bedtime=local_datetime(date(2026, 1, 1), time(23, 30)),
        wake_time=local_datetime(date(2026, 1, 2), time(6, 45)),
    )
    assert log.duration_minutes == 435


@pytest.mark.parametrize(
    ("kcal", "expected"),
    [
        (150, WorkoutIntensity.LOW),
        (200, WorkoutIntensity.MODERATE),
        (399, WorkoutIntensity.MODERATE),
        (400, WorkoutIntensity.HIGH),
    ],
)
def test_intensity_thresholds_use_calories_burned_first(kcal: float, expected: object) -> None:
    assert workout_intensity(_workout(kcal=kcal, exercises=[])) == expected


def test_intensity_falls_back_to_cardio_minutes_and_strength_volume() -> None:
    cardio = _workout(
        kcal=None,
        exercises=[
            WorkoutExercise(
                exercise_name="run", category=ExerciseCategory.CARDIO, duration_minutes=60
            )
        ],
    )
    assert cardio_minutes(cardio) == 60
    assert workout_intensity(cardio) == WorkoutIntensity.HIGH  # 60分*7=420kcal

    strength = _workout(
        kcal=None,
        exercises=[
            WorkoutExercise(
                exercise_name="press",
                category=ExerciseCategory.STRENGTH,
                sets=[WorkoutSet(reps=10, weight_kg=30), WorkoutSet(reps=10, weight_kg=20)],
            )
        ],
    )
    assert strength_volume_kg(strength) == 500
    assert workout_intensity(strength) == WorkoutIntensity.LOW  # 500*0.04=20kcal


def test_bodyweight_sets_count_as_zero_volume() -> None:
    w = _workout(
        kcal=None,
        exercises=[
            WorkoutExercise(
                exercise_name="pushup",
                category=ExerciseCategory.STRENGTH,
                sets=[WorkoutSet(reps=20, weight_kg=None)],
            )
        ],
    )
    assert strength_volume_kg(w) == 0


def test_local_date_uses_jst_not_utc() -> None:
    # UTC 2026-01-01 20:00 は JST 2026-01-02 05:00(日付が変わる)
    assert local_date_of(datetime(2026, 1, 1, 20, 0, tzinfo=UTC)) == date(2026, 1, 2)


def test_local_day_range_is_half_open_and_inclusive_of_end_day() -> None:
    start, end = local_day_range(date(2026, 1, 1), date(2026, 1, 1))
    assert (end - start).total_seconds() == 24 * 3600
    with pytest.raises(ValueError):
        local_day_range(date(2026, 1, 2), date(2026, 1, 1))
