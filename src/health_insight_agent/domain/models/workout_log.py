"""WorkoutLog。セッション/種目/セットの3階層(既存アプリと同じ)。

有酸素は種目の`duration_minutes`、筋トレは`sets`を使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from health_insight_agent.domain.enums import ExerciseCategory


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkoutSet:
    reps: int
    weight_kg: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkoutExercise:
    exercise_name: str
    category: ExerciseCategory
    duration_minutes: int | None = None
    sets: list[WorkoutSet] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkoutLog:
    workout_log_id: UUID
    user_id: UUID
    performed_at: datetime
    exercises: list[WorkoutExercise]
    calories_burned: float | None = None
    memo: str | None = None
