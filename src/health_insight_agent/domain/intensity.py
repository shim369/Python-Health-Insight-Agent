"""運動強度(派生指標)。

既存スキーマに強度の列は無く、既存アプリは筋トレを運動時間に計上しない。そこで
スキーマを変えず、ログから都度算出する(requirements.md 7章 決定2)。

推定消費kcal = `calories_burned`(あれば最優先)。無ければ、有酸素の分数と筋トレのボリューム
(重さ×回数)から概算。しきい値: >=400 HIGH, >=200 MODERATE, それ以外 LOW。
係数は目安であり医学的根拠はない(合成データの仕込みもこの定義に合わせている)。
"""

from __future__ import annotations

from health_insight_agent.domain.enums import ExerciseCategory, WorkoutIntensity
from health_insight_agent.domain.models import WorkoutLog

HIGH_KCAL = 400.0
MODERATE_KCAL = 200.0
_KCAL_PER_CARDIO_MIN = 7.0
_KCAL_PER_VOLUME_KG = 0.04  # 1kg持ち上げ(重さ×回数)あたりの概算


def strength_volume_kg(workout: WorkoutLog) -> float:
    """筋トレの総ボリューム(重さ×回数の合計)。自重(重さなし)は0扱い。"""
    return sum(
        (s.weight_kg or 0.0) * s.reps
        for e in workout.exercises
        if e.category == ExerciseCategory.STRENGTH
        for s in e.sets
    )


def cardio_minutes(workout: WorkoutLog) -> int:
    return sum(
        e.duration_minutes or 0 for e in workout.exercises if e.category == ExerciseCategory.CARDIO
    )


def estimated_kcal(workout: WorkoutLog) -> float:
    if workout.calories_burned is not None:
        return workout.calories_burned
    return (
        cardio_minutes(workout) * _KCAL_PER_CARDIO_MIN
        + strength_volume_kg(workout) * _KCAL_PER_VOLUME_KG
    )


def workout_intensity(workout: WorkoutLog) -> WorkoutIntensity:
    kcal = estimated_kcal(workout)
    if kcal >= HIGH_KCAL:
        return WorkoutIntensity.HIGH
    if kcal >= MODERATE_KCAL:
        return WorkoutIntensity.MODERATE
    return WorkoutIntensity.LOW
