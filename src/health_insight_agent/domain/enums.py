"""Domain Modelで使うEnum。値は既存アプリ(domain/enums.py)と同一にする(統合時の互換性のため)。"""

from __future__ import annotations

from enum import StrEnum


class Locale(StrEnum):
    JA = "ja"
    EN = "en"


class MealType(StrEnum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    PRE_WORKOUT = "pre_workout"


class ExerciseCategory(StrEnum):
    CARDIO = "cardio"
    STRENGTH = "strength"


class SleepQuality(StrEnum):
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class WorkoutIntensity(StrEnum):
    """運動強度。DBには保存せず、ログから都度算出する派生指標(domain/intensity.py)。"""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
