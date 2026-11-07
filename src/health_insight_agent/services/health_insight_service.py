"""HealthScore/Recommendation を返すサービス。

`HealthInsightPort`(Protocol)の背後に置く。現在は移植版(analysis/, recommendation/)を使い、
統合時は既存アプリの RecommendationService を包むアダプタに差し替える(既存アプリは変更しない)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol
from uuid import UUID

from health_insight_agent.analysis.health_score import (
    HealthScore,
    analyze_nutrition,
    analyze_sleep,
    analyze_weight,
    analyze_workout,
    compute_health_score,
    compute_nutrition_features,
    compute_sleep_features,
    compute_weight_features,
    compute_workout_features,
)
from health_insight_agent.local_time import local_day_range
from health_insight_agent.recommendation.rule_based import (
    Recommendation,
    generate_recommendations,
)
from health_insight_agent.repositories.protocols import UnitOfWork

_RECENT_DAYS = 7
"""食事・睡眠・運動は直近7日(既存アプリと同じ週間サイクル)。"""
_WEIGHT_DAYS = 30
"""体重は日々のブレに引きずられないよう30日。"""


@dataclass(frozen=True, kw_only=True, slots=True)
class HealthReport:
    as_of: date
    health_score: HealthScore
    recommendations: list[Recommendation]


class HealthInsightPort(Protocol):
    def report(self, user_id: UUID, as_of: date, locale: str = "ja") -> HealthReport: ...


class LocalHealthInsightService:
    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def report(self, user_id: UUID, as_of: date, locale: str = "ja") -> HealthReport:
        recent_start, recent_end = local_day_range(as_of - timedelta(days=_RECENT_DAYS - 1), as_of)
        weight_start, _ = local_day_range(as_of - timedelta(days=_WEIGHT_DAYS - 1), as_of)
        with self._uow_factory() as uow:
            meals = uow.meal_logs.list_between(user_id, recent_start, recent_end)
            sleeps = uow.sleep_logs.list_between(user_id, recent_start, recent_end)
            workouts = uow.workout_logs.list_between(user_id, recent_start, recent_end)
            weights = uow.weight_logs.list_between(user_id, weight_start, recent_end)
        score = compute_health_score(
            nutrition=analyze_nutrition(compute_nutrition_features(meals), locale),
            weight=analyze_weight(compute_weight_features(weights), locale),
            sleep=analyze_sleep(compute_sleep_features(sleeps), locale),
            workout=analyze_workout(compute_workout_features(workouts), locale),
        )
        return HealthReport(
            as_of=as_of, health_score=score, recommendations=generate_recommendations(score)
        )
