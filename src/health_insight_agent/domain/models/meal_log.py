"""MealLog。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from health_insight_agent.domain.enums import MealType


@dataclass(frozen=True, kw_only=True, slots=True)
class MealLog:
    meal_log_id: UUID
    user_id: UUID
    consumed_at: datetime
    meal_type: MealType
    calories: float
    protein_g: float
    fat_g: float
    carbohydrate_g: float
    memo: str | None = None
