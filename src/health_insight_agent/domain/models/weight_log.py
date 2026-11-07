"""WeightLog。HealthScore(体重領域)の入力として必要なため持つ(requirements.md 4.2-4)。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, kw_only=True, slots=True)
class WeightLog:
    weight_log_id: UUID
    user_id: UUID
    recorded_at: datetime
    weight_kg: float
