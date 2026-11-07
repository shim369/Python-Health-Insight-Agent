"""HealthLog(体調5軸: 朝・夜・メンタル・疲労・ストレス。各1〜5)。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, kw_only=True, slots=True)
class HealthLog:
    health_log_id: UUID
    user_id: UUID
    recorded_at: datetime
    condition_score: int | None = None
    morning_condition: int | None = None
    evening_condition: int | None = None
    mental_score: int | None = None
    fatigue_score: int | None = None
    stress_score: int | None = None
    symptoms: list[str] = field(default_factory=list)
    memo: str | None = None
