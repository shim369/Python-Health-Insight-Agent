"""SleepLog。既存アプリと同じく、睡眠時間は保存せず就寝・起床から都度算出する。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from health_insight_agent.domain.enums import SleepQuality


@dataclass(frozen=True, kw_only=True, slots=True)
class SleepLog:
    sleep_log_id: UUID
    user_id: UUID
    bedtime: datetime
    wake_time: datetime
    quality: SleepQuality | None = None

    @property
    def duration_minutes(self) -> int:
        # 保存値と算出値のズレを防ぐため、列として持たない(既存アプリの設計に合わせる)。
        return int((self.wake_time - self.bedtime).total_seconds() // 60)
