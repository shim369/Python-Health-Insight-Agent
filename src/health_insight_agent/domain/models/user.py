"""User Domain Model。Geminiには送らない(メール等の個人識別情報を含むため)。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from health_insight_agent.domain.enums import Locale


@dataclass(frozen=True, kw_only=True, slots=True)
class User:
    user_id: UUID
    email: str
    display_name: str
    created_at: datetime
    locale: Locale = Locale.JA
