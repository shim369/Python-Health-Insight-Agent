"""DailyJournal(振り返りP/C/A)。RAG検索の対象は振り返り系のテキスト列。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from uuid import UUID


@dataclass(frozen=True, kw_only=True, slots=True)
class DailyJournal:
    daily_journal_id: UUID
    user_id: UUID
    journal_date: date
    diet_policy_note: str | None = None
    wake_target_time: time | None = None
    sleep_target_time: time | None = None
    shopping_memo: str | None = None
    food_expense_yen: int | None = None
    reflection_memo: str | None = None
    improvement_point: str | None = None
    cooking_shopping_note: str | None = None
    study_prep_note: str | None = None
