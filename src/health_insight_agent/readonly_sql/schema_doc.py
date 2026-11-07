"""LLMに渡すスキーマ説明(ORMメタデータから自動生成。実スキーマとのズレを防ぐ)。"""

from __future__ import annotations

from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg

from health_insight_agent.database.schema import Base
from health_insight_agent.readonly_sql.guard import ALLOWED_TABLES, USER_TABLES

NOTES = (
    "Rules: SELECT only (one statement). Your data is automatically restricted to the current "
    "user, so do NOT filter by user_id. workout_exercises/workout_sets have no user_id; join "
    "workout_logs -> workout_exercises (workout_log_id) -> workout_sets (workout_exercise_id). "
    "All timestamps are timestamptz in UTC: convert with AT TIME ZONE 'Asia/Tokyo' before taking "
    "dates or hours. There is no sleep duration column: use wake_time - bedtime (e.g. "
    "EXTRACT(EPOCH FROM wake_time - bedtime)/60 minutes). A sleep belongs to its wake-up date. "
    "Intensity is not stored. Prefer the typed tools when they can answer; use SQL for custom "
    "aggregation."
)


def build_schema_doc() -> str:
    dialect = PGDialect_psycopg()  # type: ignore[no-untyped-call]
    lines = []
    for name in sorted(ALLOWED_TABLES):
        table = Base.metadata.tables[name]
        cols = ", ".join(
            f"{c.name} {c.type.compile(dialect=dialect)}"
            for c in table.columns
            if not (c.name == "user_id" and name in USER_TABLES)  # user_idは自動で絞られるので隠す
        )
        lines.append(f"{name}({cols})")
    return "Tables: " + "; ".join(lines) + ". " + NOTES
