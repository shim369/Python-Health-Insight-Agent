"""seed CLIの中核(生成→投入→再投入)が冪等であること。"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from health_insight_agent.database.connection import create_db_engine

pytestmark = pytest.mark.integration


def test_generated_sleep_rows_satisfy_db_constraints(migrated_database: str) -> None:
    """生成データがDBの制約(外部キー・ユニーク等)に通ること。"""
    from health_insight_agent.database.connection import create_session_factory
    from health_insight_agent.repositories.sqlalchemy import repositories as r
    from health_insight_agent.synthetic.generator import generate_dataset
    from tests.conftest import FIXED_END_DATE, make_user

    engine = create_db_engine(migrated_database)
    with engine.connect() as conn:
        outer = conn.begin()
        sess = create_session_factory(engine)(bind=conn)
        uid = make_user(sess)
        ds = generate_dataset(uid, end_date=FIXED_END_DATE, days=90, seed=42)
        r.SqlAlchemyWorkoutLogRepository(sess).add_many(ds.workout_logs)
        r.SqlAlchemyDailyJournalRepository(sess).add_many(ds.daily_journals)
        sess.flush()
        n = sess.execute(
            text("SELECT count(*) FROM workout_logs WHERE user_id = :u"), {"u": uid}
        ).scalar_one()
        assert n == len(ds.workout_logs)
        sess.close()
        outer.rollback()
    engine.dispose()
