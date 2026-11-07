"""ガード付きSQLを実DBで検証: 分離・READ ONLY・タイムアウト・行数上限・DBロール。"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from health_insight_agent.agent.loop import HealthInsightAgent
from health_insight_agent.ai.rule_based_fake import RuleBasedFakeLLM
from health_insight_agent.database.connection import create_db_engine
from health_insight_agent.database.readonly_role import (
    create_login_role,
    drop_login_role,
    readonly_url,
)
from health_insight_agent.readonly_sql.executor import ReadOnlySqlExecutor, SqlExecutionError
from health_insight_agent.services.health_insight_service import LocalHealthInsightService
from health_insight_agent.services.insight_data_service import InsightDataService
from health_insight_agent.tools.base import ToolRegistry
from health_insight_agent.tools.builtin import build_default_registry
from tests.conftest import FIXED_END_DATE

pytestmark = pytest.mark.integration


@pytest.fixture
def env(seeded: Any, migrated_database: str) -> tuple[ToolRegistry, UUID, UUID, str]:
    factory, uid, other = seeded
    reg = build_default_registry(
        InsightDataService(factory, clock=lambda: FIXED_END_DATE),
        LocalHealthInsightService(factory),
        sql=ReadOnlySqlExecutor(migrated_database),
    )
    return reg, uid, other, migrated_database


def _sql(reg: ToolRegistry, user: UUID, sql: str, **kw: Any) -> dict[str, Any]:
    return reg.execute(user, "run_readonly_sql", {"sql": sql, **kw}).content


def _admin_count(url: str, sql: str, **params: Any) -> int:
    engine = create_db_engine(url)
    with engine.connect() as c:
        n = c.execute(text(sql), params).scalar_one()
    engine.dispose()
    return int(n)


def test_results_are_scoped_to_the_calling_user(env: Any) -> None:
    reg, uid, other, url = env
    mine = _sql(reg, uid, "SELECT count(*) AS n FROM sleep_logs")
    theirs = _sql(reg, other, "SELECT count(*) AS n FROM sleep_logs")
    assert mine["ok"] and theirs["ok"]
    expected_mine = _admin_count(url, "SELECT count(*) FROM sleep_logs WHERE user_id = :u", u=uid)
    expected_theirs = _admin_count(
        url, "SELECT count(*) FROM sleep_logs WHERE user_id = :u", u=other
    )
    assert mine["data"]["rows"] == [[expected_mine]]
    assert theirs["data"]["rows"] == [[expected_theirs]]
    assert expected_mine != expected_theirs


def test_asking_for_another_users_rows_returns_nothing(env: Any) -> None:
    reg, uid, other, _ = env
    r = _sql(reg, uid, f"SELECT * FROM meal_logs WHERE user_id = '{other}'")
    assert r["ok"] and r["data"]["rows"] == []
    r = _sql(reg, uid, "SELECT count(DISTINCT user_id) AS users FROM meal_logs")
    assert r["data"]["rows"] == [[1]]  # 見えるユーザーは自分だけ
    r = _sql(reg, uid, "SELECT count(*) FROM meal_logs WHERE user_id <> user_id OR true")
    own = _admin_count(env[3], "SELECT count(*) FROM meal_logs WHERE user_id = :u", u=uid)
    assert r["data"]["rows"] == [[own]]  # OR true でも自分の分だけ


def test_child_tables_are_isolated_through_parents(env: Any) -> None:
    reg, uid, _, url = env
    r = _sql(
        reg,
        uid,
        "SELECT count(*) AS n FROM workout_logs w JOIN workout_exercises e "
        "ON e.workout_log_id = w.id JOIN workout_sets s ON s.workout_exercise_id = e.id",
    )
    expected = _admin_count(
        url,
        "SELECT count(*) FROM workout_sets s JOIN workout_exercises e ON s.workout_exercise_id"
        " = e.id JOIN workout_logs w ON e.workout_log_id = w.id WHERE w.user_id = :u",
        u=uid,
    )
    assert r["data"]["rows"] == [[expected]] and expected > 0
    # 親を介さず子表だけを直接読んでも、自分の行だけ
    direct = _sql(reg, uid, "SELECT count(*) FROM workout_sets")
    assert direct["data"]["rows"] == [[expected]]


def test_row_limit_and_truncation_flag(env: Any) -> None:
    reg, uid, _, _ = env
    r = _sql(reg, uid, "SELECT * FROM meal_logs", max_rows=5)
    assert r["data"]["row_count"] == 5 and r["data"]["truncated"] is True
    r = _sql(reg, uid, "SELECT * FROM meal_logs LIMIT 3", max_rows=50)
    assert r["data"]["row_count"] == 3 and r["data"]["truncated"] is False
    big = _sql(reg, uid, "SELECT * FROM meal_logs LIMIT 100000", max_rows=500)
    assert big["data"]["row_count"] <= 500


def test_percent_signs_timezones_and_json_safe_values(env: Any) -> None:
    reg, uid, _, _ = env
    like = _sql(reg, uid, "SELECT count(*) FROM daily_journals WHERE reflection_memo LIKE '%食%'")
    assert like["ok"]
    r = _sql(
        reg,
        uid,
        "SELECT consumed_at, calories, id, meal_type FROM meal_logs ORDER BY consumed_at LIMIT 1",
    )
    ts = r["data"]["rows"][0][0]
    assert ts.endswith("+09:00")  # JSTに直して返す(UTCだと日付を取り違えやすい)
    assert isinstance(r["data"]["rows"][0][2], str)  # UUIDは文字列


def test_aggregation_with_chart(env: Any) -> None:
    reg, uid, _, _ = env
    ex = reg.execute(
        uid,
        "run_readonly_sql",
        {
            "sql": "SELECT CASE WHEN EXTRACT(ISODOW FROM wake_time AT TIME ZONE 'Asia/Tokyo') >= 6 "
            "THEN 'weekend' ELSE 'weekday' END AS day_type, "
            "ROUND(AVG(EXTRACT(EPOCH FROM (wake_time - bedtime)) / 60)::numeric, 1) AS avg_min, "
            "COUNT(*) AS n FROM sleep_logs GROUP BY 1 ORDER BY 1"
        },
    )
    assert ex.ok
    rows = ex.content["data"]["rows"]
    assert [r[0] for r in rows] == ["weekday", "weekend"]
    assert all(isinstance(r[1], float) for r in rows)
    assert ex.charts and ex.charts[0]["kind"] == "bar"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM meal_logs",
        "UPDATE meal_logs SET calories = 0",
        "DROP TABLE meal_logs",
        "SELECT 1; DELETE FROM meal_logs",
        "WITH d AS (DELETE FROM meal_logs RETURNING *) SELECT * FROM d",
        "SELECT pg_sleep(30)",
        "SELECT * FROM users",
    ],
)
def test_dangerous_sql_through_the_tool_is_rejected_and_data_is_untouched(
    env: Any, sql: str
) -> None:
    reg, uid, _, url = env
    before = _admin_count(url, "SELECT count(*) FROM meal_logs")
    r = _sql(reg, uid, sql)
    assert r["ok"] is False and "safety guard" in r["error"]
    assert _admin_count(url, "SELECT count(*) FROM meal_logs") == before


def test_executor_itself_is_read_only_even_if_the_guard_were_bypassed(env: Any) -> None:
    """ガードを通さず直接executorに危険なSQLを渡しても、READ ONLYトランザクションが拒否する。"""
    _, _, _, url = env
    ex = ReadOnlySqlExecutor(url)
    before = _admin_count(url, "SELECT count(*) FROM meal_logs")
    for sql in ("DELETE FROM meal_logs", "CREATE TABLE zz_should_not_exist (id int)"):
        with pytest.raises(SqlExecutionError) as e:
            ex.run(sql, max_rows=10)
        assert e.value.kind == "read_only"
    assert _admin_count(url, "SELECT count(*) FROM meal_logs") == before
    leftover = _admin_count(
        url,
        "SELECT count(*) FROM information_schema.tables WHERE table_name = :t",
        t="zz_should_not_exist",
    )
    assert leftover == 0


def test_statement_timeout_cancels_long_queries(env: Any) -> None:
    _, _, _, url = env
    ex = ReadOnlySqlExecutor(url, statement_timeout_ms=200)
    with pytest.raises(SqlExecutionError) as e:
        ex.run("SELECT pg_sleep(3)", max_rows=10)  # ガードは通さず、executorのタイムアウトを検証
    assert e.value.kind == "timeout"


def test_database_role_cannot_write_or_read_users(env: Any) -> None:
    """5層目: 読み取り専用DBロール。アプリ側の検証が全て破られても、DBが拒否する。"""
    _, uid, _, admin_url = env
    name, password = f"ro_test_{uuid4().hex[:8]}", uuid4().hex
    create_login_role(admin_url, name, password)
    try:
        ex = ReadOnlySqlExecutor(readonly_url(admin_url, name, password))
        assert ex.run("SELECT count(*) FROM meal_logs", max_rows=5).rows
        with pytest.raises(SqlExecutionError) as e:
            ex.run("SELECT email FROM users", max_rows=5)
        assert e.value.kind == "permission"  # usersテーブルにSELECT権限が無い
        with pytest.raises(SqlExecutionError) as e2:
            ex.run("DELETE FROM meal_logs", max_rows=5)
        assert e2.value.kind in ("read_only", "permission")
        # ロール経由でもツールのガードは同じように動く
        reg = build_default_registry(
            InsightDataService(None),  # type: ignore[arg-type]
            LocalHealthInsightService(None),  # type: ignore[arg-type]
            sql=ex,
        )
        r = _sql(reg, uid, "SELECT count(*) AS n FROM meal_logs")
        assert r["ok"] and r["data"]["rows"][0][0] > 0
    finally:
        drop_login_role(admin_url, name)


def test_agent_can_use_the_sql_tool_end_to_end(env: Any) -> None:
    reg, uid, _, _ = env
    agent = HealthInsightAgent(
        RuleBasedFakeLLM(), reg, today=lambda: FIXED_END_DATE, sink=lambda _l: None
    )
    ans = agent.ask(uid, "平日と休日で睡眠時間の平均はどう違う?")
    assert [t.name for t in ans.tool_trace] == ["run_readonly_sql"] and ans.tool_trace[0].ok
    assert "weekday" in ans.text and "weekend" in ans.text
