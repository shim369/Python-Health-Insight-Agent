"""SQLガードの安全性テスト。危険なSQLが「通らない」ことを網羅的に確認する。"""

from __future__ import annotations

from uuid import UUID

import pytest

from health_insight_agent.readonly_sql.guard import SqlGuardError, guard_sql

UID = UUID("11111111-2222-4333-8444-555555555555")
OTHER = UUID("99999999-8888-4777-8666-555555555555")


def ok(sql: str, **kw: object) -> str:
    return guard_sql(sql, UID, max_rows=100, **kw).sql  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sql",
    [
        # --- 更新・削除・DDL ---
        "INSERT INTO meal_logs (id) VALUES (gen_random_uuid())",
        "UPDATE meal_logs SET calories = 0",
        "DELETE FROM meal_logs",
        "DROP TABLE meal_logs",
        "TRUNCATE meal_logs",
        "ALTER TABLE meal_logs ADD COLUMN x int",
        "CREATE TABLE t (id int)",
        "CREATE TABLE t AS SELECT * FROM meal_logs",
        "GRANT ALL ON meal_logs TO public",
        "COPY meal_logs TO '/tmp/x'",
        "SET statement_timeout = 0",
        "RESET ALL",
        "BEGIN",
        "COMMIT",
        "EXPLAIN ANALYZE DELETE FROM meal_logs",
        "VACUUM",
        "DO $$ BEGIN DELETE FROM meal_logs; END $$",
        "CALL some_proc()",
        # --- 複数文・コメント・引用による偽装 ---
        "SELECT 1; DROP TABLE meal_logs",
        "SELECT 1; SELECT 2",
        "SELECT 1 /* ; */; DELETE FROM meal_logs",
        "SELECT 1 -- x\n; DELETE FROM meal_logs",
        "",
        "   ",
        ";",
        "-- only a comment",
        # --- SELECT内の書き込み・副作用 ---
        "SELECT * INTO new_table FROM meal_logs",
        "SELECT * FROM meal_logs FOR UPDATE",
        "SELECT * FROM meal_logs FOR SHARE",
        "WITH d AS (DELETE FROM meal_logs RETURNING *) SELECT * FROM d",
        "WITH u AS (UPDATE meal_logs SET calories = 0 RETURNING *) SELECT * FROM u",
        "WITH i AS (INSERT INTO meal_logs (id) VALUES (gen_random_uuid()) RETURNING *) SELECT 1",
        # --- 危険な関数 ---
        "SELECT pg_sleep(10)",
        "SELECT PG_SLEEP(10)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT lo_import('/etc/passwd')",
        "SELECT set_config('statement_timeout', '0', false)",
        "SELECT current_setting('server_version')",
        "SELECT pg_terminate_backend(1)",
        "SELECT * FROM dblink('host=x', 'select 1') AS t(a int)",
        "SELECT nextval('s')",
        "SELECT version()",
        "SELECT current_catalog",
        "SELECT user",
        "SELECT current_role",
        "SELECT current_user",
        "SELECT unknown_custom_function(1)",
        "SELECT * FROM meal_logs WHERE calories > (SELECT pg_sleep(5))::int",
        # --- 許可外のテーブル・スキーマ ---
        "SELECT * FROM users",
        "SELECT email FROM users",
        "SELECT * FROM public.users",
        "SELECT * FROM alembic_version",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT * FROM pg_roles",
        "SELECT * FROM pg_user",
        "SELECT * FROM otherdb.public.meal_logs",
        "SELECT * FROM secret.meal_logs",
        "SELECT * FROM meal_logs m JOIN users u ON u.id = m.user_id",
        "SELECT * FROM meal_logs UNION SELECT * FROM users",
        "SELECT (SELECT email FROM users LIMIT 1)",
        "SELECT * FROM meal_logs WHERE user_id IN (SELECT id FROM users)",
        "WITH x AS (SELECT * FROM users) SELECT * FROM x",
        # --- CTE名による書き換えすり抜け ---
        "WITH meal_logs AS (SELECT * FROM meal_logs) SELECT * FROM meal_logs",
        "WITH sleep_logs AS (SELECT 1) SELECT * FROM sleep_logs",
        # --- その他 ---
        "SELEC * FROM meal_logs",
        "SELECT * FROM",
        "SELECT 1\x00; DROP TABLE meal_logs",
        "SELECT '" + "a" * 5000 + "'",
    ],
)
def test_dangerous_or_invalid_sql_is_rejected(sql: str) -> None:
    with pytest.raises(SqlGuardError):
        guard_sql(sql, UID, max_rows=100)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM meal_logs",
        "SELECT * FROM public.meal_logs",
        "SELECT m.calories FROM meal_logs AS m WHERE m.calories > 500",
        "SELECT date_trunc('week', wake_time AT TIME ZONE 'Asia/Tokyo') d, avg(1) "
        "FROM sleep_logs GROUP BY 1",
        "SELECT * FROM workout_logs w JOIN workout_exercises e ON e.workout_log_id = w.id "
        "JOIN workout_sets s ON s.workout_exercise_id = e.id",
        "WITH recent AS (SELECT * FROM meal_logs WHERE calories > 100) SELECT count(*) FROM recent",
        "SELECT * FROM meal_logs UNION ALL SELECT * FROM meal_logs",
        "SELECT * FROM generate_series(1, 5) AS g",
        "SELECT corr(fatigue_score, stress_score) FROM health_logs",
        "SELECT * FROM meal_logs WHERE memo LIKE '%肉%' ORDER BY consumed_at DESC LIMIT 5;",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY calories) FROM meal_logs",
        "SELECT 1",
    ],
)
def test_legitimate_read_queries_are_accepted(sql: str) -> None:
    assert ok(sql)


def test_every_table_reference_is_rewritten_to_the_callers_rows_only() -> None:
    out = ok("SELECT * FROM meal_logs m JOIN sleep_logs s ON true")
    assert out.count(f"user_id = '{UID}'") == 2  # 両方の表が絞り込まれる
    assert "99999999" not in out
    nested = ok("SELECT * FROM meal_logs WHERE id IN (SELECT id FROM meal_logs)")
    assert nested.count(f"user_id = '{UID}'") == 2  # サブクエリ内も
    cte = ok("WITH r AS (SELECT * FROM sleep_logs) SELECT * FROM r")
    assert cte.count(f"user_id = '{UID}'") == 1  # CTE本体の表参照が絞られる(CTE参照自体はそのまま)


def test_child_tables_are_scoped_through_their_parents() -> None:
    out = ok("SELECT * FROM workout_sets")
    assert f"w.user_id = '{UID}'" in out and "workout_logs" in out
    out2 = ok("SELECT * FROM workout_exercises")
    assert f"w.user_id = '{UID}'" in out2


def test_callers_own_user_id_filter_cannot_widen_access() -> None:
    # LLMが他人のIDで絞っても、強制された自分のユーザー絞り込みとANDになるだけ(0行になる)。
    out = ok(f"SELECT * FROM meal_logs WHERE user_id = '{OTHER}'")
    assert f"user_id = '{UID}'" in out and f"user_id = '{OTHER}'" in out
    out = ok("SELECT * FROM meal_logs WHERE true OR user_id <> user_id OR 1 = 1")
    assert f"user_id = '{UID}'" in out  # ORで回避しようとしても強制条件は外側のサブクエリ


def test_existing_alias_is_preserved_so_qualified_columns_still_work() -> None:
    out = ok("SELECT m.calories FROM meal_logs AS m")
    assert "AS m" in out
    assert "AS meal_logs" in ok("SELECT meal_logs.calories FROM meal_logs")


def test_row_limit_is_injected_or_respected() -> None:
    assert guard_sql("SELECT * FROM meal_logs", UID, max_rows=100).sql.endswith("LIMIT 101")
    assert guard_sql("SELECT * FROM meal_logs", UID, max_rows=100).limit == 101
    assert guard_sql("SELECT * FROM meal_logs LIMIT 5", UID, max_rows=100).sql.endswith("LIMIT 5")
    # 上限より大きいLIMITは引き下げる
    assert guard_sql("SELECT * FROM meal_logs LIMIT 100000", UID, max_rows=100).sql.endswith(
        "LIMIT 101"
    )
    union = guard_sql(
        "SELECT id FROM meal_logs UNION ALL SELECT id FROM sleep_logs", UID, max_rows=10
    )
    assert union.sql.endswith("LIMIT 11")


def test_rejection_messages_are_safe_to_show_to_the_llm() -> None:
    with pytest.raises(SqlGuardError) as e:
        guard_sql("SELECT * FROM users", UID, max_rows=10)
    assert "not available" in str(e.value) and "meal_logs" in str(e.value)
    with pytest.raises(SqlGuardError) as e2:
        guard_sql("SELECT pg_sleep(1)", UID, max_rows=10)
    assert "pg_sleep" in str(e2.value)


def test_user_id_must_be_a_uuid() -> None:
    with pytest.raises(TypeError):
        guard_sql("SELECT 1", "1' OR '1'='1", max_rows=10)  # type: ignore[arg-type]
