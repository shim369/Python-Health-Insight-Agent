"""SQLガード: LLMが書いたSQLを、実行前に検証し、ユーザー絞り込みを強制する。

多層防御の1層目(アプリ側)。以降の層: READ ONLYトランザクション・statement_timeout・行数上限・
読み取り専用DBロール(executor.py / migration 0002)。1つが破られても他で止まる設計にする。

方針:
- SELECT系(WITH/UNION含む)の**1文だけ**を許可。複数文・DML/DDL・SET/COPY/EXPLAIN等は拒否。
- 参照できるテーブルは許可リストのみ(usersはメールアドレス等を含むため対象外)。
  information_schema / pg_catalog / 他スキーマは拒否。
- 関数は危険なもの(pg_sleep, pg_read_file, dblink, set_config等)を名前で拒否。
  未知の関数は、許可リストに無ければ拒否(Anonymous関数は許可リスト方式)。
- **user_idの絞り込みは「書かせる」のではなく「強制する」**: 全テーブル参照を、呼び出しユーザー
  だけを返すサブクエリに書き換える。LLMのSQLが何を書いても他ユーザーの行には届かない。
- 行数上限はLIMITをASTに注入して強制する(既存LIMITが小さければそれを尊重)。
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

MAX_SQL_CHARS = 4000

# 直接user_idを持つ表
USER_TABLES = frozenset(
    {"meal_logs", "sleep_logs", "workout_logs", "health_logs", "weight_logs", "daily_journals"}
)
# user_idを持たず、親を介してユーザーに紐づく表
CHILD_TABLES = frozenset({"workout_exercises", "workout_sets"})
ALLOWED_TABLES = USER_TABLES | CHILD_TABLES

# 名前で拒否する関数(型付き関数・Anonymous関数の両方に適用)
DENIED_FUNCTIONS = frozenset(
    {
        "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file", "pg_read_binary_file",
        "pg_ls_dir", "pg_stat_file", "lo_import", "lo_export", "lo_get", "lo_put", "dblink",
        "dblink_exec", "set_config", "current_setting", "pg_terminate_backend",
        "pg_cancel_backend", "pg_reload_conf", "pg_advisory_lock", "pg_advisory_xact_lock",
        "pg_try_advisory_lock", "nextval", "setval", "currval", "txid_current", "query_to_xml",
        "table_to_xml", "database_to_xml", "pg_notify", "pg_logical_emit_message",
        "current_user", "session_user", "current_database", "current_schema", "version",
        "current_version", "current_catalog", "user",
        "inet_server_addr", "pg_backend_pid", "pg_read_server_files", "copy",
    }
)  # fmt: skip

_BARE_VALUE_FUNCTIONS = frozenset(
    {"user", "current_user", "session_user", "current_role", "current_catalog", "current_schema"}
)

# Anonymous(sqlglotが型を知らない)関数のうち許可するもの
ALLOWED_ANONYMOUS = frozenset(
    {
        "date_part", "date_trunc", "regr_slope", "regr_intercept", "regr_r2", "corr",
        "stddev_samp", "stddev_pop", "var_samp", "var_pop", "percentile_cont", "percentile_disc",
        "mode", "to_char", "make_interval", "age", "width_bucket", "log", "cbrt", "trunc",
        "generate_series", "greatest", "least", "starts_with", "split_part", "btrim",
    }
)  # fmt: skip

_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "Command", "Copy", "Into",
    "Lock", "Set", "Transaction", "Commit", "Rollback", "TruncateTable", "Grant", "Analyze",
    "Pragma", "Use", "Declare", "Attach", "Detach", "Refresh", "Comment", "Kill",
)  # fmt: skip
_FORBIDDEN_NODES: tuple[type[exp.Expr], ...] = tuple(
    getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n)
)
_QUERY_ROOTS = (exp.Select, exp.Union, exp.Subquery)


class SqlGuardError(ValueError):
    """ガードがSQLを拒否した。メッセージは利用者(LLM)に見せても安全な内容にする。"""


@dataclass(frozen=True, slots=True)
class GuardedSql:
    sql: str
    """実行するSQL(ユーザー絞り込み・LIMITを注入済み)。"""
    limit: int
    """注入した行数上限(max_rows + 1。超過検知のため1行多く取る)。"""


def _parse_single(sql: str) -> exp.Expr:
    if len(sql) > MAX_SQL_CHARS:
        raise SqlGuardError(f"SQL is too long (max {MAX_SQL_CHARS} characters)")
    if "\x00" in sql:
        raise SqlGuardError("SQL contains invalid characters")
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except SqlglotError as e:
        raise SqlGuardError(f"SQL could not be parsed: {str(e).splitlines()[0][:150]}") from e
    parsed = [s for s in statements if s is not None]
    if not parsed:
        raise SqlGuardError("empty SQL")
    if len(parsed) != 1:
        raise SqlGuardError("only a single statement is allowed (no ';' separated statements)")
    return parsed[0]


def _function_names(node: exp.Func) -> set[str]:
    """関数の名前候補。sqlglotの内部名(例: CURRENT_VERSION)と、Postgresとして出力した名前
    (例: VERSION)の両方を見る。どちらか一方でも拒否リストに載っていれば拒否するため。"""
    if isinstance(node, exp.Anonymous):
        return {node.name.lower()}
    rendered = node.sql(dialect="postgres").split("(", 1)[0].strip().lower()
    return {node.sql_name().lower(), rendered}


def _check_structure(tree: exp.Expr) -> set[str]:
    """文種・危険ノード・関数・テーブルを検証し、CTE名の集合を返す。"""
    if not isinstance(tree, _QUERY_ROOTS):
        raise SqlGuardError("only SELECT queries are allowed")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SqlGuardError(f"{type(node).__name__.upper()} is not allowed; SELECT only")
        if (
            isinstance(node, exp.Column)
            and not node.table
            and node.name.lower() in _BARE_VALUE_FUNCTIONS
        ):
            # `SELECT user` のように、括弧なしで書ける値関数(列名として解釈される)も拒否する。
            raise SqlGuardError(f"{node.name.lower()!r} is not allowed")
        if isinstance(node, exp.Func):
            names = _function_names(node)
            denied = names & DENIED_FUNCTIONS
            if denied:
                raise SqlGuardError(f"function {sorted(denied)[0]!r} is not allowed")
            if isinstance(node, exp.Anonymous) and node.name.lower() not in ALLOWED_ANONYMOUS:
                raise SqlGuardError(f"function {node.name.lower()!r} is not in the allowed list")
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name}
    collisions = cte_names & ALLOWED_TABLES
    if collisions:
        # CTE名が実テーブル名と同じだと、書き換え対象の判定をすり抜けられるため拒否する。
        raise SqlGuardError(f"CTE name {sorted(collisions)[0]!r} collides with a table name")
    return cte_names


def _table_name(table: exp.Table, cte_names: set[str]) -> str | None:
    """実テーブル参照ならその名前、CTE参照・許可された関数ならNone。それ以外は拒否。"""
    if not isinstance(table.this, exp.Identifier):
        # FROM generate_series(...) のようなテーブル関数。構造チェックで関数名は検証済み。
        return None
    name = table.name.lower()
    if table.catalog or (table.db and table.db.lower() != "public"):
        raise SqlGuardError("only the public schema is accessible")
    if not table.db and name in cte_names:
        return None
    if name not in ALLOWED_TABLES:
        raise SqlGuardError(
            f"table {name!r} is not available; allowed: {', '.join(sorted(ALLOWED_TABLES))}"
        )
    return name


def _scoped_subquery(name: str, user_id: UUID) -> exp.Expr:
    uid = f"'{user_id}'"  # UUID型で検証済みの値のみ埋め込む(任意文字列は入らない)
    if name in USER_TABLES:
        sql = f"SELECT * FROM {name} WHERE user_id = {uid}"
    elif name == "workout_exercises":
        sql = (
            "SELECT e.* FROM workout_exercises e JOIN workout_logs w ON e.workout_log_id = w.id "
            f"WHERE w.user_id = {uid}"
        )
    else:  # workout_sets
        sql = (
            "SELECT s.* FROM workout_sets s "
            "JOIN workout_exercises e ON s.workout_exercise_id = e.id "
            f"JOIN workout_logs w ON e.workout_log_id = w.id WHERE w.user_id = {uid}"
        )
    return sqlglot.parse_one(sql, read="postgres")


def guard_sql(sql: str, user_id: UUID, *, max_rows: int) -> GuardedSql:
    """検証→ユーザー絞り込みの強制→LIMIT注入。拒否時は SqlGuardError。"""
    if not isinstance(user_id, UUID):
        raise TypeError("user_id must be a UUID")
    tree = _parse_single(sql.strip().rstrip(";"))
    cte_names = _check_structure(tree)

    def rewrite(node: exp.Expr) -> exp.Expr:
        if isinstance(node, exp.Table):
            name = _table_name(node, cte_names)
            if name is not None:
                alias = node.alias or name  # 元のエイリアスを保つので m.calories 等もそのまま動く
                return exp.Subquery(
                    this=_scoped_subquery(name, user_id),
                    alias=exp.TableAlias(this=exp.to_identifier(alias)),
                )
        return node

    # 書き換え前に全Tableを走査して検証(transformの途中で例外を出さず、確実に全件を検査する)。
    for t in tree.find_all(exp.Table):
        _table_name(t, cte_names)
    guarded = tree.transform(rewrite, copy=True)

    limit = max_rows + 1
    existing = guarded.args.get("limit")
    existing_n: int | None = None
    if existing is not None:
        expr = existing.args.get("expression")
        if isinstance(expr, exp.Literal) and expr.is_int:
            existing_n = int(expr.name)
    if existing_n is None or existing_n > limit:
        guarded = guarded.limit(limit, copy=False)  # type: ignore[attr-defined]
    return GuardedSql(sql=guarded.sql(dialect="postgres"), limit=limit)
