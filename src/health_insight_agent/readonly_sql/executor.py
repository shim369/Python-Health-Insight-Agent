"""読み取り専用SQLの実行(多層防御の2〜4層目)。

- トランザクションを READ ONLY にする(書き込みはDBが拒否)。
- statement_timeout / lock_timeout を `SET LOCAL` で設定(長いクエリ・ロック待ちを打ち切る)。
- 取得は max_rows+1 行まで(超過を検知するため1行多く取り、切り詰めを明示する)。
- 接続先は READONLY_DATABASE_URL(読み取り専用DBロール)を推奨(5層目)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

from health_insight_agent.database.connection import create_db_engine

_QUERY_CANCELED = "57014"
_INSUFFICIENT_PRIVILEGE = "42501"
_READ_ONLY_TRANSACTION = "25006"


class SqlExecutionError(Exception):
    """実行時エラー。`kind`は timeout / permission / read_only / db_error。"""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class SqlResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


class ReadOnlySqlExecutor:
    def __init__(
        self, url: str, *, statement_timeout_ms: int = 5000, engine: Engine | None = None
    ) -> None:
        self._engine = engine or create_db_engine(url)
        self._timeout_ms = statement_timeout_ms

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_ms / 1000

    def run(self, sql: str, *, max_rows: int) -> SqlResult:
        with self._engine.connect() as conn:
            try:
                # 最初の文である必要がある(トランザクション開始直後)。
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(self._timeout_ms)}")
                conn.exec_driver_sql("SET LOCAL lock_timeout = 1000")
                # '%'はpsycopgのプレースホルダ記号なので、LIKE '%x%' が壊れないよう二重にする。
                result = conn.exec_driver_sql(sql.replace("%", "%%"))
                columns = [str(c) for c in result.keys()]
                fetched = result.fetchmany(max_rows + 1)
            except DBAPIError as e:
                raise self._classify(e) from e
            finally:
                conn.rollback()  # 何も書かない。トランザクションは必ず破棄する
        return SqlResult(
            columns=columns,
            rows=[list(r) for r in fetched[:max_rows]],
            truncated=len(fetched) > max_rows,
        )

    @staticmethod
    def _classify(e: DBAPIError) -> SqlExecutionError:
        state = getattr(e.orig, "sqlstate", None)
        first_line = str(e.orig).strip().splitlines()[0][:200] if e.orig else "database error"
        if state == _QUERY_CANCELED:
            return SqlExecutionError("timeout", "query timed out")
        if state == _INSUFFICIENT_PRIVILEGE:
            return SqlExecutionError("permission", "permission denied for this table")
        if state == _READ_ONLY_TRANSACTION:
            return SqlExecutionError("read_only", "writes are not allowed")
        return SqlExecutionError("db_error", first_line)
