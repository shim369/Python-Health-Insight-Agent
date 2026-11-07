"""DB接続。Neon等のscale-to-zeroで、起動直後の接続失敗に再試行で対応する。"""

from __future__ import annotations

import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker


def create_db_engine(url: str) -> Engine:
    # pool_pre_ping: スリープ復帰直後の切断済みコネクションを検知して張り直す。
    return create_engine(url, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False: commit後もORMオブジェクトを読めるようにする。
    return sessionmaker(bind=engine, expire_on_commit=False)


def wait_for_database(
    engine: Engine, *, attempts: int = 5, base_delay_seconds: float = 1.0
) -> None:
    """DBが応答するまで指数バックオフで待つ(Neonの起動待ち用)。

    起動直後の1回目はタイムアウトしがちなため、アプリ起動時に1度呼ぶ想定。
    """
    for attempt in range(1, attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError:
            if attempt == attempts:
                raise
            time.sleep(base_delay_seconds * (2 ** (attempt - 1)))
