"""評価の前提(固定の合成データ・固定の「今日」)。

評価は再現できなければ意味がないので、シード・最終日・ユーザーを固定する。正解はこの
データ(と synthetic/patterns.py の仕込み)から決まる。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import UUID

from health_insight_agent.repositories.sqlalchemy.factory import build_uow_factory
from health_insight_agent.synthetic.generator import SyntheticDataset, generate_dataset

EVAL_USER_ID = UUID("00000000-0000-4000-8000-0000000000e1")
EVAL_EMAIL = "eval@example.com"
EVAL_TODAY = date(2026, 11, 7)
EVAL_DAYS = 90
EVAL_SEED = 42


@dataclass(frozen=True, slots=True)
class EvalContext:
    dataset: SyntheticDataset
    today: date = EVAL_TODAY


def build_context() -> EvalContext:
    """評価用の合成データ(DBに入れるものと同一)をメモリ上に再生成する。"""
    return EvalContext(
        generate_dataset(
            EVAL_USER_ID, end_date=EVAL_TODAY, days=EVAL_DAYS, seed=EVAL_SEED, email=EVAL_EMAIL
        )
    )


def seed_eval_data(database_url: str, ctx: EvalContext | None = None) -> EvalContext:
    """評価ユーザーのデータをDBに作り直す(冪等。デモユーザー等の他ユーザーには触れない)。"""
    ctx = ctx or build_context()
    ds = ctx.dataset
    with build_uow_factory(database_url)() as uow:
        if uow.users.get(EVAL_USER_ID) is None:
            uow.users.add(ds.user)
            uow.commit()
        uow.purge_user_data(EVAL_USER_ID)
        uow.sleep_logs.add_many(ds.sleep_logs)
        uow.meal_logs.add_many(ds.meal_logs)
        uow.workout_logs.add_many(ds.workout_logs)
        uow.health_logs.add_many(ds.health_logs)
        uow.weight_logs.add_many(ds.weight_logs)
        uow.daily_journals.add_many(ds.daily_journals)
        uow.commit()
    return ctx
