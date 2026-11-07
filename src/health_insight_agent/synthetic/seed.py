"""合成データをDBに投入するCLI。

    uv run python -m health_insight_agent.synthetic.seed --days 90 --seed 42

デモユーザー(DEMO_USER_ID)のデータだけを作り直す(他ユーザーの行は触らない)。
接続先は DATABASE_URL のみ。READONLY_DATABASE_URL(既存アプリ)には書き込まない。
"""

from __future__ import annotations

import argparse
from datetime import date

from health_insight_agent.config import load_settings
from health_insight_agent.local_time import local_today
from health_insight_agent.repositories.sqlalchemy.factory import build_uow_factory
from health_insight_agent.synthetic.generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="合成データをDBに投入する")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()

    settings = load_settings()
    ds = generate_dataset(
        settings.demo_user_id,
        end_date=args.end_date or local_today(),
        days=args.days,
        seed=args.seed,
    )
    with build_uow_factory(settings.database_url)() as uow:
        if uow.users.get(settings.demo_user_id) is None:
            uow.users.add(ds.user)
            uow.commit()  # users行を先に確定(外部キーのため)
        uow.purge_user_data(settings.demo_user_id)
        uow.sleep_logs.add_many(ds.sleep_logs)
        uow.meal_logs.add_many(ds.meal_logs)
        uow.workout_logs.add_many(ds.workout_logs)
        uow.health_logs.add_many(ds.health_logs)
        uow.weight_logs.add_many(ds.weight_logs)
        uow.daily_journals.add_many(ds.daily_journals)
        uow.commit()
    print(
        f"seeded user={settings.demo_user_id}: sleep={len(ds.sleep_logs)} meals={len(ds.meal_logs)}"
        f" workouts={len(ds.workout_logs)} health={len(ds.health_logs)}"
        f" weight={len(ds.weight_logs)} journals={len(ds.daily_journals)}"
    )


if __name__ == "__main__":
    main()
