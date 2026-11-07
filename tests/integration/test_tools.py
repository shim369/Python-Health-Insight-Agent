"""ツールを実DB(合成データ)で通しで検証。仕込みパターンを「見つけられる」ことの確認。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast
from uuid import UUID

import pytest

from health_insight_agent.services.health_insight_service import LocalHealthInsightService
from health_insight_agent.services.insight_data_service import InsightDataService
from health_insight_agent.tools.base import ToolRegistry
from health_insight_agent.tools.builtin import build_default_registry
from tests.conftest import FIXED_END_DATE

pytestmark = pytest.mark.integration

START = (FIXED_END_DATE - timedelta(days=89)).isoformat()
END = FIXED_END_DATE.isoformat()


@pytest.fixture
def env(seeded: tuple[Any, UUID, UUID]) -> tuple[ToolRegistry, UUID, UUID]:
    factory, uid, other = seeded
    reg = build_default_registry(
        InsightDataService(factory, clock=lambda: FIXED_END_DATE),
        LocalHealthInsightService(factory),
    )
    return reg, uid, other


def _run(
    env: tuple[ToolRegistry, UUID, UUID], name: str, args: dict[str, Any], *, other: bool = False
) -> dict[str, Any]:
    reg, uid, oth = env
    ex = reg.execute(oth if other else uid, name, args)
    return cast(dict[str, Any], ex.content)


def test_compare_days_finds_the_late_meal_effect(env: Any) -> None:
    out = _run(
        env,
        "compare_days",
        {
            "start_date": START,
            "end_date": END,
            "condition": {"metric": "prev_day_late_meal", "op": "==", "value": 1},
            "compare_metrics": ["sleep_duration_min", "prev_day_breakfast_kcal"],
        },
    )
    assert out["ok"]
    sleep = out["data"]["metrics"]["sleep_duration_min"]
    assert sleep["interpretation"] == "a_lower" and sleep["reliability"] == "ok"
    assert -85 < sleep["difference_a_minus_b"] < -35  # 仕込みは約-60分
    # 仕込んでいない朝食カロリーには差が出ない(でっち上げないこと)
    assert out["data"]["metrics"]["prev_day_breakfast_kcal"]["interpretation"] == (
        "no_clear_difference"
    )
    assert out["evidence"]["rows_used"] > 50


def test_correlate_finds_fatigue_and_rejects_null_patterns(env: Any) -> None:
    fatigue = _run(
        env,
        "correlate",
        {
            "start_date": START,
            "end_date": END,
            "x_metric": "sleep_duration_min",
            "y_metric": "same_day_fatigue_score",
        },
    )["data"]
    assert fatigue["interpretation"].endswith("negative") and fatigue["n_pairs"] >= 50
    for x in ("prev_day_breakfast_kcal", "same_day_weight_kg", "is_weekend"):
        r = _run(
            env,
            "correlate",
            {"start_date": START, "end_date": END, "x_metric": x, "y_metric": "sleep_duration_min"},
        )["data"]
        assert r["interpretation"] == "no_clear_relationship", x


def test_bedtime_delay_by_evening_high_workout(env: Any) -> None:
    out = _run(
        env,
        "compare_days",
        {
            "start_date": START,
            "end_date": END,
            "condition": {"metric": "prev_day_evening_high_workout", "op": "==", "value": 1},
            "compare_metrics": ["sleep_bedtime_hour", "sleep_duration_min"],
        },
    )["data"]["metrics"]
    assert out["sleep_bedtime_hour"]["interpretation"] == "a_higher"
    assert 0.4 < out["sleep_bedtime_hour"]["difference_a_minus_b"] < 1.3  # 約50分


def test_small_period_is_insufficient_not_a_conclusion(env: Any) -> None:
    short_start = (FIXED_END_DATE - timedelta(days=4)).isoformat()
    r = _run(
        env,
        "correlate",
        {
            "start_date": short_start,
            "end_date": END,
            "x_metric": "prev_day_late_meal",
            "y_metric": "sleep_duration_min",
        },
    )["data"]
    assert r["interpretation"] == "insufficient_data" and r["pearson_r"] is None


def test_logs_overview_and_scores(env: Any) -> None:
    sleep = _run(env, "get_sleep_logs", {"start_date": START, "end_date": END, "limit": 10})
    assert len(sleep["data"]["rows"]) == 10 and sleep["data"]["truncated"] is True
    assert sleep["data"]["total"] >= 80
    workouts = _run(env, "get_workout_logs", {"start_date": START, "end_date": END, "limit": 200})
    first = workouts["data"]["rows"][0]
    assert first["intensity"] in {"low", "moderate", "high"} and first["exercises"]
    assert _run(env, "get_meal_logs", {"start_date": START, "end_date": END})["ok"]
    ov = _run(env, "get_data_overview", {})["data"]
    assert ov["tables"]["sleep_logs"]["count"] >= 80 and ov["today_jst"] == END
    hs = _run(env, "get_health_score", {})["data"]
    assert 0 <= hs["overall"] <= 100 and {"nutrition", "weight", "sleep", "workout"} <= set(hs)
    recs = _run(env, "get_recommendations", {})["data"]["recommendations"]
    assert recs == sorted(recs, key=lambda r: -r["priority"])


def test_other_user_never_sees_first_users_data(env: Any) -> None:
    mine = _run(env, "get_data_overview", {})["data"]["tables"]
    theirs = _run(env, "get_data_overview", {}, other=True)["data"]["tables"]
    assert theirs["sleep_logs"]["count"] < mine["sleep_logs"]["count"]  # 20日分だけ
    rows = _run(
        env, "get_sleep_logs", {"start_date": START, "end_date": END, "limit": 200}, other=True
    )
    assert rows["data"]["total"] == theirs["sleep_logs"]["count"]


def test_invalid_requests_are_errors_for_the_llm_to_fix(env: Any) -> None:
    reversed_period = _run(env, "get_sleep_logs", {"start_date": END, "end_date": START})
    assert not reversed_period["ok"] and "end_date" in reversed_period["error"]
    bad_metric = _run(
        env,
        "correlate",
        {
            "start_date": START,
            "end_date": END,
            "x_metric": "mood",
            "y_metric": "sleep_duration_min",
        },
    )
    assert not bad_metric["ok"] and "unknown metric" in bad_metric["error"]


def test_tools_attach_charts_without_adding_them_to_llm_content(seeded: Any) -> None:
    factory, uid, _ = seeded
    reg = build_default_registry(
        InsightDataService(factory, clock=lambda: FIXED_END_DATE),
        LocalHealthInsightService(factory),
    )
    args = {"start_date": START, "end_date": END}
    cmp = reg.execute(
        uid,
        "compare_days",
        {
            **args,
            "condition": {"metric": "prev_day_late_meal", "op": "==", "value": 1},
            "compare_metrics": ["sleep_duration_min"],
        },
    )
    assert cmp.charts[0]["kind"] == "bar" and "charts" not in str(cmp.content)
    corr = reg.execute(
        uid,
        "correlate",
        {**args, "x_metric": "sleep_duration_min", "y_metric": "same_day_fatigue_score"},
    )
    assert corr.charts[0]["kind"] == "scatter"
    assert len(corr.charts[0]["series"][0]["points"]) == corr.content["data"]["n_pairs"]
    assert reg.execute(uid, "get_sleep_logs", args).charts[0]["kind"] == "line"
    assert reg.execute(uid, "get_meal_logs", args).charts[0]["kind"] == "bar"
    assert reg.execute(uid, "get_workout_logs", args).charts[0]["kind"] == "bar"
    assert len(reg.execute(uid, "get_health_score", {}).charts[0]["series"][0]["points"]) == 5
