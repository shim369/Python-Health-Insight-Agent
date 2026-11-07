"""合成データが「仕込んだ因果パターン」を実際に含んでいることの検証。

評価(Phase 8)の正解はこのパターンなので、生成側が壊れると評価も無意味になる。
"""

from __future__ import annotations

import statistics
from datetime import timedelta
from uuid import UUID

import pytest

from health_insight_agent.domain.enums import MealType
from health_insight_agent.domain.intensity import workout_intensity
from health_insight_agent.local_time import local_date_of
from health_insight_agent.synthetic import patterns as p
from health_insight_agent.synthetic.generator import SyntheticDataset, generate_dataset
from tests.conftest import FIXED_END_DATE

USER = UUID("00000000-0000-4000-8000-000000000001")


@pytest.fixture(scope="module")
def ds() -> SyntheticDataset:
    return generate_dataset(USER, end_date=FIXED_END_DATE, days=90, seed=42)


def _corr(xs: list[float], ys: list[float]) -> float:
    return statistics.correlation(xs, ys)


def test_is_deterministic_for_same_seed(ds: SyntheticDataset) -> None:
    again = generate_dataset(USER, end_date=FIXED_END_DATE, days=90, seed=42)
    assert [s.bedtime for s in again.sleep_logs] == [s.bedtime for s in ds.sleep_logs]
    assert generate_dataset(USER, end_date=FIXED_END_DATE, days=90, seed=7).sleep_logs != (
        ds.sleep_logs
    )


def test_covers_90_days_with_small_realistic_gaps(ds: SyntheticDataset) -> None:
    meal_days = {local_date_of(m.consumed_at) for m in ds.meal_logs}
    assert len(meal_days) == 90
    assert len(ds.truths) == 89  # 最初の夜は要因が記録されていないので作らない
    # 睡眠欠損は約3%(0〜8件)。欠損率30%基準(Phase 9)を十分下回る。
    assert 80 <= len(ds.sleep_logs) <= 89
    assert all(local_date_of(m.consumed_at) <= FIXED_END_DATE for m in ds.meal_logs)


def test_all_records_belong_to_the_given_user(ds: SyntheticDataset) -> None:
    everything = [
        *ds.sleep_logs,
        *ds.meal_logs,
        *ds.workout_logs,
        *ds.health_logs,
        *ds.weight_logs,
        *ds.daily_journals,
    ]
    assert {r.user_id for r in everything} == {USER}


def test_one_journal_per_day(ds: SyntheticDataset) -> None:
    dates = [j.journal_date for j in ds.daily_journals]
    assert len(dates) == len(set(dates))  # (user_id, journal_date)一意制約に合う


def test_late_meal_label_matches_actual_meal_times(ds: SyntheticDataset) -> None:
    for t in ds.truths:
        assert t.late_meal == (t.bedtime - t.last_meal_at < timedelta(hours=p.LATE_MEAL_GAP_HOURS))


def _durations_by(ds: SyntheticDataset, flag: str, value: bool) -> list[int]:
    by_wake = {local_date_of(s.wake_time): s for s in ds.sleep_logs}
    return [
        by_wake[t.wake_date].duration_minutes
        for t in ds.truths
        if getattr(t, flag) is value and t.wake_date in by_wake
    ]


def test_pattern_late_meal_shortens_sleep(ds: SyntheticDataset) -> None:
    late = _durations_by(ds, "late_meal", True)
    normal = _durations_by(ds, "late_meal", False)
    assert len(late) >= 15 and len(normal) >= 30  # 比較に十分な件数
    diff = statistics.mean(normal) - statistics.mean(late)
    assert p.LATE_MEAL_SLEEP_PENALTY_MIN - 20 <= diff <= p.LATE_MEAL_SLEEP_PENALTY_MIN + 20


def test_pattern_evening_high_workout_delays_bedtime_not_duration(ds: SyntheticDataset) -> None:
    def bedtime_minutes(flag: bool) -> list[float]:
        return [
            (t.bedtime.hour * 60 + t.bedtime.minute) % (24 * 60) / 1.0
            for t in ds.truths
            if t.evening_high_workout is flag
        ]

    # JSTの時刻で比較する(0時を跨いだ就寝は+24h)
    def jst_minutes(flag: bool) -> list[float]:
        out = []
        for t in ds.truths:
            if t.evening_high_workout is not flag:
                continue
            local = t.bedtime.astimezone(t.last_meal_at.tzinfo)
            mins = local.hour * 60 + local.minute
            out.append(mins + (24 * 60 if local.date() > t.factor_date else 0))
        return out

    hi, other = jst_minutes(True), jst_minutes(False)
    assert len(hi) >= 8
    diff = statistics.mean(hi) - statistics.mean(other)
    assert (
        p.EVENING_HIGH_WORKOUT_BEDTIME_DELAY_MIN - 25
        <= diff
        <= p.EVENING_HIGH_WORKOUT_BEDTIME_DELAY_MIN + 25
    )


def test_pattern_short_sleep_raises_fatigue(ds: SyntheticDataset) -> None:
    by_wake = {local_date_of(s.wake_time): s.duration_minutes for s in ds.sleep_logs}
    pairs = [
        (by_wake[local_date_of(h.recorded_at)], h.fatigue_score)
        for h in ds.health_logs
        if local_date_of(h.recorded_at) in by_wake and h.fatigue_score is not None
    ]
    assert len(pairs) >= 50
    assert _corr([float(d) for d, _ in pairs], [float(f) for _, f in pairs]) < -0.3


def test_null_patterns_show_no_relationship(ds: SyntheticDataset) -> None:
    """仕込んでいない関係は弱いままであること(「ない」と答える評価用)。"""
    by_wake = {local_date_of(s.wake_time): s.duration_minutes for s in ds.sleep_logs}
    # 朝食カロリー(D) vs その夜の睡眠(D+1)
    breakfast = {
        local_date_of(m.consumed_at): m.calories
        for m in ds.meal_logs
        if m.meal_type == MealType.BREAKFAST
    }
    xs, ys = [], []
    for d, kcal in breakfast.items():
        wake = d + timedelta(days=1)
        if wake in by_wake:
            xs.append(kcal)
            ys.append(float(by_wake[wake]))
    assert abs(_corr(xs, ys)) < 0.3

    # 平日/休日の差(睡眠の所属日の曜日)
    weekday = [v for d, v in by_wake.items() if d.weekday() < 5]
    weekend = [v for d, v in by_wake.items() if d.weekday() >= 5]
    assert abs(statistics.mean(weekday) - statistics.mean(weekend)) < 25


def test_workout_intensity_is_consistent_with_derived_metric(ds: SyntheticDataset) -> None:
    assert ds.workout_logs
    intensities = {workout_intensity(w) for w in ds.workout_logs}
    assert len(intensities) == 3  # low/moderate/high が全て現れる
    # 3階層(種目・セット)が実際に生成されている
    assert any(e.sets for w in ds.workout_logs for e in w.exercises)
    assert any(e.duration_minutes for w in ds.workout_logs for e in w.exercises)
