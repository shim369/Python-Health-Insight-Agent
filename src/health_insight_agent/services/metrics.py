"""日次テーブル(DayRow)と指標カタログ。

★重要な時間の約束★ DayRowは「睡眠の起床日 X」を1行とする(睡眠は起床日に属する)。
  - `sleep_*`: 日付Xの朝に起きた睡眠(=前夜の睡眠)。
  - `prev_day_*`: 日付X-1の食事・運動(=その睡眠の「前日の行動」)。因果の候補はこちら。
  - `same_day_*`: 日付Xの体調・体重など。
「昨夜の食事と睡眠の関係」を調べるには sleep_* と prev_day_* を組み合わせる。
この対応づけをLLMに任せず、カタログ側で固定する(日付ずれによる誤答を防ぐ)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from operator import attrgetter

from health_insight_agent.domain.enums import SleepQuality, WorkoutIntensity
from health_insight_agent.domain.intensity import estimated_kcal, workout_intensity
from health_insight_agent.domain.models import (
    HealthLog,
    MealLog,
    SleepLog,
    WeightLog,
    WorkoutLog,
)
from health_insight_agent.local_time import APP_TIMEZONE, local_date_of, local_datetime
from health_insight_agent.synthetic.patterns import EVENING_WORKOUT_START_HOUR, LATE_MEAL_GAP_HOURS

_QUALITY_SCORE = {
    SleepQuality.POOR: 1.0,
    SleepQuality.FAIR: 2.0,
    SleepQuality.GOOD: 3.0,
    SleepQuality.EXCELLENT: 4.0,
}
_INTENSITY_LEVEL = {
    WorkoutIntensity.LOW: 1.0,
    WorkoutIntensity.MODERATE: 2.0,
    WorkoutIntensity.HIGH: 3.0,
}


@dataclass(frozen=True, kw_only=True, slots=True)
class DayRow:
    """起床日Xの1行。値が無い(記録なし)ものはNone。"""

    date: date
    sleep_duration_min: float | None = None
    sleep_bedtime_hour: float | None = None
    """就寝時刻(前日0時からの経過時間)。23:30→23.5、翌0:30→24.5。"""
    sleep_quality_score: float | None = None
    prev_day_total_kcal: float | None = None
    prev_day_breakfast_kcal: float | None = None
    prev_day_last_meal_hour: float | None = None
    prev_day_meal_to_bed_gap_h: float | None = None
    prev_day_late_meal: float | None = None
    prev_day_workout_kcal: float | None = None
    prev_day_workout_intensity: float | None = None
    prev_day_evening_high_workout: float | None = None
    same_day_fatigue_score: float | None = None
    same_day_stress_score: float | None = None
    same_day_mental_score: float | None = None
    same_day_morning_condition: float | None = None
    same_day_weight_kg: float | None = None
    is_weekend: float | None = None


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    label_ja: str
    label_en: str
    unit: str
    description: str
    kind: str  # "numeric" | "flag"(0/1)
    getter: Callable[[DayRow], float | None]


def _m(
    name: str, label_ja: str, label_en: str, unit: str, desc: str, kind: str = "numeric"
) -> Metric:
    return Metric(name, label_ja, label_en, unit, desc, kind, attrgetter(name))


METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        _m("sleep_duration_min", "睡眠時間", "Sleep duration", "分", "その日(起床日)の睡眠時間"),
        _m(
            "sleep_bedtime_hour",
            "就寝時刻",
            "Bedtime",
            "時",
            "前夜の就寝時刻。24超は日付をまたいだ深夜",
        ),
        _m(
            "sleep_quality_score",
            "睡眠の質",
            "Sleep quality",
            "1-4",
            "主観の質。1=poor 4=excellent",
        ),
        _m(
            "prev_day_total_kcal",
            "前日の総カロリー",
            "Previous day total kcal",
            "kcal",
            "睡眠の前日の食事の合計",
        ),
        _m(
            "prev_day_breakfast_kcal",
            "前日の朝食カロリー",
            "Previous day breakfast kcal",
            "kcal",
            "睡眠の前日の朝食",
        ),
        _m(
            "prev_day_last_meal_hour",
            "前日の最後の食事時刻",
            "Previous day last meal time",
            "時",
            "睡眠の前日の最終食事",
        ),
        _m(
            "prev_day_meal_to_bed_gap_h",
            "最後の食事から就寝までの時間",
            "Time from last meal to bed",
            "時間",
            "睡眠の前日の最終食事→その夜の就寝",
        ),
        _m(
            "prev_day_late_meal",
            "遅い食事(就寝3時間以内)",
            "Late meal (within 3h of bed)",
            "0/1",
            "最後の食事が就寝の3時間以内なら1",
            "flag",
        ),
        _m(
            "prev_day_workout_kcal",
            "前日の運動消費カロリー",
            "Previous day workout kcal",
            "kcal",
            "前日の運動の推定消費",
        ),
        _m(
            "prev_day_workout_intensity",
            "前日の運動強度",
            "Previous day workout intensity",
            "0-3",
            "0=なし 1=低 2=中 3=高(前日の最大)",
        ),
        _m(
            "prev_day_evening_high_workout",
            "前日の夜(20時以降)の高強度運動",
            "High-intensity workout after 20:00 (previous day)",
            "0/1",
            "前日20時以降に高強度運動をしたら1",
            "flag",
        ),
        _m(
            "same_day_fatigue_score",
            "疲労スコア",
            "Fatigue score",
            "1-5",
            "その日の疲労(高いほど疲れ)",
        ),
        _m("same_day_stress_score", "ストレススコア", "Stress score", "1-5", "その日のストレス"),
        _m("same_day_mental_score", "メンタルスコア", "Mental score", "1-5", "その日のメンタル"),
        _m(
            "same_day_morning_condition", "朝の体調", "Morning condition", "1-5", "その日の朝の体調"
        ),
        _m("same_day_weight_kg", "体重", "Weight", "kg", "その日の体重"),
        _m(
            "is_weekend",
            "週末(起床日が土日)",
            "Weekend (wake-up on Sat/Sun)",
            "0/1",
            "起床日が土曜か日曜なら1",
            "flag",
        ),
    )
}

METRIC_NAMES: tuple[str, ...] = tuple(METRICS)

# 画面に出すときだけ単位を変える指標(値は分のまま保存・計算し、表示でだけ割る)。
# 「睡眠時間(分)」は読み取りにくいので、7.5 のような「時間」で見せる。
_DISPLAY: dict[str, tuple[float, str, str]] = {"sleep_duration_min": (60.0, "時間", "h")}


_UNIT_EN = {"分": "min", "時": "h", "時間": "h"}
"""日本語の単位 → 英語。英語表示に「時」「分」が残らないようにする。"""


def unit_text(unit: str, locale: str) -> str:
    return _UNIT_EN.get(unit, unit) if locale == "en" else unit


def metric_display(name: str, locale: str = "ja") -> tuple[float, str]:
    """(表示用の割る数, 表示用の単位)。変換が無い指標は(1.0, 元の単位)。"""
    if name in _DISPLAY:
        divisor, ja, en = _DISPLAY[name]
        return divisor, en if locale == "en" else ja
    m = METRICS.get(name)
    return 1.0, unit_text(m.unit if m else "", locale)


def display_value(name: str, value: float | None) -> float | None:
    """表示用の値(小数2桁まで)。"""
    if value is None:
        return None
    divisor, _ = metric_display(name)
    return round(value / divisor, 2)


def _hour_offset(base_day: date, when: datetime) -> float:
    """base_day の JST 0時から when までの経過時間(時)。24超は日付をまたいだ時刻。"""
    return (when - local_datetime(base_day, time.min)).total_seconds() / 3600


def build_day_rows(
    *,
    start: date,
    end: date,
    sleeps: list[SleepLog],
    meals: list[MealLog],
    workouts: list[WorkoutLog],
    health_logs: list[HealthLog],
    weights: list[WeightLog],
) -> list[DayRow]:
    """起床日[start, end]の各日のDayRowを作る(睡眠ログが無い日も行は作る)。

    meals/workoutsは前日(start-1)分から渡すこと。
    """
    sleep_by_wake: dict[date, SleepLog] = {}
    for s in sleeps:
        d = local_date_of(s.wake_time)
        cur = sleep_by_wake.get(d)
        if cur is None or s.duration_minutes > cur.duration_minutes:
            sleep_by_wake[d] = s  # 同日に複数(仮眠等)あれば最長を主睡眠とする
    meals_by_day: dict[date, list[MealLog]] = {}
    for m in meals:
        meals_by_day.setdefault(local_date_of(m.consumed_at), []).append(m)
    workouts_by_day: dict[date, list[WorkoutLog]] = {}
    for w in workouts:
        workouts_by_day.setdefault(local_date_of(w.performed_at), []).append(w)
    health_by_day: dict[date, list[HealthLog]] = {}
    for h in health_logs:
        health_by_day.setdefault(local_date_of(h.recorded_at), []).append(h)
    weight_by_day: dict[date, WeightLog] = {}
    for wl in sorted(weights, key=lambda x: x.recorded_at):
        weight_by_day[local_date_of(wl.recorded_at)] = wl

    def mean_of(vals: list[int | None]) -> float | None:
        xs = [v for v in vals if v is not None]
        return sum(xs) / len(xs) if xs else None

    rows: list[DayRow] = []
    day = start
    while day <= end:
        prev = day - timedelta(days=1)
        sleep = sleep_by_wake.get(day)
        pm = meals_by_day.get(prev, [])
        pw = workouts_by_day.get(prev, [])
        hl = health_by_day.get(day, [])

        bed_h = _hour_offset(prev, sleep.bedtime) if sleep else None
        last_meal_h = _hour_offset(prev, max(m.consumed_at for m in pm)) if pm else None
        gap = bed_h - last_meal_h if bed_h is not None and last_meal_h is not None else None
        breakfast = [m.calories for m in pm if m.meal_type.value == "breakfast"]
        has_prev = bool(pm)  # 前日の記録が全く無ければ「前日の行動」は不明(Noneにする)
        intensities = [_INTENSITY_LEVEL[workout_intensity(w)] for w in pw]
        evening_high = any(
            workout_intensity(w) == WorkoutIntensity.HIGH
            and w.performed_at.astimezone(APP_TIMEZONE).hour >= EVENING_WORKOUT_START_HOUR
            for w in pw
        )
        rows.append(
            DayRow(
                date=day,
                sleep_duration_min=float(sleep.duration_minutes) if sleep else None,
                sleep_bedtime_hour=bed_h,
                sleep_quality_score=(
                    _QUALITY_SCORE[sleep.quality] if sleep and sleep.quality else None
                ),
                prev_day_total_kcal=sum(m.calories for m in pm) if has_prev else None,
                prev_day_breakfast_kcal=sum(breakfast) if breakfast else None,
                prev_day_last_meal_hour=last_meal_h,
                prev_day_meal_to_bed_gap_h=gap,
                prev_day_late_meal=(float(gap < LATE_MEAL_GAP_HOURS) if gap is not None else None),
                # 運動なし=0(欠損ではない)。既存アプリで運動なしを欠損扱いして不具合になった教訓。
                prev_day_workout_kcal=sum(estimated_kcal(w) for w in pw) if has_prev else None,
                prev_day_workout_intensity=(max(intensities) if intensities else 0.0)
                if has_prev
                else None,
                prev_day_evening_high_workout=float(evening_high) if has_prev else None,
                same_day_fatigue_score=mean_of([h.fatigue_score for h in hl]),
                same_day_stress_score=mean_of([h.stress_score for h in hl]),
                same_day_mental_score=mean_of([h.mental_score for h in hl]),
                same_day_morning_condition=mean_of([h.morning_condition for h in hl]),
                same_day_weight_kg=weight_by_day[day].weight_kg if day in weight_by_day else None,
                is_weekend=float(day.weekday() >= 5),
            )
        )
        day += timedelta(days=1)
    return rows


def mean_display(name: str, mean: float | None, locale: str = "ja") -> tuple[float | None, str]:
    """平均値の(表示値, 単位)。0/1のフラグ指標の平均は「該当した日の割合」なので%で見せる。"""
    if mean is None:
        return None, metric_display(name, locale)[1]
    m = METRICS.get(name)
    if m is not None and m.kind == "flag":
        return round(mean * 100, 1), "%"
    divisor, unit = metric_display(name, locale)
    return round(mean / divisor, 2), unit
