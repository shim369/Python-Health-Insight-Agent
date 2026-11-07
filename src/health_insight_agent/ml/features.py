"""睡眠予測の特徴量・サンプル生成・データ品質チェック。

目的変数: ある日(起床日X)の睡眠時間(分)。特徴量は「その睡眠の前日(X-1)の行動」と「直近の睡眠」。
既存アプリの ml/dataset.py と同じ考え方(最低60日・欠損率30%未満・欠損は評価の前に判定)。

★運動していない日を「欠損」にしない★ 既存アプリでは、運動時間がNone(=運動していない)を他の欠損と
同じに扱ったため、常にフォールバックになる不具合があった。ここでは、前日の食事が記録されている日は
「運動なし=0」とし(services/metrics.py の DayRow が既にそうなっている)、欠損判定の対象にしない。
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from health_insight_agent.services.metrics import DayRow

MIN_DAYS = 60
"""学習に進む最低のログ日数(requirements: 60日分以上)。"""
MAX_MISSING_RATE = 0.30
"""許容する欠損率の上限(requirements: 30%未満)。"""
MIN_SAMPLES = 30
"""交差検証が意味を持つための最低サンプル数(欠損除外後)。"""
BASELINE_WINDOW = 7
"""Baseline(直近N夜の平均)の窓。"""

FEATURE_NAMES: tuple[str, ...] = (
    "last_meal_hour",
    "bedtime_hour",
    "meal_to_bed_gap_h",
    "total_kcal",
    "workout_kcal",
    "evening_high_workout",
    "workout_intensity",
    "sleep_prev_night_min",
    "sleep_mean_3",
    "is_weekend",
)
FEATURE_LABELS_JA: dict[str, str] = {
    "last_meal_hour": "最後の食事の時刻",
    "bedtime_hour": "就寝時刻",
    "meal_to_bed_gap_h": "最後の食事から就寝までの時間",
    "total_kcal": "前日の総カロリー",
    "workout_kcal": "前日の運動の消費カロリー",
    "evening_high_workout": "前日夜(20時以降)の高強度運動",
    "workout_intensity": "前日の運動強度",
    "sleep_prev_night_min": "前の晩の睡眠時間",
    "sleep_mean_3": "直近3夜の平均睡眠時間",
    "is_weekend": "週末",
}
FEATURE_LABELS_EN: dict[str, str] = {
    "last_meal_hour": "Last meal time",
    "bedtime_hour": "Bedtime",
    "meal_to_bed_gap_h": "Time from last meal to bed",
    "total_kcal": "Previous day total kcal",
    "workout_kcal": "Previous day workout kcal",
    "evening_high_workout": "High-intensity workout after 20:00",
    "workout_intensity": "Previous day workout intensity",
    "sleep_prev_night_min": "Previous night's sleep",
    "sleep_mean_3": "Average sleep of the last 3 nights",
    "is_weekend": "Weekend",
}
MEAL_TIMING_FEATURES = frozenset({"last_meal_hour", "meal_to_bed_gap_h"})


@dataclass(frozen=True, slots=True)
class SleepSample:
    day: date
    features: tuple[float, ...]
    """FEATURE_NAMES の順。"""
    target: float
    """その日の睡眠時間(分)。"""
    baseline: float
    """直近7夜の平均(過去のみ。Baselineの予測値)。"""


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    total_days: int
    usable_samples: int
    missing_rate: float
    is_sufficient: bool
    reason: str
    reason_en: str = ""


def _mean(values: Sequence[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return statistics.fmean(xs) if xs else None


def _row_features(rows: Sequence[DayRow], i: int) -> tuple[tuple[float, ...], float] | None:
    """i番目の行の(特徴量, Baseline)。必須値が欠けていればNone。"""
    r = rows[i]
    if i < 1:
        return None  # 前の晩の睡眠が無い(先頭)
    prev_night = rows[i - 1].sleep_duration_min
    mean3 = _mean([x.sleep_duration_min for x in rows[max(0, i - 3) : i]])
    baseline = _mean([x.sleep_duration_min for x in rows[max(0, i - BASELINE_WINDOW) : i]])
    required = (
        r.prev_day_last_meal_hour,
        r.sleep_bedtime_hour,
        r.prev_day_meal_to_bed_gap_h,
        r.prev_day_total_kcal,
        r.prev_day_workout_kcal,  # 運動なしは0(欠損ではない)。Noneなのは前日の記録自体が無いとき
        r.prev_day_evening_high_workout,
        r.prev_day_workout_intensity,
        prev_night,
        mean3,
        r.is_weekend,
        baseline,
    )
    if any(v is None for v in required):
        return None
    assert prev_night is not None and mean3 is not None and baseline is not None
    assert r.prev_day_last_meal_hour is not None and r.sleep_bedtime_hour is not None
    assert r.prev_day_meal_to_bed_gap_h is not None and r.prev_day_total_kcal is not None
    assert r.prev_day_workout_kcal is not None and r.prev_day_evening_high_workout is not None
    assert r.prev_day_workout_intensity is not None and r.is_weekend is not None
    feats = (
        r.prev_day_last_meal_hour,
        r.sleep_bedtime_hour,
        r.prev_day_meal_to_bed_gap_h,
        r.prev_day_total_kcal,
        r.prev_day_workout_kcal,
        r.prev_day_evening_high_workout,
        r.prev_day_workout_intensity,
        prev_night,
        mean3,
        r.is_weekend,
    )
    return feats, baseline


def build_samples(rows: Sequence[DayRow]) -> tuple[list[SleepSample], DataQualityReport]:
    """日次テーブル(日付昇順・連続)から学習サンプルとデータ品質を作る。

    欠損の定義: 目的変数(その日の睡眠)か、必須の特徴量のどれかが無い日。
    欠損率 = 欠損の日数 / 全日数。
    """
    total = len(rows)
    samples: list[SleepSample] = []
    for i, r in enumerate(rows):
        if r.sleep_duration_min is None:
            continue
        got = _row_features(rows, i)
        if got is None:
            continue
        feats, baseline = got
        samples.append(SleepSample(r.date, feats, r.sleep_duration_min, baseline))
    missing_rate = 1 - len(samples) / total if total else 1.0
    if total == 0:
        ok, reason, reason_en = False, "データが0日分です", "No data (0 days)"
    elif total < MIN_DAYS:
        ok = False
        reason = f"データ量が不足しています({total}日 < 最低{MIN_DAYS}日)"
        reason_en = f"Not enough data ({total} days < minimum {MIN_DAYS} days)"
    elif missing_rate >= MAX_MISSING_RATE:
        ok = False
        reason = f"欠損率が高すぎます({missing_rate:.0%} >= {MAX_MISSING_RATE:.0%})"
        reason_en = f"Missing rate is too high ({missing_rate:.0%} >= {MAX_MISSING_RATE:.0%})"
    elif len(samples) < MIN_SAMPLES:
        ok = False
        reason = f"使えるサンプルが少なすぎます({len(samples)} < {MIN_SAMPLES})"
        reason_en = f"Too few usable samples ({len(samples)} < {MIN_SAMPLES})"
    else:
        ok = True
        reason = "データ量・欠損率は基準を満たしています"
        reason_en = "Data volume and missing rate meet the criteria"
    return samples, DataQualityReport(total, len(samples), missing_rate, ok, reason, reason_en)
