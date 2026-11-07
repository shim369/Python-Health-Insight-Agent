"""HealthScore(栄養・体重・睡眠・運動の4領域を0〜100で評価)。

出典: shim369/Python-Smart-Health-Intelligence-Platform の analysis/・features/ を移植
(requirements.md 決定4=案A。既存アプリは変更しない)。ロジック・閾値は同一で、
統合時は services/health_insight_service.py の実装を既存サービスのアダプタに差し替えるだけで済む。
スコアは独自の目安であり、医療診断・疾病予測ではない。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from health_insight_agent.domain.intensity import cardio_minutes, strength_volume_kg
from health_insight_agent.domain.models import MealLog, SleepLog, WeightLog, WorkoutLog


@dataclass(frozen=True, kw_only=True, slots=True)
class AnalysisResult:
    score: float
    """0(要改善)〜100(良好)。"""
    findings: list[str] = field(default_factory=list)


def _t(locale: str, ja: str, en: str) -> str:
    return en if locale == "en" else ja


def _not_enough(locale: str, ja_noun: str, en_noun: str) -> AnalysisResult:
    """記録が無いときは、中立の50点で「評価できない」と伝える。"""
    return AnalysisResult(
        score=50.0,
        findings=[
            _t(
                locale,
                f"{ja_noun}記録が不足しているため評価できません",
                f"Not enough {en_noun} records to evaluate",
            )
        ],
    )


def clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))


# ---------------- features(既存 features/*.py より) ----------------
_SUFFICIENT_SLEEP_MINUTES = 360
_MOVING_AVERAGE_WINDOW = 7


@dataclass(frozen=True, kw_only=True, slots=True)
class NutritionFeatures:
    total_calories: float
    protein_ratio: float
    fat_ratio: float
    carbohydrate_ratio: float


def compute_nutrition_features(meals: list[MealLog]) -> NutritionFeatures:
    protein = sum(m.protein_g for m in meals) * 4.0  # 4-9-4換算
    fat = sum(m.fat_g for m in meals) * 9.0
    carb = sum(m.carbohydrate_g for m in meals) * 4.0
    total = protein + fat + carb
    if total == 0:
        return NutritionFeatures(
            total_calories=sum(m.calories for m in meals),
            protein_ratio=0.0,
            fat_ratio=0.0,
            carbohydrate_ratio=0.0,
        )
    return NutritionFeatures(
        total_calories=sum(m.calories for m in meals),
        protein_ratio=protein / total,
        fat_ratio=fat / total,
        carbohydrate_ratio=carb / total,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class SleepFeatures:
    average_duration_minutes: float | None
    insufficient_sleep_days: int
    duration_stddev_minutes: float | None


def compute_sleep_features(logs: list[SleepLog]) -> SleepFeatures:
    if not logs:
        return SleepFeatures(
            average_duration_minutes=None, insufficient_sleep_days=0, duration_stddev_minutes=None
        )
    d = [s.duration_minutes for s in logs]
    return SleepFeatures(
        average_duration_minutes=sum(d) / len(d),
        insufficient_sleep_days=sum(1 for x in d if x < _SUFFICIENT_SLEEP_MINUTES),
        duration_stddev_minutes=statistics.stdev(d) if len(d) >= 2 else 0.0,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class WeightFeatures:
    moving_average_kg: float | None
    change_kg: float | None
    change_rate: float | None


def compute_weight_features(logs: list[WeightLog]) -> WeightFeatures:
    if not logs:
        return WeightFeatures(moving_average_kg=None, change_kg=None, change_rate=None)
    ordered = sorted(logs, key=lambda w: w.recorded_at)
    recent = ordered[-_MOVING_AVERAGE_WINDOW:]
    avg = sum(w.weight_kg for w in recent) / len(recent)
    if len(ordered) < 2:
        return WeightFeatures(moving_average_kg=avg, change_kg=None, change_rate=None)
    first, last = ordered[0].weight_kg, ordered[-1].weight_kg
    return WeightFeatures(
        moving_average_kg=avg,
        change_kg=last - first,
        change_rate=(last - first) / first if first != 0 else None,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkoutFeatures:
    session_count: int
    total_cardio_minutes: int
    total_strength_volume_kg: float


def compute_workout_features(logs: list[WorkoutLog]) -> WorkoutFeatures:
    return WorkoutFeatures(
        session_count=len(logs),
        total_cardio_minutes=sum(cardio_minutes(w) for w in logs),
        total_strength_volume_kg=sum(strength_volume_kg(w) for w in logs),
    )


# ---------------- analyzers(既存 analysis/*_analyzer.py より) ----------------
def _range_penalty(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def analyze_nutrition(f: NutritionFeatures, locale: str = "ja") -> AnalysisResult:
    if f.total_calories == 0:
        return AnalysisResult(
            score=50.0,
            findings=[
                _t(
                    locale,
                    "食事記録が不足しているため評価できません",
                    "Not enough meal records to evaluate",
                )
            ],
        )
    penalty = (
        _range_penalty(f.protein_ratio, 0.13, 0.20)
        + _range_penalty(f.fat_ratio, 0.20, 0.30)
        + _range_penalty(f.carbohydrate_ratio, 0.50, 0.65)
    )
    findings = []
    if f.protein_ratio < 0.13:
        findings.append(
            _t(
                locale,
                "タンパク質の摂取割合が目安より低めです",
                "Protein share is below the guideline range",
            )
        )
    if f.fat_ratio > 0.30:
        findings.append(
            _t(locale, "脂質の摂取割合が目安より高めです", "Fat share is above the guideline range")
        )
    if not findings:
        findings.append(
            _t(locale, "PFCバランスは目安の範囲内です", "PFC balance is within the guideline range")
        )
    return AnalysisResult(score=clamp_score(100.0 - penalty * 300.0), findings=findings)


def analyze_weight(f: WeightFeatures, locale: str = "ja") -> AnalysisResult:
    if f.change_rate is None:
        return AnalysisResult(
            score=50.0,
            findings=[
                _t(
                    locale,
                    "体重記録が不足しているため評価できません",
                    "Not enough weight records to evaluate",
                )
            ],
        )
    excess = max(0.0, abs(f.change_rate) - 0.01)  # 週1%以内を「安定」とする暫定値
    if excess > 0:
        up = f.change_rate > 0
        findings = [
            _t(
                locale,
                f"体重が急激に{'増加' if up else '減少'}しています(変化率 {f.change_rate:+.1%})",
                f"Weight is {'increasing' if up else 'decreasing'} rapidly "
                f"(change rate {f.change_rate:+.1%})",
            )
        ]
    else:
        findings = [_t(locale, "体重は安定して推移しています", "Weight is stable")]
    return AnalysisResult(score=clamp_score(100.0 - excess * 1000.0), findings=findings)


def analyze_sleep(f: SleepFeatures, locale: str = "ja") -> AnalysisResult:
    if f.average_duration_minutes is None:
        return AnalysisResult(
            score=50.0,
            findings=[
                _t(
                    locale,
                    "睡眠記録が不足しているため評価できません",
                    "Not enough sleep records to evaluate",
                )
            ],
        )
    low, high = 360, 540  # 推奨レンジ(6〜9時間)
    avg = f.average_duration_minutes
    duration_penalty = (low - avg) * 0.3 if avg < low else (avg - high) * 0.3 if avg > high else 0.0
    score = clamp_score(100.0 - duration_penalty - f.insufficient_sleep_days * 5.0)
    findings = []
    if duration_penalty > 0:
        findings.append(
            _t(
                locale,
                f"平均睡眠時間が目安から外れています({avg / 60:.1f}時間)",
                f"Average sleep ({avg / 60:.1f} h) is outside the guideline range",
            )
        )
    if f.insufficient_sleep_days > 0:
        findings.append(
            _t(
                locale,
                f"睡眠不足の日が{f.insufficient_sleep_days}日あります",
                f"{f.insufficient_sleep_days} day(s) with short sleep (under 6 h)",
            )
        )
    if not findings:
        findings.append(_t(locale, "睡眠は良好な状態です", "Sleep looks good"))
    return AnalysisResult(score=score, findings=findings)


def analyze_workout(f: WorkoutFeatures, locale: str = "ja") -> AnalysisResult:
    if f.session_count == 0:
        return AnalysisResult(
            score=50.0,
            findings=[
                _t(
                    locale,
                    "運動記録が不足しているため評価できません",
                    "Not enough workout records to evaluate",
                )
            ],
        )
    missing = max(0, 3 - f.session_count)  # 目安は週3回
    findings = (
        [
            _t(
                locale,
                f"週あたりの運動回数が目安(3回)より{missing}回少ないです",
                f"Workout frequency is {missing} session(s) below the guideline (3 per week)",
            )
        ]
        if missing > 0
        else [_t(locale, "運動頻度は目安を満たしています", "Workout frequency meets the guideline")]
    )
    if f.total_strength_volume_kg > 0:
        findings.append(
            _t(
                locale,
                f"筋力トレーニングの総ボリューム: {f.total_strength_volume_kg:.0f}kg",
                f"Total strength-training volume: {f.total_strength_volume_kg:.0f} kg",
            )
        )
    return AnalysisResult(score=clamp_score(100.0 - missing * 15.0), findings=findings)


_DEFAULT_WEIGHTS = {"nutrition": 0.25, "weight": 0.25, "sleep": 0.25, "workout": 0.25}


@dataclass(frozen=True, kw_only=True, slots=True)
class HealthScore:
    overall_score: float
    nutrition: AnalysisResult
    weight: AnalysisResult
    sleep: AnalysisResult
    workout: AnalysisResult


def compute_health_score(
    *,
    nutrition: AnalysisResult,
    weight: AnalysisResult,
    sleep: AnalysisResult,
    workout: AnalysisResult,
) -> HealthScore:
    w = _DEFAULT_WEIGHTS
    overall = (
        nutrition.score * w["nutrition"]
        + weight.score * w["weight"]
        + sleep.score * w["sleep"]
        + workout.score * w["workout"]
    )
    return HealthScore(
        overall_score=overall, nutrition=nutrition, weight=weight, sleep=sleep, workout=workout
    )
