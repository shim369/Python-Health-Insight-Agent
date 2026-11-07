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


def analyze_nutrition(f: NutritionFeatures) -> AnalysisResult:
    if f.total_calories == 0:
        return AnalysisResult(score=50.0, findings=["食事記録が不足しているため評価できません"])
    penalty = (
        _range_penalty(f.protein_ratio, 0.13, 0.20)
        + _range_penalty(f.fat_ratio, 0.20, 0.30)
        + _range_penalty(f.carbohydrate_ratio, 0.50, 0.65)
    )
    findings = []
    if f.protein_ratio < 0.13:
        findings.append("タンパク質の摂取割合が目安より低めです")
    if f.fat_ratio > 0.30:
        findings.append("脂質の摂取割合が目安より高めです")
    if not findings:
        findings.append("PFCバランスは目安の範囲内です")
    return AnalysisResult(score=clamp_score(100.0 - penalty * 300.0), findings=findings)


def analyze_weight(f: WeightFeatures) -> AnalysisResult:
    if f.change_rate is None:
        return AnalysisResult(score=50.0, findings=["体重記録が不足しているため評価できません"])
    excess = max(0.0, abs(f.change_rate) - 0.01)  # 週1%以内を「安定」とする暫定値
    if excess > 0:
        direction = "増加" if f.change_rate > 0 else "減少"
        findings = [f"体重が急激に{direction}しています(変化率 {f.change_rate:+.1%})"]
    else:
        findings = ["体重は安定して推移しています"]
    return AnalysisResult(score=clamp_score(100.0 - excess * 1000.0), findings=findings)


def analyze_sleep(f: SleepFeatures) -> AnalysisResult:
    if f.average_duration_minutes is None:
        return AnalysisResult(score=50.0, findings=["睡眠記録が不足しているため評価できません"])
    low, high = 360, 540  # 推奨レンジ(6〜9時間)
    avg = f.average_duration_minutes
    duration_penalty = (low - avg) * 0.3 if avg < low else (avg - high) * 0.3 if avg > high else 0.0
    score = clamp_score(100.0 - duration_penalty - f.insufficient_sleep_days * 5.0)
    findings = []
    if duration_penalty > 0:
        findings.append(f"平均睡眠時間が目安から外れています({avg:.0f}分)")
    if f.insufficient_sleep_days > 0:
        findings.append(f"睡眠不足の日が{f.insufficient_sleep_days}日あります")
    if not findings:
        findings.append("睡眠は良好な状態です")
    return AnalysisResult(score=score, findings=findings)


def analyze_workout(f: WorkoutFeatures) -> AnalysisResult:
    if f.session_count == 0:
        return AnalysisResult(score=50.0, findings=["運動記録が不足しているため評価できません"])
    missing = max(0, 3 - f.session_count)  # 目安は週3回
    findings = (
        [f"週あたりの運動回数が目安(3回)より{missing}回少ないです"]
        if missing > 0
        else ["運動頻度は目安を満たしています"]
    )
    if f.total_strength_volume_kg > 0:
        findings.append(f"筋力トレーニングの総ボリューム: {f.total_strength_volume_kg:.0f}kg")
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
