"""自作モデルのツール: predict_sleep_duration / explain_sleep_factors。

返す内容はすべてコードが確定させる。LLMには「参考値であること」「モデルのMAEとBaseline比」
「モデルを使っていない場合はその旨」を必ず併記させる(prompts.MODEL_RULE)。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from health_insight_agent.ml.features import (
    FEATURE_LABELS_EN,
    FEATURE_LABELS_JA,
    MEAL_TIMING_FEATURES,  # noqa: F401  (評価で参照する定義の公開)
)
from health_insight_agent.ml.model_registry import ModelDecision
from health_insight_agent.ml.service import PredictionConditions, SleepModelService
from health_insight_agent.tools import charts
from health_insight_agent.tools.base import Tool, ToolArgs, ToolOutput

REFERENCE_LABEL = charts.tr("モデルによる参考値", "Model-based reference value")
DISCLAIMER = (
    "reference value from a small statistical model, not a measurement or medical advice; "
    "always state the model MAE and the improvement vs baseline when you report it"
)


class _PredictArgs(ToolArgs):
    last_meal_hour: float = Field(
        ge=0, le=30, description="Hour of day (24h, decimal) of the last meal. 22.5 = 22:30"
    )
    bedtime_hour: float = Field(
        ge=0,
        le=36,
        description="Planned bedtime as hours since midnight of the meal day. 24 = midnight, "
        "24.5 = 00:30. Must be later than last_meal_hour",
    )
    total_kcal: float | None = Field(
        default=None,
        ge=0,
        le=10000,
        description="Total kcal eaten that day (default: user's usual)",
    )
    workout_kcal: float = Field(
        default=0, ge=0, le=3000, description="Kcal burned by workouts that day"
    )
    evening_high_workout: bool = Field(
        default=False, description="True if a high-intensity workout was done after 20:00"
    )
    is_weekend: bool = Field(default=False, description="True if the sleep ends on Sat/Sun")


class _NoArgs(ToolArgs):
    pass


def _decision_json(d: ModelDecision) -> dict[str, Any]:
    def r(v: float | None, n: int = 3) -> float | None:
        return None if v is None else round(v, n)

    return {
        "adopted": d.adopted,
        "reason": d.reason,
        "mae_minutes": r(d.mae_minutes, 1),
        "baseline": d.baseline_name,
        "baseline_mae_minutes": r(d.baseline_mae_minutes, 1),
        "improvement_vs_baseline": r(d.improvement_rate),
        "overfit_gap": r(d.overfit_gap),
        "total_days": d.total_days,
        "usable_samples": d.usable_samples,
        "missing_rate": r(d.missing_rate),
    }


def build_model_tools(service: SleepModelService) -> list[Tool]:
    def predict(user_id: UUID, a: _PredictArgs) -> ToolOutput:
        res = service.predict(
            user_id,
            PredictionConditions(
                last_meal_hour=a.last_meal_hour,
                bedtime_hour=a.bedtime_hour,
                total_kcal=a.total_kcal,
                workout_kcal=a.workout_kcal,
                evening_high_workout=a.evening_high_workout,
                is_weekend=a.is_weekend,
            ),
        )
        d = res.decision
        data: dict[str, Any] = {
            "model_used": res.model_used,
            "reference_only": True,
            "model": _decision_json(d),
            "inputs_used": {k: round(v, 2) for k, v in res.inputs.items()},
            "defaulted_inputs": res.defaulted,
        }
        notes = [DISCLAIMER]
        out_charts: list[dict[str, Any]] = []
        if res.model_used and res.predicted_minutes is not None:
            data["predicted_sleep_minutes"] = round(res.predicted_minutes, 1)
            data["predicted_sleep_hours"] = round(res.predicted_minutes / 60, 2)
            mae = d.mae_minutes or 0.0
            gain = (d.improvement_rate or 0) * 100
            out_charts.append(
                charts.value_card(
                    charts.tr("明日の睡眠時間(予測)", "Tomorrow's sleep duration (prediction)"),
                    round(res.predicted_minutes / 60, 2),
                    unit=charts.tr("時間", "h"),
                    label=REFERENCE_LABEL,
                    note=charts.tr(
                        f"モデルの誤差(MAE)は約{mae:.0f}分。直近平均より{gain:.0f}%良い(Baseline比)",
                        f"Model error (MAE) is about {mae:.0f} min; {gain:.0f}% better than "
                        "the baseline",
                    ),
                )
            )
        else:
            fb = res.fallback
            data["predicted_sleep_minutes"] = None
            data["fallback"] = (
                None
                if fb is None
                else {
                    "source": "statistics (no model was used)",
                    "recent_mean_minutes": None
                    if fb.recent_mean_minutes is None
                    else round(fb.recent_mean_minutes, 1),
                    "recent_nights": fb.recent_n,
                    "late_meal_nights_mean_minutes": None
                    if fb.late_meal_mean_minutes is None
                    else round(fb.late_meal_mean_minutes, 1),
                    "late_meal_nights": fb.late_meal_n,
                    "other_nights_mean_minutes": None
                    if fb.other_mean_minutes is None
                    else round(fb.other_mean_minutes, 1),
                    "other_nights": fb.other_n,
                }
            )
            notes.append("NO MODEL WAS USED: tell the user the model was not used and why")
        return ToolOutput(data=data, rows_used=d.usable_samples, notes=notes, charts=out_charts)

    def explain(user_id: UUID, _: _NoArgs) -> ToolOutput:
        res = service.factors(user_id)
        d = res.decision
        data: dict[str, Any] = {"model_used": res.model_used, "model": _decision_json(d)}
        notes = [
            "importance = how much the model used a feature to predict; it is NOT a causal effect",
        ]
        out_charts: list[dict[str, Any]] = []
        if res.model_used:
            data["factors"] = [
                {
                    "feature": name,
                    "label": FEATURE_LABELS_JA.get(name, name),
                    "importance": round(share, 3),
                }
                for name, share in res.factors
            ]
            out_charts.append(
                charts.bar_chart(
                    charts.tr(
                        "予測に使われた要因の重要度(モデル。因果ではない)",
                        "Feature importance used by the model (not causal)",
                    ),
                    [
                        (
                            charts.tr(FEATURE_LABELS_JA.get(n, n), FEATURE_LABELS_EN.get(n, n)),
                            round(sh, 3),
                        )
                        for n, sh in res.factors
                    ],
                    y_label=charts.tr("重要度(合計1)", "Importance (sums to 1)"),
                )
            )
        else:
            data["factors"] = []
            data["fallback_correlations"] = [
                {
                    "feature": n,
                    "label": FEATURE_LABELS_JA.get(n, n),
                    "pearson_r": round(r, 3),
                    "n": cnt,
                    "interpretation": interp,
                }
                for n, r, cnt, interp in res.fallback_correlations
            ]
            notes.append("NO MODEL WAS USED: tell the user; these are plain correlations instead")
        return ToolOutput(data=data, rows_used=d.usable_samples, notes=notes, charts=out_charts)

    return [
        Tool(
            "predict_sleep_duration",
            "Predict the coming night's sleep duration from planned conditions (last meal time, "
            "bedtime, kcal, workout) using a small trained model. The result is a REFERENCE value: "
            "always report it as a reference value together with the model's MAE and the "
            "improvement vs baseline. If model_used is false, say that no model was used and give "
            "the fallback statistics instead.",
            _PredictArgs,
            predict,
        ),
        Tool(
            "explain_sleep_factors",
            "Which factors the sleep-duration model relied on most (importance shares). "
            "Not causal. If model_used is false, say no model was used; plain correlations "
            "are returned instead.",
            _NoArgs,
            explain,
        ),
    ]
