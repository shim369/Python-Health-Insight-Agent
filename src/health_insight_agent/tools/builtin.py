"""組み込みツール(型付き・読み取り専用)。

数値・判定はここで確定させて返す。LLMは結果を言葉にするだけで、計算や有意性の判断はしない
(compare_days/correlate は `interpretation` と `reliability` を計算済みで返す)。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from health_insight_agent.domain.intensity import estimated_kcal, workout_intensity
from health_insight_agent.domain.models import MealLog
from health_insight_agent.local_time import local_date_of
from health_insight_agent.ml.service import SleepModelService
from health_insight_agent.rag.note_search import NoteSearchService
from health_insight_agent.readonly_sql.executor import ReadOnlySqlExecutor
from health_insight_agent.services import stats
from health_insight_agent.services.health_insight_service import HealthInsightPort
from health_insight_agent.services.insight_data_service import (
    OPERATORS,
    Condition,
    InsightDataService,
    InvalidRequestError,
)
from health_insight_agent.services.metrics import (
    METRIC_NAMES,
    METRICS,
    display_value,
    mean_display,
    metric_display,
)
from health_insight_agent.tools import charts
from health_insight_agent.tools.base import (
    Tool,
    ToolArgs,
    ToolOutput,
    ToolRegistry,
    current_locale,
    iso_jst,
)

MAX_ROWS = 200
_METRIC_HELP = "; ".join(f"{m.name}={m.label_en}" for m in METRICS.values())
_TIMING_HELP = (
    "TIMING: one row = one wake-up date X. sleep_* = the sleep that ended on morning X; "
    "prev_day_* = meals/workouts of day X-1 (the day BEFORE that sleep, i.e. candidate causes); "
    "same_day_* = condition/weight logged on day X."
)


class _NoteSearchArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=300, description="What to look for in the notes")
    top_k: int = Field(default=5, ge=1, le=10, description="How many notes to return")
    start_date: date | None = Field(default=None, description="Only notes on/after this date")
    end_date: date | None = Field(default=None, description="Only notes on/before this date")


class _Period(ToolArgs):
    start_date: date = Field(description="Start date (JST), YYYY-MM-DD, inclusive")
    end_date: date = Field(description="End date (JST), YYYY-MM-DD, inclusive")


class _LogArgs(_Period):
    limit: int = Field(default=60, ge=1, le=MAX_ROWS, description="Max rows to return")


class _NoArgs(ToolArgs):
    pass


class _AsOfArgs(ToolArgs):
    as_of_date: date | None = Field(
        default=None, description="Evaluate as of this date (JST). Default: today"
    )


class _ConditionModel(ToolArgs):
    metric: str = Field(description="Metric that defines the group of interest")
    op: str = Field(description=f"Comparison operator, one of {list(OPERATORS)}")
    value: float = Field(description="Threshold. For 0/1 flag metrics use 1 or 0")


class _CompareArgs(_Period):
    condition: _ConditionModel = Field(
        description="Days matching this condition form group A; the other days form group B"
    )
    compare_metrics: list[str] = Field(
        min_length=1, max_length=8, description="Metrics to compare between group A and group B"
    )


class _CorrelateArgs(_Period):
    x_metric: str = Field(description="First metric")
    y_metric: str = Field(description="Second metric")


def _r(v: float | None, nd: int = 3) -> float | None:
    return None if v is None else round(v, nd)


def _enum_metrics(schema: dict[str, Any]) -> None:
    """指標名・演算子をJSON Schemaのenumで縛る(LLMが存在しない指標名を作るのを防ぐ)。"""
    props = schema.get("properties", {})
    cond = props.get("condition", {}).get("properties")
    if cond:
        cond["metric"]["enum"] = list(METRIC_NAMES)
        cond["op"]["enum"] = list(OPERATORS)
    if "compare_metrics" in props:
        props["compare_metrics"]["items"]["enum"] = list(METRIC_NAMES)
    for k in ("x_metric", "y_metric"):
        if k in props:
            props[k]["enum"] = list(METRIC_NAMES)


def _daily_kcal(meals: list[MealLog]) -> list[tuple[str, float]]:
    totals: dict[str, float] = {}
    for m in meals:
        day = local_date_of(m.consumed_at).isoformat()
        totals[day] = totals.get(day, 0.0) + m.calories
    return sorted(totals.items())


def _comparison_json(c: stats.Comparison) -> dict[str, Any]:
    def g(s: stats.GroupStats) -> dict[str, Any]:
        return {"n": s.n, "mean": _r(s.mean), "std": _r(s.std), "median": _r(s.median)}

    return {
        "group_a": g(c.group_a),
        "group_b": g(c.group_b),
        "difference_a_minus_b": _r(c.difference),
        "cohens_d": _r(c.cohens_d),
        "p_value": _r(c.p_value, 4),
        "reliability": c.reliability,
        "interpretation": c.interpretation,
    }


def build_default_registry(
    data: InsightDataService,
    health: HealthInsightPort,
    notes: NoteSearchService | None = None,
    sql: ReadOnlySqlExecutor | None = None,
    model: SleepModelService | None = None,
) -> ToolRegistry:
    def overview(user_id: UUID, _: _NoArgs) -> ToolOutput:
        o = data.overview(user_id)
        tables = {
            name: {
                "count": b.count,
                # UTCのまま返すとLLMが日付を1日取り違えるため、JSTにして返す。
                "first": iso_jst(b.first) if isinstance(b.first, datetime) else None,
                "last": iso_jst(b.last) if isinstance(b.last, datetime) else None,
            }
            for name, b in o.tables.items()
        }
        return ToolOutput(
            data={"today_jst": o.today.isoformat(), "tables": tables},
            rows_used=sum(b.count for b in o.tables.values()),
            notes=["counts are for all time, not a period"],
        )

    def sleep_logs(user_id: UUID, a: _LogArgs) -> ToolOutput:
        logs = data.sleep_logs(user_id, a.start_date, a.end_date)
        rows = [
            {
                "wake_date": local_date_of(s.wake_time).isoformat(),
                "bedtime": iso_jst(s.bedtime),
                "wake_time": iso_jst(s.wake_time),
                "duration_min": s.duration_minutes,
                "quality": s.quality.value if s.quality else None,
            }
            for s in logs[: a.limit]
        ]
        return ToolOutput(
            data={"rows": rows, "total": len(logs), "truncated": len(logs) > a.limit},
            rows_used=len(rows),
            period=f"{a.start_date}..{a.end_date}",
            notes=["sleep belongs to its wake-up date"],
            charts=[
                charts.line_chart(
                    charts.tr("睡眠時間(起床日ごと)", "Sleep duration by wake-up date"),
                    [
                        (local_date_of(x.wake_time).isoformat(), round(x.duration_minutes / 60, 2))
                        for x in logs[: a.limit]
                    ],
                    y_label=charts.tr("時間", "h"),
                )
            ]
            if rows
            else [],
        )

    def meal_logs(user_id: UUID, a: _LogArgs) -> ToolOutput:
        logs = data.meal_logs(user_id, a.start_date, a.end_date)
        rows = [
            {
                "consumed_at": iso_jst(m.consumed_at),
                "meal_type": m.meal_type.value,
                "calories": m.calories,
                "protein_g": m.protein_g,
                "fat_g": m.fat_g,
                "carbohydrate_g": m.carbohydrate_g,
            }
            for m in logs[: a.limit]
        ]
        return ToolOutput(
            data={"rows": rows, "total": len(logs), "truncated": len(logs) > a.limit},
            rows_used=len(rows),
            period=f"{a.start_date}..{a.end_date}",
            charts=[
                charts.bar_chart(
                    charts.tr("1日の摂取カロリー", "Daily calories"),
                    _daily_kcal(logs[: a.limit]),
                    y_label="kcal",
                )
            ]
            if rows
            else [],
        )

    def workout_logs(user_id: UUID, a: _LogArgs) -> ToolOutput:
        logs = data.workout_logs(user_id, a.start_date, a.end_date)
        rows = [
            {
                "performed_at": iso_jst(w.performed_at),
                "estimated_kcal": round(estimated_kcal(w)),
                "intensity": workout_intensity(w).value,  # 派生指標(domain/intensity.py)
                "exercises": [
                    {
                        "name": e.exercise_name,
                        "category": e.category.value,
                        "duration_min": e.duration_minutes,
                        "sets": [{"reps": s.reps, "weight_kg": s.weight_kg} for s in e.sets],
                    }
                    for e in w.exercises
                ],
            }
            for w in logs[: a.limit]
        ]
        return ToolOutput(
            data={"rows": rows, "total": len(logs), "truncated": len(logs) > a.limit},
            rows_used=len(rows),
            period=f"{a.start_date}..{a.end_date}",
            notes=["intensity is derived (low/moderate/high), not stored"],
            charts=[
                charts.bar_chart(
                    charts.tr(
                        "運動セッションの推定消費カロリー", "Estimated kcal per workout session"
                    ),
                    [
                        (iso_jst(w.performed_at)[:10], round(estimated_kcal(w)))
                        for w in logs[: a.limit]
                    ],
                    y_label="kcal",
                )
            ]
            if rows
            else [],
        )

    def compare_days(user_id: UUID, a: _CompareArgs) -> ToolOutput:
        r = data.compare_days(
            user_id,
            a.start_date,
            a.end_date,
            Condition(a.condition.metric, a.condition.op, a.condition.value),
            a.compare_metrics,
        )
        notes = [
            "group A = days matching the condition; group B = the other days (with data)",
            "interpretation/reliability are computed in code; do not re-judge significance",
        ]
        if r.n_matching == 0 or r.n_other == 0:
            notes.append("one group is empty, so no comparison is possible")
        return ToolOutput(
            data={
                "condition": {
                    "metric": r.condition.metric,
                    "op": r.condition.op,
                    "value": r.condition.value,
                },
                "n_days_in_period": r.n_days_in_period,
                "n_days_with_condition_data": r.n_days_with_condition_data,
                "n_group_a": r.n_matching,
                "n_group_b": r.n_other,
                "group_a_dates": [d.isoformat() for d in r.matching_dates],
                "metrics": {m.metric: _comparison_json(m.comparison) for m in r.metrics},
            },
            rows_used=r.n_days_with_condition_data,
            period=f"{a.start_date}..{a.end_date}",
            notes=notes,
            charts=[
                charts.bar_chart(
                    charts.tr(
                        f"{METRICS[m.metric].label_ja}({mean_display(m.metric, None, 'ja')[1]})",
                        f"{METRICS[m.metric].label_en} ({mean_display(m.metric, None, 'en')[1]})",
                    ),
                    [
                        (
                            charts.tr(
                                f"条件に該当 n={m.comparison.group_a.n}",
                                f"Matching n={m.comparison.group_a.n}",
                            ),
                            mean_display(m.metric, m.comparison.group_a.mean)[0] or 0.0,
                        ),
                        (
                            charts.tr(
                                f"それ以外 n={m.comparison.group_b.n}",
                                f"Others n={m.comparison.group_b.n}",
                            ),
                            mean_display(m.metric, m.comparison.group_b.mean)[0] or 0.0,
                        ),
                    ],
                    y_label=charts.tr(
                        mean_display(m.metric, None, "ja")[1], mean_display(m.metric, None, "en")[1]
                    ),
                )
                for m in r.metrics
                if m.comparison.group_a.n and m.comparison.group_b.n
            ][:6],
        )

    def correlate(user_id: UUID, a: _CorrelateArgs) -> ToolOutput:
        r = data.correlate(user_id, a.start_date, a.end_date, a.x_metric, a.y_metric)
        c = r.correlation
        return ToolOutput(
            data={
                "x_metric": r.x,
                "y_metric": r.y,
                "n_pairs": c.n,
                "pearson_r": _r(c.pearson_r, 4),
                "spearman_rho": _r(c.spearman_rho, 4),
                "p_value": _r(c.p_value, 4),
                "strength": c.strength,
                "reliability": c.reliability,
                "interpretation": c.interpretation,
            },
            rows_used=c.n,
            period=f"{a.start_date}..{a.end_date}",
            notes=[
                "correlation is not causation",
                f"fewer than {stats.MIN_PAIRS_N} pairs => insufficient_data; "
                f"fewer than {stats.LOW_PAIRS_N} => low reliability",
            ],
            charts=[
                charts.scatter_chart(
                    charts.tr(
                        f"{METRICS[r.x].label_ja} × {METRICS[r.y].label_ja}",
                        f"{METRICS[r.x].label_en} × {METRICS[r.y].label_en}",
                    ),
                    # 睡眠時間は「分」ではなく「時間」で描く(7.5 のほうが読み取りやすい)
                    [
                        (display_value(r.x, p[1]) or 0.0, display_value(r.y, p[2]) or 0.0)
                        for p in r.pairs
                    ],
                    x_label=charts.tr(
                        f"{METRICS[r.x].label_ja}({metric_display(r.x, 'ja')[1]})",
                        f"{METRICS[r.x].label_en} ({metric_display(r.x, 'en')[1]})",
                    ),
                    y_label=charts.tr(
                        f"{METRICS[r.y].label_ja}({metric_display(r.y, 'ja')[1]})",
                        f"{METRICS[r.y].label_en} ({metric_display(r.y, 'en')[1]})",
                    ),
                )
            ]
            if r.pairs
            else [],
        )

    def health_score(user_id: UUID, a: _AsOfArgs) -> ToolOutput:
        rep = health.report(user_id, a.as_of_date or data.today(), current_locale())
        s = rep.health_score
        domains = {
            name: {"score": round(res.score, 1), "findings": res.findings}
            for name, res in (
                ("nutrition", s.nutrition),
                ("weight", s.weight),
                ("sleep", s.sleep),
                ("workout", s.workout),
            )
        }
        return ToolOutput(
            data={"as_of": rep.as_of.isoformat(), "overall": round(s.overall_score, 1), **domains},
            rows_used=0,
            notes=[
                "windows: meals/sleep/workouts last 7 days, weight last 30 days",
                "score 0-100 is this app's own guide, not a medical measure",
            ],
            charts=[
                charts.bar_chart(
                    charts.tr("ヘルススコア(0-100)", "Health score (0-100)"),
                    [
                        (charts.tr("総合", "Overall"), round(s.overall_score, 1)),
                        (charts.tr("栄養", "Nutrition"), round(s.nutrition.score, 1)),
                        (charts.tr("体重", "Weight"), round(s.weight.score, 1)),
                        (charts.tr("睡眠", "Sleep"), round(s.sleep.score, 1)),
                        (charts.tr("運動", "Workout"), round(s.workout.score, 1)),
                    ],
                    y_label=charts.tr("点", "points"),
                )
            ],
        )

    def recommendations(user_id: UUID, a: _AsOfArgs) -> ToolOutput:
        rep = health.report(user_id, a.as_of_date or data.today(), current_locale())
        return ToolOutput(
            data={
                "as_of": rep.as_of.isoformat(),
                "overall": round(rep.health_score.overall_score, 1),
                "recommendations": [
                    {"domain": r.domain, "message": r.message, "priority": r.priority}
                    for r in rep.recommendations
                ],
            },
            rows_used=0,
            notes=["rule-based suggestions, not medical advice"],
        )

    def search_notes(user_id: UUID, a: _NoteSearchArgs) -> ToolOutput:
        assert notes is not None
        if a.start_date and a.end_date and a.end_date < a.start_date:
            raise InvalidRequestError("end_date must be on or after start_date")
        hits, total = notes.search(
            user_id, a.query, top_k=a.top_k, start=a.start_date, end=a.end_date
        )
        return ToolOutput(
            data={
                "total_notes_indexed": total,
                "hits": [
                    {"date": h.note_date.isoformat(), "text": h.text, "score": round(h.score, 3)}
                    for h in hits
                ],
            },
            rows_used=len(hits),
            notes=[
                "semantic similarity search over the user's own reflection notes",
                "score is similarity, not relevance or probability; judge relevance from the text",
                "quote only these notes; no hits does not prove an event did not happen",
            ],
        )

    reg = ToolRegistry()
    for t in (
        Tool(
            "get_data_overview",
            "Counts and first/last dates of each log type, and today's date (JST). "
            "Call this first when unsure what data exists or which dates to query.",
            _NoArgs,
            overview,
        ),
        Tool(
            "get_sleep_logs",
            "Raw sleep logs in a date range (bedtime, wake time, duration, quality).",
            _LogArgs,
            sleep_logs,
        ),
        Tool(
            "get_meal_logs",
            "Raw meal logs in a date range (time, type, calories, PFC).",
            _LogArgs,
            meal_logs,
        ),
        Tool(
            "get_workout_logs",
            "Raw workout sessions in a date range, with exercises and sets, "
            "plus derived intensity and estimated kcal.",
            _LogArgs,
            workout_logs,
        ),
        Tool(
            "compare_days",
            "Split days into group A (matching a condition) and group B (the rest) and compare "
            f"metrics between them. Use it to find what days with bad sleep have in common. "
            f"{_TIMING_HELP} Valid metrics: {_METRIC_HELP}.",
            _CompareArgs,
            compare_days,
            _enum_metrics,
        ),
        Tool(
            "correlate",
            "Correlation (Pearson/Spearman, n, p-value, computed interpretation) between two "
            f"metrics. Small n weakens the conclusion. {_TIMING_HELP} "
            f"Valid metrics: {_METRIC_HELP}.",
            _CorrelateArgs,
            correlate,
            _enum_metrics,
        ),
        Tool(
            "get_health_score",
            "The app's rule-based health score (0-100) for nutrition, weight, sleep, workout.",
            _AsOfArgs,
            health_score,
        ),
        Tool(
            "get_recommendations",
            "Rule-based improvement suggestions derived from the health score.",
            _AsOfArgs,
            recommendations,
        ),
    ):
        reg.register(t)
    if notes is not None:
        reg.register(
            Tool(
                "search_notes",
                "Semantic search over the person's own daily reflection notes (diary-like text). "
                "Use for questions about what they wrote or felt. Returns dated original text.",
                _NoteSearchArgs,
                search_notes,
            )
        )
    if sql is not None:
        from health_insight_agent.tools.sql_tool import build_sql_tool

        reg.register(build_sql_tool(sql))
    if model is not None:
        from health_insight_agent.tools.model_tools import build_model_tools

        for t in build_model_tools(model):
            reg.register(t)
    return reg
