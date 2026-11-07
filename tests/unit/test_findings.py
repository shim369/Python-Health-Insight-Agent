"""統計ツールの結果を平易な所見にする処理(日英・単位・強さ・信頼度)。"""

from __future__ import annotations

from typing import Any

from health_insight_agent.services.findings import (
    CorrelationFinding,
    build_findings,
    comparison_finding,
    correlation_finding,
)
from health_insight_agent.services.metrics import mean_display, metric_display
from health_insight_agent.tools.base import ToolExecution


def _corr(x: str, y: str, interp: str, r: float | None = -0.75, n: int = 85) -> dict[str, Any]:
    return {
        "x_metric": x,
        "y_metric": y,
        "n_pairs": n,
        "pearson_r": r,
        "interpretation": interp,
        "reliability": "ok",
    }


def test_flag_cause_reads_as_a_plain_sentence_in_both_languages() -> None:
    d = _corr("prev_day_late_meal", "sleep_duration_min", "strong_negative")
    ja, en = correlation_finding(d, "ja"), correlation_finding(d, "en")
    assert ja.verdict_text == (
        "強い負の相関: 「遅い食事(就寝3時間以内)」の日は「睡眠時間」が短い傾向"
    )
    assert en.verdict_text == (
        "Strong negative correlation: on days with “Late meal (within 3h of bed)”, "
        "“Sleep duration” tends to be shorter"
    )
    assert (
        "prev_day" not in ja.verdict_text + en.verdict_text + ja.title + en.title
    )  # 内部名は出さない
    assert (ja.tone, ja.strength, ja.r, ja.n) == ("negative", 0.75, -0.75, 85)
    assert ja.reliability_text == "信頼度: 高" and en.reliability_text == "Reliability: high"


def test_numeric_cause_and_metric_specific_words() -> None:
    pos = correlation_finding(
        _corr("prev_day_meal_to_bed_gap_h", "sleep_duration_min", "moderate_positive", 0.5), "ja"
    )
    assert (
        "「最後の食事から就寝までの時間」が大きいほど、「睡眠時間」は長くなる傾向"
        in pos.verdict_text
    )
    bed = correlation_finding(
        _corr("prev_day_evening_high_workout", "sleep_bedtime_hour", "weak_positive", 0.3), "ja"
    )
    assert "「就寝時刻」が遅い傾向" in bed.verdict_text  # 指標ごとに自然な言い方
    en = correlation_finding(
        _corr("sleep_duration_min", "same_day_fatigue_score", "moderate_negative", -0.5), "en"
    )
    assert (
        "the higher “Sleep duration” is, the lower “Fatigue score” tends to be" in en.verdict_text
    )


def test_no_relationship_and_insufficient_data_say_so_plainly() -> None:
    none = correlation_finding(
        _corr("same_day_weight_kg", "sleep_duration_min", "no_clear_relationship", 0.1), "ja"
    )
    assert (
        none.verdict_text == "「体重」と「睡眠時間」に明確な関係は見られません"
        and none.tone == "none"
    )
    few = correlation_finding(
        _corr("is_weekend", "sleep_duration_min", "insufficient_data", None, 4), "en"
    )
    assert "Too few records" in few.verdict_text and few.tone == "unknown"
    assert few.strength is None and few.r is None


def _cmp() -> dict[str, Any]:
    return {
        "condition": {"metric": "prev_day_late_meal", "op": "==", "value": 1},
        "n_group_a": 30,
        "n_group_b": 54,
        "metrics": {
            "sleep_duration_min": {
                "group_a": {"n": 30, "mean": 372.0},
                "group_b": {"n": 54, "mean": 432.0},
                "interpretation": "a_lower",
            },
            "prev_day_breakfast_kcal": {
                "group_a": {"n": 30, "mean": 400.0},
                "group_b": {"n": 54, "mean": 410.0},
                "interpretation": "no_clear_difference",
            },
        },
    }


def test_comparison_shows_sleep_in_hours_and_verdicts_in_words() -> None:
    f = comparison_finding(_cmp(), "ja")
    assert f.title == "「遅い食事(就寝3時間以内)」の日 vs それ以外の日"
    sleep, kcal = f.rows
    assert (sleep.mean_a, sleep.mean_b, sleep.unit) == (6.2, 7.2, "時間")  # 372分→6.2時間
    assert sleep.verdict_text == "グループAの方が短い" and sleep.tone == "negative"
    assert (kcal.unit, kcal.verdict_text, kcal.tone) == ("kcal", "差は見られません", "none")
    en = comparison_finding(_cmp(), "en")
    assert en.rows[0].unit == "h" and en.rows[0].verdict_text == "Group A is shorter"


def test_threshold_conditions_are_written_in_the_display_unit() -> None:
    d = _cmp()
    d["condition"] = {"metric": "sleep_duration_min", "op": "<", "value": 390}
    assert comparison_finding(d, "ja").title == "「睡眠時間」が6.5時間より小さい日 vs それ以外の日"
    assert "6.5 h" in comparison_finding(d, "en").title


def test_flag_means_are_shown_as_percentages() -> None:
    assert mean_display("prev_day_late_meal", 0.81) == (81.0, "%")
    assert mean_display("sleep_duration_min", 450.0) == (7.5, "時間")
    assert mean_display("sleep_duration_min", 450.0, "en") == (7.5, "h")
    assert metric_display("same_day_weight_kg") == (1.0, "kg")


def test_build_findings_only_uses_successful_statistics_tools() -> None:
    def ex(name: str, data: dict[str, Any], ok: bool = True) -> ToolExecution:
        return ToolExecution(name, {}, ok, {"ok": ok, "data": data}, 1.0)

    trace = [
        ex("correlate", _corr("is_weekend", "sleep_duration_min", "no_clear_relationship", 0.0)),
        ex("compare_days", _cmp()),
        ex("get_sleep_logs", {"rows": []}),
        ex("correlate", {}, ok=False),
    ]
    found = build_findings(trace, "en")
    assert [type(f).__name__ for f in found] == ["CorrelationFinding", "ComparisonFinding"]
    assert isinstance(found[0], CorrelationFinding)
