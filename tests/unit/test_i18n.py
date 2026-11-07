"""日英切替: グラフ文言の解決と、キー無しFake LLMの英語対応。"""

from __future__ import annotations

from datetime import date

from health_insight_agent.agent.prompts import DISCLAIMERS, build_system_prompt
from health_insight_agent.ai.protocols import Message, ToolCall, ToolResultPart
from health_insight_agent.ai.rule_based_fake import RuleBasedFakeLLM, _plan
from health_insight_agent.domain.enums import Locale
from health_insight_agent.services.metrics import METRICS
from health_insight_agent.tools import charts

TODAY = date(2026, 11, 7)


def test_localize_resolves_nested_language_pairs_only() -> None:
    chart = charts.bar_chart(
        charts.tr("睡眠", "Sleep"),
        [(charts.tr("全体", "Overall"), 1.0), ("2026-11-07", 2.0)],
        y_label=charts.tr("分", "min"),
    )
    ja, en = charts.localize(chart, "ja"), charts.localize(chart, "en")
    assert ja["title"] == "睡眠" and en["title"] == "Sleep"
    assert [p["x"] for p in en["series"][0]["points"]] == ["Overall", "2026-11-07"]
    assert en["series"][0]["name"] == "min"  # 系列名もy軸ラベルから解決される
    assert chart["title"] == {"ja": "睡眠", "en": "Sleep"}  # 元のデータは変更しない
    # 言語キーが同じに見えても、2キー以外の辞書(通常のデータ)は壊さない
    assert charts.localize({"ja": "x", "en": "y", "other": 1}, "en") == {
        "ja": "x",
        "en": "y",
        "other": 1,
    }
    assert charts.localize("plain", "en") == "plain"


def test_every_metric_has_both_language_labels() -> None:
    assert all(m.label_ja and m.label_en for m in METRICS.values())
    assert all(m.label_en.isascii() for m in METRICS.values())  # 英語ラベルは英語


def test_english_questions_are_routed_to_the_right_tools() -> None:
    cases = {
        "What did my bad-sleep days last week have in common?": "compare_days",
        "Does eating late affect how long I sleep?": "correlate",
        "Is breakfast size related to my sleep?": "correlate",
        "What is my health score and what should I improve?": "get_health_score",
        "What did I write in my notes about being tired?": "search_notes",
        "Which factors affect my sleep the most?": "explain_sleep_factors",
        "How will I sleep tomorrow if I eat dinner late?": "predict_sleep_duration",
        "How many high-intensity workouts did I do?": "get_workout_logs",
    }
    for question, tool in cases.items():
        assert _plan(question, TODAY)[0].name == tool, question
    call = _plan("Does eating late affect how long I sleep?", TODAY)[0]
    assert (call.arguments["x_metric"], call.arguments["y_metric"]) == (
        "prev_day_late_meal",
        "sleep_duration_min",
    )


def _tool_result(name: str, data: dict) -> list[Message]:  # type: ignore[type-arg]
    content = {"ok": True, "data": data, "evidence": {"rows_used": 5, "period": "p"}}
    return [
        Message("user", "q"),
        Message("assistant", tool_calls=(ToolCall("c1", name, {}),)),
        Message("tool", tool_results=(ToolResultPart("c1", name, content),)),
    ]


CORRELATE = {
    "x_metric": "a",
    "y_metric": "b",
    "n_pairs": 30,
    "pearson_r": -0.5,
    "interpretation": "moderate_negative",
    "reliability": "ok",
}


def test_fake_answers_in_the_language_of_the_system_prompt() -> None:
    llm = RuleBasedFakeLLM()
    msgs = _tool_result("correlate", CORRELATE)
    ja = llm.generate(
        system=build_system_prompt(today=TODAY, locale=Locale.JA), messages=msgs, tools=[]
    )
    en = llm.generate(
        system=build_system_prompt(today=TODAY, locale=Locale.EN), messages=msgs, tools=[]
    )
    assert "中程度の負の相関" in ja.message.text and "因果を意味しません" in ja.message.text
    assert "Moderate negative correlation" in en.message.text
    assert "does not imply causation" in en.message.text
    assert not any(ord(ch) > 0x2FFF for ch in en.message.text)  # 英語の回答に日本語が混ざらない


def test_fake_model_answer_in_english_keeps_the_required_wording() -> None:
    data = {
        "model_used": True,
        "predicted_sleep_hours": 6.0,
        "predicted_sleep_minutes": 360.0,
        "model": {"mae_minutes": 21.5, "improvement_vs_baseline": 0.316, "reason": "r"},
    }
    en = RuleBasedFakeLLM().generate(
        system=build_system_prompt(today=TODAY, locale=Locale.EN),
        messages=_tool_result("predict_sleep_duration", data),
        tools=[],
    )
    text = en.message.text
    assert (
        "reference value" in text and "MAE" in text and "baseline" in text
    )  # 参考値・誤差・Baseline比
    assert "31.6%" in text


def test_fake_uses_the_latest_question_not_the_first_one_in_the_history() -> None:
    history = [
        Message("user", "What is my health score?"),
        Message("assistant", "Your score is 90."),
        Message("user", "Is breakfast size related to my sleep?"),
    ]
    resp = RuleBasedFakeLLM().generate(
        system=build_system_prompt(today=TODAY, locale=Locale.EN), messages=history, tools=[]
    )
    assert resp.message.tool_calls[0].name == "correlate"  # 最初の質問(スコア)に引きずられない


def test_disclaimers_exist_in_both_languages() -> None:
    # 両言語とも「※」で始める(UIが本文と分離して控えめに表示するための目印)
    assert DISCLAIMERS[Locale.JA].startswith("※") and DISCLAIMERS[Locale.EN].startswith("※")
    assert "medical" in DISCLAIMERS[Locale.EN]


# ---------------- ツール結果の言語(ロケールの受け渡し) ----------------
import re  # noqa: E402
from uuid import UUID  # noqa: E402

from health_insight_agent.analysis.health_score import (  # noqa: E402
    NutritionFeatures,
    SleepFeatures,
    WeightFeatures,
    WorkoutFeatures,
    analyze_nutrition,
    analyze_sleep,
    analyze_weight,
    analyze_workout,
)
from health_insight_agent.ml import features as F  # noqa: E402
from health_insight_agent.tools.base import (  # noqa: E402
    Tool,
    ToolArgs,
    ToolOutput,
    ToolRegistry,
    current_locale,
)

JAPANESE = re.compile(r"[぀-ヿ一-鿿]")


class _NoArgs(ToolArgs):
    pass


def _locale_tool() -> ToolRegistry:
    def handler(user_id: UUID, a: _NoArgs) -> ToolOutput:
        return ToolOutput(data={"locale": current_locale()}, rows_used=0)

    return ToolRegistry([Tool("which_locale", "d", _NoArgs, handler)])


def test_tool_handlers_see_the_callers_language_and_it_is_reset_afterwards() -> None:
    reg, user = _locale_tool(), UUID(int=1)
    assert reg.execute(user, "which_locale", {}).content["data"]["locale"] == "ja"  # 既定
    assert reg.execute(user, "which_locale", {}, "en").content["data"]["locale"] == "en"
    assert current_locale() == "ja"  # 呼び出しが終われば元に戻る(他の呼び出しに漏れない)


def test_locale_survives_the_mcp_round_trip() -> None:
    from health_insight_agent.mcp_server.client import McpToolExecutor
    from health_insight_agent.mcp_server.server import build_mcp_server

    user = UUID(int=1)
    ex = McpToolExecutor(build_mcp_server(_locale_tool(), user), user)
    try:
        assert ex.execute(user, "which_locale", {}, "en").content["data"]["locale"] == "en"
        assert ex.execute(user, "which_locale", {}, "ja").content["data"]["locale"] == "ja"
    finally:
        ex.close()


def test_health_score_findings_are_written_in_the_requested_language() -> None:
    sleep = SleepFeatures(
        average_duration_minutes=300.0, insufficient_sleep_days=2, duration_stddev_minutes=10.0
    )
    results = {
        "sleep": (analyze_sleep(sleep, "ja"), analyze_sleep(sleep, "en")),
        "weight": (
            analyze_weight(
                WeightFeatures(moving_average_kg=60, change_kg=2, change_rate=0.03), "ja"
            ),
            analyze_weight(
                WeightFeatures(moving_average_kg=60, change_kg=2, change_rate=0.03), "en"
            ),
        ),
        "workout": (
            analyze_workout(
                WorkoutFeatures(
                    session_count=1, total_cardio_minutes=20, total_strength_volume_kg=900.0
                ),
                "ja",
            ),
            analyze_workout(
                WorkoutFeatures(
                    session_count=1, total_cardio_minutes=20, total_strength_volume_kg=900.0
                ),
                "en",
            ),
        ),
        "nutrition": (
            analyze_nutrition(
                NutritionFeatures(
                    total_calories=2000, protein_ratio=0.1, fat_ratio=0.35, carbohydrate_ratio=0.55
                ),
                "ja",
            ),
            analyze_nutrition(
                NutritionFeatures(
                    total_calories=2000, protein_ratio=0.1, fat_ratio=0.35, carbohydrate_ratio=0.55
                ),
                "en",
            ),
        ),
        "no-data": (
            analyze_sleep(
                SleepFeatures(
                    average_duration_minutes=None,
                    insufficient_sleep_days=0,
                    duration_stddev_minutes=None,
                ),
                "ja",
            ),
            analyze_sleep(
                SleepFeatures(
                    average_duration_minutes=None,
                    insufficient_sleep_days=0,
                    duration_stddev_minutes=None,
                ),
                "en",
            ),
        ),
    }
    for name, (ja, en) in results.items():
        assert ja.score == en.score, name  # 言語で判定は変わらない
        assert all(JAPANESE.search(f) for f in ja.findings), name
        assert not any(JAPANESE.search(f) for f in en.findings), (name, en.findings)
    assert "5.0 h" in results["sleep"][1].findings[0]  # 睡眠は「分」ではなく時間で述べる


def test_model_adoption_reasons_exist_in_english() -> None:
    _, q = F.build_samples([])
    assert q.reason == "データが0日分です" and q.reason_en == "No data (0 days)"
    from tests.unit.test_ml import _series

    _, q40 = F.build_samples(_series(40))
    assert "データ量が不足" in q40.reason and "Not enough data" in q40.reason_en
    assert not JAPANESE.search(q40.reason_en)


def test_english_units_contain_no_japanese() -> None:
    from health_insight_agent.services.metrics import (
        METRICS,
        mean_display,
        metric_display,
        unit_text,
    )

    for name in METRICS:
        assert not JAPANESE.search(metric_display(name, "en")[1]), name
        assert not JAPANESE.search(mean_display(name, 1.0, "en")[1]), name
    assert unit_text("時", "en") == "h" and unit_text("分", "en") == "min"
    assert unit_text("時", "ja") == "時" and unit_text("kcal", "en") == "kcal"
