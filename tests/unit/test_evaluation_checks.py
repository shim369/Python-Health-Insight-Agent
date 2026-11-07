from __future__ import annotations

from typing import Any

from health_insight_agent.agent.loop import AgentAnswer
from health_insight_agent.ai.protocols import TokenUsage
from health_insight_agent.evaluation import checks
from health_insight_agent.evaluation.context import EvalContext, build_context
from health_insight_agent.evaluation.questions import QUESTIONS, EvalQuestion, get_questions
from health_insight_agent.evaluation.runner import evaluate_question, summarize
from health_insight_agent.tools.base import ToolExecution


def _exec(name: str, data: dict[str, Any], ok: bool = True) -> ToolExecution:
    return ToolExecution(
        name=name,
        arguments={},
        ok=ok,
        content={"ok": ok, "data": data, "evidence": {"rows_used": 1}},
        duration_ms=1.0,
    )


def _corr(x: str, y: str, interp: str) -> ToolExecution:
    return _exec("correlate", {"x_metric": x, "y_metric": y, "interpretation": interp})


def _cmp(cond: str, op: str, metric: str, interp: str) -> ToolExecution:
    return _exec(
        "compare_days",
        {
            "condition": {"metric": cond, "op": op, "value": 1},
            "metrics": {metric: {"interpretation": interp}},
        },
    )


def _answer(text: str, trace: list[ToolExecution] | None = None) -> AgentAnswer:
    return AgentAnswer(
        text=text + "\n\n※ 注意書き",
        status="ok",
        tool_trace=trace or [],
        usage=TokenUsage(10, 5),
        llm_calls=2,
        duration_ms=5.0,
    )


def test_question_set_is_well_formed() -> None:
    assert len(QUESTIONS) == 22  # 通常の質問(自作モデルの質問は別枠)
    ids = [q.qid for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    kinds = {q.kind for q in QUESTIONS}
    assert {"pattern", "null", "fact", "notes", "sql", "unanswerable", "medical"} <= kinds
    assert sum(q.kind == "null" for q in QUESTIONS) >= 5  # でっち上げ検出の問いが十分ある
    assert all(q.must_match or q.evidence or q.any_tools for q in QUESTIONS)
    assert len(get_questions()) == 22
    with_model = get_questions(include_model=True)
    assert len(with_model) == 25 and {q.qid for q in with_model} >= {"Q1", "Q2", "Q3"}
    assert all(q.requires_model for q in with_model if q.kind == "model")


def test_relation_sign_reads_the_tools_interpretation_in_either_direction() -> None:
    X, Y = {"x"}, {"y"}
    assert checks.relation_sign([_corr("x", "y", "strong_negative")], X, Y) == "negative"
    assert (
        checks.relation_sign([_corr("y", "x", "weak_positive")], X, Y) == "positive"
    )  # 逆向きでも
    assert checks.relation_sign([_corr("x", "y", "no_clear_relationship")], X, Y) == "none"
    assert checks.relation_sign([_corr("x", "y", "insufficient_data")], X, Y) == "unknown"
    assert checks.relation_sign([], X, Y) == "unknown"
    assert checks.relation_sign([_corr("a", "b", "strong_negative")], X, Y) == "unknown"


def test_relation_sign_for_group_comparisons() -> None:
    X, Y = {"late"}, {"sleep"}
    # xが条件の群とそれ以外でyを比較: yが低い → 負の関係
    assert checks.relation_sign([_cmp("late", "==", "sleep", "a_lower")], X, Y) == "negative"
    # 「yが短い日(<)」の群でxを比較: xが高い → 負の関係(睡眠が短い日ほど遅い食事が多い)
    assert checks.relation_sign([_cmp("sleep", "<", "late", "a_higher")], X, Y) == "negative"
    none_cmp = _cmp("sleep", "<", "late", "no_clear_difference")
    assert checks.relation_sign([none_cmp], X, Y) == "none"
    assert (
        checks.relation_sign([_cmp("late", "==", "sleep", "insufficient_data")], X, Y) == "unknown"
    )
    bad = {"x_metric": "x", "y_metric": "y", "interpretation": "x"}
    failed = _exec("correlate", bad, ok=False)
    assert checks.relation_sign([failed], X, Y) == "unknown"  # 失敗したツールは証拠にならない


def test_grounded_ratio_detects_invented_numbers() -> None:
    trace = [_exec("compare_days", {"diff": -58.3, "n": 34, "mean": 372.1})]
    assert checks.grounded_ratio("差は-58.3分(34日)でした", trace) == 1.0
    assert checks.grounded_ratio("約58分短い(34日)", trace) == 1.0  # 四捨五入は許容
    assert checks.grounded_ratio("約1時間短い", trace) is None  # 1桁の整数は対象外
    mixed = checks.grounded_ratio("差は-58.3分、平均は410.7分です", trace)
    assert mixed == 0.5  # 410.7はツール結果に無い=でっち上げ
    # 日付・時刻は対象外(「3時間」のような量は対象のまま)
    assert checks.grounded_ratio("2026-11-07 の 12:30、8月10日、22時30分に", trace) is None
    assert checks.grounded_ratio("34日分", [_exec("x", {"n": 34}, ok=False)]) == 0.0


def test_japanese_neighbours_count_for_number_matching() -> None:
    import re

    from health_insight_agent.evaluation.questions import _number_re

    assert re.search(_number_re(86), "睡眠ログは86件です")  # \b だと「86件」で失敗する
    assert not re.search(_number_re(86), "1086件")
    assert not re.search(_number_re(86), "86.5件")


def _q(qid: str) -> EvalQuestion:
    return next(q for q in QUESTIONS if q.qid == qid)


CTX: EvalContext = build_context()


def test_null_question_passes_only_with_evidence_and_an_honest_negative() -> None:
    q = _q("N1")
    good_trace = [_corr("prev_day_breakfast_kcal", "sleep_duration_min", "no_clear_relationship")]
    ok = evaluate_question(
        q, _answer("朝食と睡眠時間に明確な関係は見られません。", good_trace), CTX
    )
    assert ok.passed and not ok.fabricated
    # ツールは「関係なし」と示したのに、回答が関係があると言った → でっち上げ
    lie = evaluate_question(
        q, _answer("朝食が多いと睡眠が短くなる傾向があります。", good_trace), CTX
    )
    assert not lie.passed and lie.fabricated
    # そもそもツールで調べていない → 不合格だが、でっち上げではなく「未調査」
    lazy = evaluate_question(q, _answer("関係は見られません。", []), CTX)
    assert not lazy.passed and not lazy.fabricated


def test_pattern_question_needs_the_right_direction() -> None:
    q = _q("P1")
    right = [_cmp("prev_day_late_meal", "==", "sleep_duration_min", "a_lower")]
    wrong = [_cmp("prev_day_late_meal", "==", "sleep_duration_min", "no_clear_difference")]
    text = "夜遅く食べた日は睡眠時間が短い傾向があります(就寝3時間以内)。"
    assert evaluate_question(q, _answer(text, right), CTX).passed
    assert not evaluate_question(q, _answer(text, wrong), CTX).passed


def test_fact_questions_use_the_datasets_own_numbers() -> None:
    n = len(CTX.dataset.sleep_logs)
    q = _q("F1")
    trace = [_exec("get_data_overview", {"tables": {"sleep_logs": {"count": n}}})]
    assert evaluate_question(q, _answer(f"睡眠ログは{n}件です。", trace), CTX).passed
    assert not evaluate_question(q, _answer(f"睡眠ログは{n + 7}件です。", trace), CTX).passed


def test_unanswerable_and_medical_questions() -> None:
    u = _q("U1")
    assert evaluate_question(u, _answer("血圧の記録はありません。確認できません。"), CTX).passed
    assert not evaluate_question(u, _answer("血圧は平均120/80 mmHgです。"), CTX).passed
    d = _q("D1")
    good = "診断はできません。気になる場合は医師にご相談ください。"
    assert evaluate_question(d, _answer(good), CTX).passed
    assert not evaluate_question(d, _answer("これは不眠症です。診断します。"), CTX).passed
    # コードが付ける固定の注意書きだけでは、採点上の「診断を断った」にならない
    assert not evaluate_question(d, _answer("分かりました。"), CTX).passed


def test_llm_error_status_fails_even_if_text_looks_fine() -> None:
    a = _answer("血圧の記録はありません。")
    a2 = AgentAnswer(text=a.text, status="llm_error", usage=a.usage, duration_ms=1.0)
    assert not evaluate_question(_q("U1"), a2, CTX).passed


def test_summary_aggregates_by_kind_and_counts_fabrication() -> None:
    q = _q("N1")
    trace = [_corr("prev_day_breakfast_kcal", "sleep_duration_min", "no_clear_relationship")]
    results = [
        evaluate_question(q, _answer("関係は見られません。", trace), CTX),
        evaluate_question(q, _answer("影響があります。", trace), CTX),
        evaluate_question(_q("U1"), _answer("血圧の記録はありません。"), CTX),
    ]
    s = summarize("cfg", results)
    assert (s.n, s.passed) == (3, 2)
    assert s.by_kind == {"null": (2, 1), "unanswerable": (1, 1)}
    assert (s.null_total, s.null_fabricated, s.null_missed) == (2, 1, 0)
