"""評価パイプライン全体(固定データ→質問セット→採点→集計)を、キー無しのFakeで通しで検証する。"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import delete

from health_insight_agent.config import load_settings
from health_insight_agent.database import schema as orm
from health_insight_agent.evaluation.configs import PRESETS, build_eval_agent
from health_insight_agent.evaluation.context import (
    EVAL_USER_ID,
    EvalContext,
    build_context,
    seed_eval_data,
)
from health_insight_agent.evaluation.questions import get_questions
from health_insight_agent.evaluation.runner import (
    QuestionResult,
    render_markdown,
    run_questions,
    summarize,
)
from health_insight_agent.repositories.sqlalchemy.factory import build_uow_factory

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ctx(migrated_database: str) -> Iterator[EvalContext]:
    context = seed_eval_data(migrated_database, build_context())
    yield context
    with build_uow_factory(migrated_database)() as uow:
        uow.purge_user_data(EVAL_USER_ID)
        uow._session.execute(delete(orm.UserORM).where(orm.UserORM.id == EVAL_USER_ID))  # type: ignore[union-attr]
        uow.commit()


def _run(name: str, ctx: EvalContext) -> list[QuestionResult]:
    agent = build_eval_agent(load_settings(), PRESETS[name], today=ctx.today)
    return run_questions(agent, get_questions(include_model=True), ctx)


def test_rule_based_baseline_finds_patterns_and_never_fabricates(ctx: EvalContext) -> None:
    results = _run("fake-all", ctx)
    by_id = {r.qid: r for r in results}
    # 仕込みパターンを見つけ、仕込んでいない関係には「ない」と答えている
    for qid in ("P1", "P2", "P3", "P4", "P5", "N1", "N2", "N3", "N4", "N5", "N6"):
        assert by_id[qid].passed, (qid, by_id[qid].failures)
    s = summarize("fake-all", results)
    assert s.null_fabricated == 0 and s.null_missed == 0
    assert s.grounded_avg == 1.0  # Fakeはツール結果をそのまま写す=数値は全て根拠あり
    assert s.rate >= 0.85
    # LLM抜きのルールベースが苦手なもの(量の言語化・データに無いこと・診断を断る)は落ちる=
    # 採点が「それっぽい文」を甘く通していないことの確認
    assert not by_id["U1"].passed and not by_id["D1"].passed and not by_id["P6"].passed


def test_removing_rag_and_sql_tools_lowers_coverage(ctx: EvalContext) -> None:
    full = {r.qid: r for r in _run("fake-all", ctx)}
    typed = {r.qid: r for r in _run("fake-typed-only", ctx)}
    for qid in ("M2", "S1", "N3"):  # メモ検索・SQLが要る問い
        assert full[qid].passed and not typed[qid].passed, qid
    assert sum(r.passed for r in typed.values()) < sum(r.passed for r in full.values())


def test_markdown_report_lists_every_config_and_question(ctx: EvalContext) -> None:
    results = {n: _run(n, ctx) for n in ("fake-all", "fake-typed-only")}
    md = render_markdown([summarize(n, r) for n, r in results.items()], results)
    assert "`fake-all`" in md and "`fake-typed-only`" in md
    assert all(f"| {q.qid} |" in md for q in get_questions(include_model=True))


def test_custom_model_questions_pass_only_when_the_model_tool_is_available(
    ctx: EvalContext,
) -> None:
    with_model = {r.qid: r for r in _run("fake-all", ctx)}
    without = {r.qid: r for r in _run("fake-no-model", ctx)}
    for qid in ("Q1", "Q2", "Q3"):
        assert with_model[qid].passed, (qid, with_model[qid].failures)
        assert not without[qid].passed
    # 予測値は「参考値」+ モデルのMAE + Baseline比つきで述べられている
    q1 = with_model["Q1"].answer
    assert "参考値" in q1 and "MAE" in q1 and "ベースライン" in q1


def test_numeric_consistency_with_and_without_the_model(ctx: EvalContext) -> None:
    s_with = summarize("fake-all", _run("fake-all", ctx))
    s_without = summarize("fake-no-model", _run("fake-no-model", ctx))
    # モデルあり: 同じ質問を3回繰り返しても予測値が毎回同じ(数値の一貫性)
    assert s_with.model_consistency == (3, 3)
    # モデルなし: 予測値そのものが存在しないので、一貫性は判定対象外
    assert s_without.model_consistency == (0, 0)
    assert s_with.passed > s_without.passed
