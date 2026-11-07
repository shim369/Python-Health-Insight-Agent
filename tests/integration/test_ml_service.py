"""睡眠モデルのService/ツールを実DB(合成データ)で検証。採用・フォールバック・分離・キャッシュ。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from health_insight_agent.ml.model_registry import ModelRegistry
from health_insight_agent.ml.service import PredictionConditions, SleepModelService
from health_insight_agent.ml.train import lightgbm_train_predict
from health_insight_agent.services.health_insight_service import LocalHealthInsightService
from health_insight_agent.services.insight_data_service import (
    InsightDataService,
    InvalidRequestError,
)
from health_insight_agent.tools.base import ToolRegistry
from health_insight_agent.tools.builtin import build_default_registry
from tests.conftest import FIXED_END_DATE

pytestmark = pytest.mark.integration

LATE = PredictionConditions(last_meal_hour=22.5, bedtime_hour=24.0)  # 就寝1.5時間前に食事
EARLY = PredictionConditions(last_meal_hour=19.0, bedtime_hour=23.0)  # 就寝4時間前に食事


class _Spy:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: list[list[float]], y: list[float], xp: list[list[float]]) -> list[float]:
        self.calls += 1
        return lightgbm_train_predict(x, y, xp)


@pytest.fixture
def env(
    seeded: Any, tmp_path: Path
) -> Iterator[tuple[SleepModelService, ModelRegistry, UUID, UUID, _Spy]]:
    factory, uid, other = seeded
    spy = _Spy()
    registry = ModelRegistry(tmp_path / "ml")
    data = InsightDataService(factory, clock=lambda: FIXED_END_DATE)
    yield SleepModelService(data, registry, train_predict=spy), registry, uid, other, spy


def test_model_is_adopted_and_predictions_follow_the_planted_pattern(env: Any) -> None:
    svc, registry, uid, _, _ = env
    d = svc.decision(uid)
    assert d.adopted and d.improvement_rate is not None and d.improvement_rate >= 0.10
    assert d.mae_minutes is not None and d.baseline_mae_minutes is not None
    assert d.mae_minutes < d.baseline_mae_minutes
    late, early = svc.predict(uid, LATE), svc.predict(uid, EARLY)
    assert late.model_used and early.model_used
    assert late.predicted_minutes is not None and early.predicted_minutes is not None
    assert early.predicted_minutes - late.predicted_minutes >= 25  # 遅い食事は睡眠が短い予測
    assert 300 <= late.predicted_minutes <= 420 and 380 <= early.predicted_minutes <= 480
    assert registry.load(uid) == d  # 採用判断がJSONに保存されている
    assert "total_kcal" in late.defaulted and late.fallback is None


def test_prediction_is_consistent_for_the_same_input(env: Any) -> None:
    svc, _, uid, _, _ = env
    first = svc.predict(uid, LATE).predicted_minutes
    assert all(svc.predict(uid, LATE).predicted_minutes == first for _ in range(3))


def test_state_is_cached_until_data_changes(env: Any) -> None:
    svc, _, uid, _, spy = env
    svc.predict(uid, LATE)
    trained = spy.calls
    assert trained > 0
    svc.predict(uid, EARLY)
    svc.factors(uid)
    assert spy.calls == trained  # 再評価・再学習はしない


def test_user_with_too_little_data_falls_back_to_statistics(env: Any) -> None:
    svc, registry, _, other, spy = env  # otherは20日分だけ
    d = svc.decision(other)
    assert not d.adopted and "データ量が不足" in d.reason and spy.calls == 0  # 学習に進まない
    res = svc.predict(other, LATE)
    assert not res.model_used and res.predicted_minutes is None
    assert res.fallback is not None and res.fallback.recent_mean_minutes is not None
    assert registry.load(other) == d  # 見送りも記録される
    f = svc.factors(other)
    assert not f.model_used and f.factors == []


def test_unknown_user_has_no_model_and_no_crash(env: Any) -> None:
    svc, *_ = env
    res = svc.predict(UUID(int=4242), LATE)
    assert not res.model_used and "0日" in res.decision.reason


def test_training_errors_degrade_to_no_model(seeded: Any, tmp_path: Path) -> None:
    factory, uid, _ = seeded

    def boom(x: Any, y: Any, xp: Any) -> Any:
        raise RuntimeError("lightgbm exploded")

    svc = SleepModelService(
        InsightDataService(factory, clock=lambda: FIXED_END_DATE),
        ModelRegistry(tmp_path),
        train_predict=boom,
    )
    res = svc.predict(uid, LATE)
    assert not res.model_used and "エラー" in res.decision.reason  # 黙って壊れず、使わないと明示


def test_invalid_conditions_are_rejected(env: Any) -> None:
    svc, _, uid, _, _ = env
    with pytest.raises(InvalidRequestError):
        svc.predict(uid, PredictionConditions(last_meal_hour=23.0, bedtime_hour=22.0))


def test_users_models_are_independent(env: Any) -> None:
    svc, _, uid, other, _ = env
    assert svc.decision(uid).adopted and not svc.decision(other).adopted
    assert svc.decision(uid).data_signature != svc.decision(other).data_signature


# ---------------- ツール経由 ----------------
@pytest.fixture
def tools(env: Any, seeded: Any) -> tuple[ToolRegistry, UUID, UUID]:
    svc, _, uid, other, _ = env
    factory = seeded[0]
    reg = build_default_registry(
        InsightDataService(factory, clock=lambda: FIXED_END_DATE),
        LocalHealthInsightService(factory),
        model=svc,
    )
    return reg, uid, other


def test_predict_tool_marks_the_value_as_a_reference_with_model_quality(tools: Any) -> None:
    reg, uid, _ = tools
    ex = reg.execute(uid, "predict_sleep_duration", {"last_meal_hour": 22.5, "bedtime_hour": 24})
    assert ex.ok
    d = ex.content["data"]
    assert d["model_used"] and d["reference_only"] is True
    assert d["model"]["adopted"] and d["model"]["mae_minutes"] > 0
    assert d["model"]["improvement_vs_baseline"] >= 0.10 and d["model"]["baseline"]
    assert d["predicted_sleep_hours"] == pytest.approx(d["predicted_sleep_minutes"] / 60, abs=0.01)
    assert any("reference value" in n for n in ex.content["evidence"]["notes"])
    card = ex.charts[0]
    assert card["kind"] == "value" and card["label"] == "モデルによる参考値"
    assert "MAE" in card["note"]  # UIにも誤差を併記する


def test_predict_tool_without_a_model_says_so_and_returns_statistics(tools: Any) -> None:
    reg, _, other = tools
    ex = reg.execute(other, "predict_sleep_duration", {"last_meal_hour": 22.5, "bedtime_hour": 24})
    d = ex.content["data"]
    assert ex.ok and d["model_used"] is False and d["predicted_sleep_minutes"] is None
    assert d["model"]["adopted"] is False and "不足" in d["model"]["reason"]
    assert d["fallback"]["source"].startswith("statistics") and d["fallback"]["recent_mean_minutes"]
    assert any("NO MODEL WAS USED" in n for n in ex.content["evidence"]["notes"])
    assert ex.charts == []  # 予測値のカードは出さない


def test_predict_tool_validates_arguments(tools: Any) -> None:
    reg, uid, _ = tools
    bad_order = reg.execute(
        uid, "predict_sleep_duration", {"last_meal_hour": 23, "bedtime_hour": 22}
    )
    assert not bad_order.ok and "later" in bad_order.content["error"]
    assert not reg.execute(
        uid, "predict_sleep_duration", {"last_meal_hour": 99, "bedtime_hour": 100}
    ).ok
    assert not reg.execute(
        uid, "predict_sleep_duration", {"last_meal_hour": 20, "bedtime_hour": 23, "user_id": "x"}
    ).ok  # user_idは受け取らない


def test_explain_tool_returns_importances_and_the_fallback_correlations(tools: Any) -> None:
    reg, uid, other = tools
    ex = reg.execute(uid, "explain_sleep_factors", {})
    d = ex.content["data"]
    assert d["model_used"] and d["factors"][0]["feature"] in {"meal_to_bed_gap_h", "last_meal_hour"}
    assert sum(f["importance"] for f in d["factors"]) == pytest.approx(1.0, abs=0.01)
    assert any("NOT a causal" in n for n in ex.content["evidence"]["notes"])
    assert ex.charts[0]["kind"] == "bar"
    fb = reg.execute(other, "explain_sleep_factors", {}).content["data"]
    assert fb["model_used"] is False and fb["factors"] == []
    assert fb["fallback_correlations"]  # モデルの代わりに通常の統計(相関)


def test_model_tools_are_absent_when_not_configured(seeded: Any) -> None:
    factory = seeded[0]
    reg = build_default_registry(InsightDataService(factory), LocalHealthInsightService(factory))
    assert "predict_sleep_duration" not in reg.names and "explain_sleep_factors" not in reg.names
