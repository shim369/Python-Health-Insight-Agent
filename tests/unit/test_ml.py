"""自作モデル(睡眠予測)の単体テスト: 特徴量・欠損の扱い・交差検証・採用判断・記録。"""

from __future__ import annotations

import random
import statistics
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from health_insight_agent.ml import features as F
from health_insight_agent.ml.cross_validation import forward_chaining_splits
from health_insight_agent.ml.evaluation import (
    IMPROVEMENT_THRESHOLD,
    OVERFIT_GAP_THRESHOLD,
    evaluate,
)
from health_insight_agent.ml.model_registry import ModelDecision, ModelRegistry, user_key
from health_insight_agent.ml.tracking import MlflowTracker, NullTracker
from health_insight_agent.ml.train import (
    OVERFIT_PRONE_PARAMS,
    feature_importance,
    fit_model,
    lightgbm_train_predict,
)
from health_insight_agent.services.metrics import DayRow, build_day_rows
from health_insight_agent.synthetic.generator import generate_dataset
from tests.conftest import FIXED_END_DATE

U = UUID("00000000-0000-4000-8000-000000000001")


def _rows(*, drop_workouts: bool = False, days: int = 90) -> list[DayRow]:
    ds = generate_dataset(U, end_date=FIXED_END_DATE, days=days, seed=42)
    start = FIXED_END_DATE - timedelta(days=days - 1)
    return build_day_rows(
        start=start + timedelta(days=1),
        end=FIXED_END_DATE,
        sleeps=ds.sleep_logs,
        meals=ds.meal_logs,
        workouts=[] if drop_workouts else ds.workout_logs,
        health_logs=ds.health_logs,
        weights=ds.weight_logs,
    )


def _full_row(d: date, **kw: float | None) -> DayRow:
    base = dict(
        date=d,
        sleep_duration_min=420.0,
        sleep_bedtime_hour=23.5,
        prev_day_total_kcal=1900.0,
        prev_day_last_meal_hour=19.5,
        prev_day_meal_to_bed_gap_h=4.0,
        prev_day_late_meal=0.0,
        prev_day_workout_kcal=0.0,
        prev_day_workout_intensity=0.0,
        prev_day_evening_high_workout=0.0,
        is_weekend=0.0,
    )
    base.update(kw)
    return DayRow(**base)  # type: ignore[arg-type]


def _series(n: int, **overrides: float | None) -> list[DayRow]:
    start = date(2026, 1, 1)
    return [_full_row(start + timedelta(days=i), **overrides) for i in range(n)]


# ---------------- 特徴量・欠損・データ品質 ----------------
def test_synthetic_data_meets_the_quality_requirements() -> None:
    samples, q = F.build_samples(_rows())
    assert q.is_sufficient and q.total_days >= F.MIN_DAYS
    assert q.missing_rate < F.MAX_MISSING_RATE and len(samples) >= 70
    s = samples[0]
    assert len(s.features) == len(F.FEATURE_NAMES) and s.target > 0


def test_days_without_workouts_are_not_treated_as_missing() -> None:
    """既存アプリでは「運動時間None=運動なし」を欠損扱いして常にフォールバックになった不具合があった。"""
    samples, q = F.build_samples(_series(80, prev_day_workout_kcal=0.0))
    assert q.is_sufficient and q.missing_rate < 0.05  # 先頭の1日(前の晩が無い)だけが欠損
    idx = F.FEATURE_NAMES.index("workout_kcal")
    assert all(s.features[idx] == 0.0 for s in samples)
    # 実データの経路でも: 90日間1度も運動しなくても、学習に進める
    no_workouts, q2 = F.build_samples(_rows(drop_workouts=True))
    assert q2.is_sufficient and len(no_workouts) >= 70


def test_missing_values_are_counted_and_gate_the_model() -> None:
    rows = _series(80)
    # 40%の日で睡眠記録が無い → 欠損率が基準を超える
    broken = [
        replace(r, sleep_duration_min=None, sleep_bedtime_hour=None) if i % 5 in (1, 3) else r
        for i, r in enumerate(rows)
    ]
    _, q = F.build_samples(broken)
    assert not q.is_sufficient and "欠損率" in q.reason and q.missing_rate >= F.MAX_MISSING_RATE
    # 前日の食事が記録されていない日は欠損
    no_meals = [
        replace(r, prev_day_last_meal_hour=None) if i % 2 else r for i, r in enumerate(rows)
    ]
    assert not F.build_samples(no_meals)[1].is_sufficient


def test_too_little_data_is_rejected_before_any_training() -> None:
    _, q = F.build_samples(_series(40))
    assert not q.is_sufficient and "データ量が不足" in q.reason and q.total_days == 40
    assert "0日" in F.build_samples([])[1].reason


def test_baseline_uses_only_past_nights() -> None:
    rows = _series(20)
    rows = [replace(r, sleep_duration_min=300.0 + i) for i, r in enumerate(rows)]
    samples = F.build_samples(rows)[0]
    s = next(x for x in samples if x.day == rows[10].date)
    expected = statistics.fmean(r.sleep_duration_min for r in rows[3:10])  # type: ignore[misc]
    assert s.baseline == pytest.approx(expected)  # 当日(i=10)は含まない=未来を見ない
    assert s.target == 310.0


# ---------------- 交差検証 ----------------
def test_forward_chaining_never_trains_on_the_future() -> None:
    splits = forward_chaining_splits(60, n_splits=5)
    assert len(splits) == 5
    for train, val in splits:
        assert train.start == 0 and train.stop == val.start  # 学習は常に検証より前
        assert len(val) > 0
    assert [t.stop for t, _ in splits] == sorted(t.stop for t, _ in splits)
    with pytest.raises(ValueError):
        forward_chaining_splits(5, n_splits=5)
    with pytest.raises(ValueError):
        forward_chaining_splits(60, n_splits=1)


# ---------------- 採用判断 ----------------
def _samples() -> list[F.SleepSample]:
    return F.build_samples(_rows())[0]


def test_regularized_lightgbm_is_adopted_on_the_planted_patterns() -> None:
    ev = evaluate(_samples(), lightgbm_train_predict)
    assert ev.adopted and not ev.is_overfitting
    assert ev.improvement_rate >= IMPROVEMENT_THRESHOLD  # Baseline比で相対10%以上
    assert ev.overfit_gap <= OVERFIT_GAP_THRESHOLD  # train/valの乖離が相対20%以内
    assert ev.baseline_mae == min(ev.recent_mean_mae, ev.train_mean_mae)  # 良い方のBaselineと比較
    assert len(ev.folds) == 5


def test_overfit_prone_model_is_rejected_even_though_it_beats_the_baseline() -> None:
    def overfit(x: list[list[float]], y: list[float], xp: list[list[float]]) -> list[float]:
        return lightgbm_train_predict(x, y, xp, OVERFIT_PRONE_PARAMS)

    ev = evaluate(_samples(), overfit)
    assert ev.improvement_rate >= IMPROVEMENT_THRESHOLD  # 精度は良く見えるが…
    assert ev.is_overfitting and not ev.adopted  # 過学習チェックで不採用
    assert "過学習" in ev.reason


def test_model_that_is_no_better_than_the_baseline_is_rejected() -> None:
    def train_mean(x: list[list[float]], y: list[float], xp: list[list[float]]) -> list[float]:
        return [statistics.fmean(y)] * len(xp)

    ev = evaluate(_samples(), train_mean)
    assert not ev.adopted and "改善率" in ev.reason and abs(ev.improvement_rate) < 0.01


def test_memorizing_model_trips_the_overfit_check() -> None:
    def memorize(x: list[list[float]], y: list[float], xp: list[list[float]]) -> list[float]:
        table = {tuple(r): v for r, v in zip(x, y, strict=True)}
        mean = statistics.fmean(y)
        return [table.get(tuple(r), mean) for r in xp]

    ev = evaluate(_samples(), memorize)
    assert ev.model_train_mae == 0 and ev.is_overfitting and not ev.adopted


def test_noise_only_target_is_not_adopted() -> None:
    """予測できる信号が無いデータ(目的変数がノイズ)では、採用されない。"""
    rng = random.Random(0)
    noisy = [replace(s, target=420 + rng.gauss(0, 25)) for s in _samples()]
    assert not evaluate(noisy, lightgbm_train_predict).adopted


def test_feature_importance_is_a_distribution_led_by_meal_timing() -> None:
    model = fit_model(_samples())
    imp = feature_importance(model)
    assert sum(imp.values()) == pytest.approx(1.0) and list(imp) != sorted(imp)  # 大きい順の辞書
    assert next(iter(imp)) in F.MEAL_TIMING_FEATURES  # 仕込んだ「食事から就寝まで」を使っている
    assert set(imp) == set(F.FEATURE_NAMES)


def test_training_is_deterministic() -> None:
    a = lightgbm_train_predict(
        [list(s.features) for s in _samples()],
        [s.target for s in _samples()],
        [list(_samples()[0].features)],
    )
    b = lightgbm_train_predict(
        [list(s.features) for s in _samples()],
        [s.target for s in _samples()],
        [list(_samples()[0].features)],
    )
    assert a == b  # 乱数・スレッドを固定しているので再現できる


# ---------------- 採用判断の保存・記録 ----------------
def _decision(adopted: bool = True) -> ModelDecision:
    return ModelDecision(
        domain="sleep",
        adopted=adopted,
        reason="r",
        total_days=89,
        usable_samples=80,
        missing_rate=0.08,
        data_signature="sig",
        trained_at="2026-11-08T00:00:00+00:00",
        mae_minutes=21.5,
        baseline_name="training_mean",
        baseline_mae_minutes=31.5,
        improvement_rate=0.316,
        overfit_gap=-0.16,
        feature_importance={"meal_to_bed_gap_h": 0.7, "last_meal_hour": 0.3},
    )


def test_registry_roundtrip_is_per_user_and_hides_the_raw_user_id(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "ml")
    path = reg.save(U, _decision())
    assert reg.load(U) == _decision()
    assert reg.load(UUID(int=7)) is None  # 別ユーザーの判断は混ざらない
    assert str(U) not in path.name and user_key(U) in path.name


def test_mlflow_records_conditions_metrics_and_the_decision(tmp_path: Path) -> None:
    import mlflow

    MlflowTracker(tmp_path / "mlruns").log_decision(
        user_key(U), _decision(), {"model": "lightgbm", "cv": "forward_chaining_5"}
    )
    uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').resolve()}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("sleep-duration-model")
    assert exp is not None
    [run] = client.search_runs([exp.experiment_id])
    assert run.data.params["model"] == "lightgbm" and run.data.tags["adopted"] == "True"
    assert run.data.metrics["mae_minutes"] == pytest.approx(21.5)
    assert run.data.metrics["improvement_rate"] == pytest.approx(0.316)
    assert str(U) not in str(run.data.tags)  # ユーザーはハッシュだけ


def test_tracking_failures_never_break_the_answer_path() -> None:
    MlflowTracker(Path("/dev/null/cannot-create")).log_decision(user_key(U), _decision(), {})
    NullTracker().log_decision(user_key(U), _decision(), {})
