"""採用判断(requirements 9.3.1。既存アプリ ml/evaluation.py と同じ基準)。

- Baseline(直近7夜の平均 / 学習データの平均 のうち**良い方**)に対し、検証MAEが相対10%以上改善
- train誤差とvalidation誤差の差が相対20%以内(過学習チェック。超えたら改善率に関わらず不採用)
どちらかを満たさなければ「モデルを使わない」。学習アルゴリズムには依存しない(関数を注入する)。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from health_insight_agent.ml.cross_validation import forward_chaining_splits
from health_insight_agent.ml.features import SleepSample

IMPROVEMENT_THRESHOLD = 0.10
OVERFIT_GAP_THRESHOLD = 0.20

TrainPredictFn = Callable[[list[list[float]], list[float], list[list[float]]], list[float]]
"""(X_train, y_train, X_predict) -> y_predict。"""


def mae(y: Sequence[float], pred: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y, pred, strict=True)) / len(y)


@dataclass(frozen=True, kw_only=True, slots=True)
class FoldResult:
    recent_mean_mae: float
    train_mean_mae: float
    model_train_mae: float
    model_val_mae: float


@dataclass(frozen=True, kw_only=True, slots=True)
class MLEvaluationResult:
    n_samples: int
    folds: list[FoldResult]
    recent_mean_mae: float
    train_mean_mae: float
    baseline_name: str
    baseline_mae: float
    """比較に使うBaseline(上の2つのうち良い方)。"""
    model_val_mae: float
    model_train_mae: float
    improvement_rate: float
    """(baseline - model) / baseline。正が大きいほどモデルが良い。"""
    overfit_gap: float
    """(val - train) / val。正の値が大きいほど過学習気味。"""
    is_overfitting: bool
    adopted: bool
    reason: str
    reason_en: str = ""


def evaluate(
    samples: Sequence[SleepSample], train_predict: TrainPredictFn, *, n_splits: int = 5
) -> MLEvaluationResult:
    x = [list(s.features) for s in samples]
    y = [s.target for s in samples]
    folds: list[FoldResult] = []
    for train_idx, val_idx in forward_chaining_splits(len(samples), n_splits=n_splits):
        x_tr, y_tr = [x[i] for i in train_idx], [y[i] for i in train_idx]
        x_va, y_va = [x[i] for i in val_idx], [y[i] for i in val_idx]
        train_mean = sum(y_tr) / len(y_tr)
        folds.append(
            FoldResult(
                recent_mean_mae=mae(y_va, [samples[i].baseline for i in val_idx]),
                train_mean_mae=mae(y_va, [train_mean] * len(y_va)),
                model_train_mae=mae(y_tr, train_predict(x_tr, y_tr, x_tr)),
                model_val_mae=mae(y_va, train_predict(x_tr, y_tr, x_va)),
            )
        )

    def avg(f: Callable[[FoldResult], float]) -> float:
        return sum(f(r) for r in folds) / len(folds)

    recent, tmean = avg(lambda r: r.recent_mean_mae), avg(lambda r: r.train_mean_mae)
    baseline_name, baseline = (
        ("recent_7_nights_mean", recent) if recent <= tmean else ("training_mean", tmean)
    )
    val, train = avg(lambda r: r.model_val_mae), avg(lambda r: r.model_train_mae)
    improvement = (baseline - val) / baseline if baseline > 0 else 0.0
    gap = (val - train) / val if val > 0 else 0.0
    overfit = gap > OVERFIT_GAP_THRESHOLD
    if overfit:
        adopted = False
        reason = f"過学習チェックに抵触(乖離{gap:.0%} > {OVERFIT_GAP_THRESHOLD:.0%})のため見送り"
        reason_en = (
            f"Rejected by the overfitting check (gap {gap:.0%} > {OVERFIT_GAP_THRESHOLD:.0%})"
        )
    elif improvement >= IMPROVEMENT_THRESHOLD:
        adopted = True
        reason = (
            f"Baseline({baseline_name})比で{improvement:.1%}改善、乖離{gap:.0%}。採用基準を満たす"
        )
        reason_en = (
            f"{improvement:.1%} better than the baseline ({baseline_name}), gap {gap:.0%}; "
            "meets the adoption criteria"
        )
    else:
        adopted = False
        reason = (
            f"Baseline比の改善率{improvement:.1%}が基準"
            f"({IMPROVEMENT_THRESHOLD:.0%})未満のため見送り"
        )
        reason_en = (
            f"Rejected: improvement over the baseline ({improvement:.1%}) is below the "
            f"{IMPROVEMENT_THRESHOLD:.0%} criterion"
        )
    return MLEvaluationResult(
        n_samples=len(samples),
        folds=folds,
        recent_mean_mae=recent,
        train_mean_mae=tmean,
        baseline_name=baseline_name,
        baseline_mae=baseline,
        model_val_mae=val,
        model_train_mae=train,
        improvement_rate=improvement,
        overfit_gap=gap,
        is_overfitting=overfit,
        adopted=adopted,
        reason=reason,
        reason_en=reason_en,
    )
