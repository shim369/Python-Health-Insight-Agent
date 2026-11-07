"""モデルの学習・評価・採用判断を実行して報告する(パイプラインの動作確認)。

    uv run python -m health_insight_agent.ml [--user-id UUID] [--today YYYY-MM-DD]

合成データで学習したモデルの精度は「パイプラインの動作実証」であり、予測精度の成果ではない。
結果は ml_artifacts/(採用判断JSON)と mlruns/(MLflow)に保存される(どちらもgit管理外)。
"""

from __future__ import annotations

import argparse
from datetime import date
from uuid import UUID

from health_insight_agent.config import PROJECT_ROOT, load_settings
from health_insight_agent.ml.evaluation import MLEvaluationResult
from health_insight_agent.ml.model_registry import ModelRegistry
from health_insight_agent.ml.service import SleepModelService
from health_insight_agent.ml.tracking import MlflowTracker, NullTracker
from health_insight_agent.repositories.sqlalchemy.factory import build_uow_factory
from health_insight_agent.services.insight_data_service import InsightDataService


def _print_eval(ev: MLEvaluationResult) -> None:
    print(f"  samples={ev.n_samples}  baseline={ev.baseline_name}")
    print(f"  MAE: model(val)={ev.model_val_mae:.1f}分  model(train)={ev.model_train_mae:.1f}分")
    print(
        f"       recent-7-mean={ev.recent_mean_mae:.1f}分  train-mean={ev.train_mean_mae:.1f}分"
        f"  -> 比較に使うBaseline={ev.baseline_mae:.1f}分"
    )
    print(f"  Baseline比の改善率={ev.improvement_rate:.1%}(基準10%以上)")
    print(f"  train/val乖離={ev.overfit_gap:.1%}(基準20%以内)")
    for i, f in enumerate(ev.folds, 1):
        print(
            f"    fold{i}: val={f.model_val_mae:.1f} train={f.model_train_mae:.1f} "
            f"recent7={f.recent_mean_mae:.1f} trainmean={f.train_mean_mae:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sleep model pipeline")
    parser.add_argument("--user-id", type=UUID, default=None)
    parser.add_argument("--today", type=date.fromisoformat, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    user_id = args.user_id or settings.demo_user_id
    url = settings.readonly_database_url or settings.database_url
    factory = build_uow_factory(url, read_only=True)
    today = args.today
    data = (
        InsightDataService(factory, clock=(lambda: today)) if today else InsightDataService(factory)
    )
    tracker = NullTracker() if args.no_mlflow else MlflowTracker(PROJECT_ROOT / "mlruns")
    svc = SleepModelService(data, ModelRegistry(PROJECT_ROOT / "ml_artifacts"), tracker=tracker)
    state = svc._state(user_id)  # CLI専用: 評価の詳細を表示するため内部状態を読む
    q, d = state.quality, state.decision
    print(
        f"データ: {q.total_days}日 / 使えるサンプル{q.usable_samples} / 欠損率{q.missing_rate:.1%}"
    )
    print(f"  -> {q.reason}")
    if state.evaluation:
        _print_eval(state.evaluation)
    print(f"採用判断: {'採用' if d.adopted else '見送り(モデルは使わない)'} - {d.reason}")
    if d.feature_importance:
        top = ", ".join(f"{k}={v:.2f}" for k, v in list(d.feature_importance.items())[:5])
        print(f"重要度(上位): {top}")
    print("※ 合成データでの結果は、パイプラインの動作実証であり予測精度の成果ではありません。")


if __name__ == "__main__":
    main()
