"""実験管理(MLflow、ローカル保存)。学習条件・指標・採用判断を記録する。

記録の失敗で回答を止めない(記録は付随機能)。user_idは生では記録せず、ハッシュだけを使う。
mlflowは重いので、使うときに初めてimportする。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from health_insight_agent.logging_config import get_logger
from health_insight_agent.ml.model_registry import ModelDecision

logger = get_logger(__name__)
EXPERIMENT = "sleep-duration-model"


class Tracker(Protocol):
    def log_decision(
        self, user_key: str, decision: ModelDecision, params: dict[str, Any]
    ) -> None: ...


class NullTracker:
    def log_decision(self, user_key: str, decision: ModelDecision, params: dict[str, Any]) -> None:
        return None


class MlflowTracker:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def log_decision(self, user_key: str, decision: ModelDecision, params: dict[str, Any]) -> None:
        try:
            import mlflow

            # file ストアはMLflowでメンテナンスモード(例外になる)なので、ローカルSQLiteを使う。
            self._dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{(self._dir / 'mlflow.db').resolve()}")
            if mlflow.get_experiment_by_name(EXPERIMENT) is None:
                mlflow.create_experiment(
                    EXPERIMENT, artifact_location=(self._dir / "artifacts").resolve().as_uri()
                )
            mlflow.set_experiment(EXPERIMENT)
            with mlflow.start_run(
                run_name=f"{user_key}-{'adopted' if decision.adopted else 'rejected'}"
            ):
                mlflow.log_params({k: str(v) for k, v in params.items()})
                metrics = {
                    "total_days": decision.total_days,
                    "usable_samples": decision.usable_samples,
                    "missing_rate": decision.missing_rate,
                    "mae_minutes": decision.mae_minutes,
                    "baseline_mae_minutes": decision.baseline_mae_minutes,
                    "improvement_rate": decision.improvement_rate,
                    "overfit_gap": decision.overfit_gap,
                }
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
                mlflow.set_tags(
                    {"adopted": str(decision.adopted), "domain": decision.domain, "user": user_key}
                )
                mlflow.log_dict(asdict(decision), "decision.json")
        except Exception:
            logger.warning("MLflow logging failed (ignored)", exc_info=True)
