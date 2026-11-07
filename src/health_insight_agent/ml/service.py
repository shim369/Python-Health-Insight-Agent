"""睡眠予測モデルのService: データ確認 → 評価 → 採用判断 → 予測/要因の返却。

モデルを使えない(データ不足・欠損過多・基準未達・エラー)ときは**黙ってモデルを使わず**、
`model_used=False` と理由、通常の統計(フォールバック)を返す。呼び出し側(ツール/LLM)は
「モデルは使っていない」ことを回答に明示する。
"""

from __future__ import annotations

import statistics
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

from health_insight_agent.domain.intensity import HIGH_KCAL, MODERATE_KCAL
from health_insight_agent.logging_config import get_logger
from health_insight_agent.ml import features as F
from health_insight_agent.ml.evaluation import MLEvaluationResult, TrainPredictFn, evaluate
from health_insight_agent.ml.model_registry import ModelDecision, ModelRegistry, user_key
from health_insight_agent.ml.predictor import LightGBMSleepPredictor, Predictor
from health_insight_agent.ml.tracking import NullTracker, Tracker
from health_insight_agent.ml.train import (
    REGULARIZED_PARAMS,
    feature_importance,
    fit_model,
    lightgbm_train_predict,
)
from health_insight_agent.services import stats
from health_insight_agent.services.insight_data_service import (
    InsightDataService,
    InvalidRequestError,
)
from health_insight_agent.services.metrics import DayRow

logger = get_logger(__name__)
LOOKBACK_DAYS = 180


@dataclass(frozen=True, slots=True)
class PredictionConditions:
    last_meal_hour: float
    """最後の食事の時刻(24時間制の小数。22.5=22:30)。"""
    bedtime_hour: float
    """就寝時刻(24.5=翌0:30)。last_meal_hourより後であること。"""
    total_kcal: float | None = None
    workout_kcal: float = 0.0
    evening_high_workout: bool = False
    is_weekend: bool = False


@dataclass(frozen=True, slots=True)
class FallbackStats:
    """モデルを使えないときの通常の統計。"""

    recent_mean_minutes: float | None
    recent_n: int
    late_meal_mean_minutes: float | None
    late_meal_n: int
    other_mean_minutes: float | None
    other_n: int


@dataclass(frozen=True, slots=True)
class PredictionResult:
    model_used: bool
    predicted_minutes: float | None
    decision: ModelDecision
    inputs: dict[str, float]
    defaulted: list[str]
    fallback: FallbackStats | None


@dataclass(frozen=True, slots=True)
class FactorsResult:
    model_used: bool
    factors: list[tuple[str, float]]
    """(特徴量名, 重要度)。model_used=Falseのときは空で、代わりに下の統計を使う。"""
    decision: ModelDecision
    fallback_correlations: list[tuple[str, float, int, str]] = field(default_factory=list)
    """(特徴量名, r, n, interpretation)。"""


@dataclass
class _State:
    signature: str
    decision: ModelDecision
    predictor: Predictor | None
    evaluation: MLEvaluationResult | None
    samples: list[F.SleepSample]
    quality: F.DataQualityReport
    rows: list[DayRow]


def _trim(rows: list[DayRow]) -> list[DayRow]:
    """先頭・末尾の「何も記録が無い日」を除く(記録開始前の空白を欠損と数えないため)。"""

    def has_data(r: DayRow) -> bool:
        return r.sleep_duration_min is not None or r.prev_day_total_kcal is not None

    idx = [i for i, r in enumerate(rows) if has_data(r)]
    return rows[idx[0] : idx[-1] + 1] if idx else []


class SleepModelService:
    def __init__(
        self,
        data: InsightDataService,
        registry: ModelRegistry,
        *,
        tracker: Tracker | None = None,
        train_predict: TrainPredictFn = lightgbm_train_predict,
        fit: Callable[[list[F.SleepSample]], object] = fit_model,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._data = data
        self._registry = registry
        self._tracker = tracker or NullTracker()
        self._train_predict = train_predict
        self._fit = fit
        self._now = now
        self._cache: dict[UUID, _State] = {}
        self._lock = threading.Lock()

    # ---- 状態(評価→採用判断)。データが変わらない限りキャッシュ ----
    def _state(self, user_id: UUID) -> _State:
        today = self._data.today()
        rows = _trim(self._data.day_rows(user_id, today - timedelta(days=LOOKBACK_DAYS - 1), today))
        n_sleep = sum(r.sleep_duration_min is not None for r in rows)
        signature = f"{len(rows)}:{rows[-1].date if rows else '-'}:{n_sleep}"
        with self._lock:
            cached = self._cache.get(user_id)
            if cached is not None and cached.signature == signature:
                return cached
            state = self._build_state(user_id, rows, signature)
            self._cache[user_id] = state
            return state

    def _build_state(self, user_id: UUID, rows: list[DayRow], signature: str) -> _State:
        samples, quality = F.build_samples(rows)
        trained_at = self._now().isoformat(timespec="seconds")

        def make(adopted: bool, reason: str, **metrics: object) -> ModelDecision:
            return ModelDecision(
                domain="sleep",
                adopted=adopted,
                reason=reason,
                total_days=quality.total_days,
                usable_samples=quality.usable_samples,
                missing_rate=quality.missing_rate,
                data_signature=signature,
                trained_at=trained_at,
                **metrics,  # type: ignore[arg-type]
            )

        evaluation: MLEvaluationResult | None = None
        predictor: Predictor | None = None
        if not quality.is_sufficient:
            # 評価を待たずに見送り(既存アプリの方針): 統計にフォールバックする
            decision = make(False, quality.reason)
        else:
            try:
                evaluation = evaluate(samples, self._train_predict)
                if evaluation.adopted:
                    model = self._fit(samples)
                    predictor = LightGBMSleepPredictor(model, feature_importance(model))
                decision = make(
                    evaluation.adopted,
                    evaluation.reason,
                    mae_minutes=evaluation.model_val_mae,
                    baseline_name=evaluation.baseline_name,
                    baseline_mae_minutes=evaluation.baseline_mae,
                    improvement_rate=evaluation.improvement_rate,
                    overfit_gap=evaluation.overfit_gap,
                    feature_importance=predictor.importance() if predictor else {},
                )
            except Exception:
                # 学習・推論の失敗でも回答は止めず、モデルを使わない(Fallback)
                logger.exception("sleep model evaluation failed")
                predictor, evaluation = None, None
                decision = make(False, "モデルの学習・評価でエラーが発生したため使用しません")
        self._registry.save(user_id, decision)
        self._tracker.log_decision(
            user_key(user_id),
            decision,
            {
                "model": "lightgbm",
                "target": "sleep_duration_min(next morning)",
                "features": ",".join(F.FEATURE_NAMES),
                "cv": "forward_chaining_5",
                "baseline": "min(recent_7_nights_mean, training_mean)",
                "min_days": F.MIN_DAYS,
                "max_missing_rate": F.MAX_MISSING_RATE,
                **{f"lgbm_{k}": v for k, v in REGULARIZED_PARAMS.items()},
            },
        )
        return _State(signature, decision, predictor, evaluation, samples, quality, rows)

    def decision(self, user_id: UUID) -> ModelDecision:
        return self._state(user_id).decision

    # ---- 予測 ----
    def predict(self, user_id: UUID, c: PredictionConditions) -> PredictionResult:
        if c.bedtime_hour <= c.last_meal_hour:
            raise InvalidRequestError("bedtime_hour must be later than last_meal_hour")
        st = self._state(user_id)
        inputs, defaulted = self._inputs(st, c)
        if st.predictor is None:
            return PredictionResult(False, None, st.decision, inputs, defaulted, self._fallback(st))
        try:
            value = st.predictor.predict(inputs)
        except Exception:
            logger.exception("sleep prediction failed")
            return PredictionResult(False, None, st.decision, inputs, defaulted, self._fallback(st))
        return PredictionResult(True, value, st.decision, inputs, defaulted, None)

    @staticmethod
    def _inputs(st: _State, c: PredictionConditions) -> tuple[dict[str, float], list[str]]:
        defaulted: list[str] = []
        recent = [r.sleep_duration_min for r in st.rows if r.sleep_duration_min is not None]
        if c.total_kcal is None:
            kcals = [s.features[F.FEATURE_NAMES.index("total_kcal")] for s in st.samples]
            total_kcal = statistics.fmean(kcals) if kcals else 2000.0
            defaulted.append("total_kcal")
        else:
            total_kcal = c.total_kcal
        if recent:
            prev_night = float(recent[-1])
            mean3 = statistics.fmean(recent[-3:])
            defaulted += ["sleep_prev_night_min", "sleep_mean_3"]  # 直近の実績から補う
        else:
            prev_night = mean3 = 420.0
            defaulted += ["sleep_prev_night_min", "sleep_mean_3"]
        kcal = max(0.0, c.workout_kcal)
        intensity = (
            3.0 if kcal >= HIGH_KCAL else 2.0 if kcal >= MODERATE_KCAL else 1.0 if kcal > 0 else 0.0
        )
        inputs = {
            "last_meal_hour": c.last_meal_hour,
            "bedtime_hour": c.bedtime_hour,
            "meal_to_bed_gap_h": c.bedtime_hour - c.last_meal_hour,
            "total_kcal": total_kcal,
            "workout_kcal": kcal,
            "evening_high_workout": 1.0 if c.evening_high_workout else 0.0,
            "workout_intensity": intensity,
            "sleep_prev_night_min": prev_night,
            "sleep_mean_3": mean3,
            "is_weekend": 1.0 if c.is_weekend else 0.0,
        }
        return inputs, defaulted

    @staticmethod
    def _fallback(st: _State) -> FallbackStats:
        durations = [r.sleep_duration_min for r in st.rows if r.sleep_duration_min is not None]
        recent = durations[-F.BASELINE_WINDOW :]
        late = [
            r.sleep_duration_min
            for r in st.rows
            if r.sleep_duration_min is not None and r.prev_day_late_meal == 1.0
        ]
        other = [
            r.sleep_duration_min
            for r in st.rows
            if r.sleep_duration_min is not None and r.prev_day_late_meal == 0.0
        ]
        enough = len(late) >= stats.MIN_GROUP_N and len(other) >= stats.MIN_GROUP_N
        return FallbackStats(
            recent_mean_minutes=statistics.fmean(recent) if recent else None,
            recent_n=len(recent),
            late_meal_mean_minutes=statistics.fmean(late) if enough else None,
            late_meal_n=len(late),
            other_mean_minutes=statistics.fmean(other) if enough else None,
            other_n=len(other),
        )

    # ---- 要因 ----
    def factors(self, user_id: UUID) -> FactorsResult:
        st = self._state(user_id)
        if st.predictor is not None:
            return FactorsResult(True, list(st.predictor.importance().items()), st.decision)
        # モデルを使わない場合は、通常の統計(特徴量と睡眠時間の相関)を返す
        corr: list[tuple[str, float, int, str]] = []
        for i, name in enumerate(F.FEATURE_NAMES):
            xs = [s.features[i] for s in st.samples]
            ys = [s.target for s in st.samples]
            c = stats.correlate_pairs(xs, ys)
            if c.pearson_r is not None:
                corr.append((name, c.pearson_r, c.n, c.interpretation))
        corr.sort(key=lambda t: -abs(t[1]))
        return FactorsResult(False, [], st.decision, corr)


__all__ = [
    "FactorsResult",
    "FallbackStats",
    "PredictionConditions",
    "PredictionResult",
    "SleepModelService",
]
