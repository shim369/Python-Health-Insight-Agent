"""PredictorのProtocolとLightGBM実装。評価・採用判断とは独立して差し替えられる。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from health_insight_agent.ml.features import FEATURE_NAMES


class Predictor(Protocol):
    def predict(self, features: Mapping[str, float]) -> float:
        """特徴量(FEATURE_NAMESの全て)から睡眠時間(分)を予測する。"""
        ...

    def importance(self) -> dict[str, float]:
        """特徴量の重要度(合計1、大きい順)。因果の強さではなく、モデルが使った度合い。"""
        ...


class LightGBMSleepPredictor:
    def __init__(self, model: Any, importance: dict[str, float]) -> None:
        self._model = model
        self._importance = importance

    def predict(self, features: Mapping[str, float]) -> float:
        missing = [n for n in FEATURE_NAMES if n not in features]
        if missing:
            raise ValueError(f"missing features: {missing}")
        row = [[float(features[n]) for n in FEATURE_NAMES]]
        return float(self._model.predict(row)[0])

    def importance(self) -> dict[str, float]:
        return dict(self._importance)
