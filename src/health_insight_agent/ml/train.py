"""LightGBMでの学習・推論・特徴量重要度。

小さなデータ(数十〜百件)なので、過学習しないよう強めに正則化する(木を浅く・葉を少なく・
葉あたり最小件数を大きく・L2)。ゆるい設定(150本・葉7)は同じデータで train 9分/val 23分
(乖離60%)となり、過学習チェックで不採用になる(tests/unit/test_ml.py)。
lightgbmは使う関数の中でimportする(評価ロジックだけを軽くテストできるように)。
"""

from __future__ import annotations

from typing import Any

from health_insight_agent.ml.features import FEATURE_NAMES, SleepSample

REGULARIZED_PARAMS: dict[str, Any] = {
    "n_estimators": 60,
    "learning_rate": 0.05,
    "num_leaves": 4,
    "min_child_samples": 10,
    "reg_lambda": 5.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,  # 再現性のため固定
    "n_jobs": 1,  # スレッド数も固定(結果の揺れを避ける)
    "verbose": -1,
}

OVERFIT_PRONE_PARAMS: dict[str, Any] = {
    **REGULARIZED_PARAMS,
    "n_estimators": 150,
    "num_leaves": 7,
    "min_child_samples": 6,
    "reg_lambda": 0.0,
}
"""わざと過学習しやすくした設定(過学習チェックが働くことのテスト用)。"""


def make_regressor(params: dict[str, Any] | None = None) -> Any:
    import lightgbm as lgb

    return lgb.LGBMRegressor(**(params or REGULARIZED_PARAMS))


def lightgbm_train_predict(
    x_train: list[list[float]],
    y_train: list[float],
    x_predict: list[list[float]],
    params: dict[str, Any] | None = None,
) -> list[float]:
    model = make_regressor(params).fit(x_train, y_train)
    return [float(v) for v in model.predict(x_predict)]


def fit_model(samples: list[SleepSample], params: dict[str, Any] | None = None) -> Any:
    return make_regressor(params).fit(
        [list(s.features) for s in samples], [s.target for s in samples]
    )


def feature_importance(model: Any) -> dict[str, float]:
    """gainベースの重要度を、合計1に正規化して返す(大きい順)。

    これは「モデルがどの特徴量を予測に使ったか」で、因果の強さではない。
    """
    gains = [float(g) for g in model.booster_.feature_importance(importance_type="gain")]
    total = sum(gains)
    shares = {
        n: (g / total if total > 0 else 0.0) for n, g in zip(FEATURE_NAMES, gains, strict=True)
    }
    return dict(sorted(shares.items(), key=lambda kv: -kv[1]))
