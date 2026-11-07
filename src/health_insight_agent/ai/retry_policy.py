"""LLM呼び出しのリトライ/タイムアウト方針。

既存アプリ(ai/retry_policy.py)と同じ値だが、1回あたりのタイムアウトだけ20秒に短縮する
(既存は60秒)。エージェントは1つの質問で複数回LLMを呼ぶため、1回が長いと質問全体が
極端に遅くなる(docs/requirements.md 決定3)。
"""

from __future__ import annotations

from dataclasses import dataclass

from health_insight_agent.ai.errors import AIRateLimitError, AIServiceUnavailableError


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    """初回を含む最大試行回数(=初回+リトライ2回)。"""
    base_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.2
    """待機時間に±20%のランダムなぶれを加える(同時リトライの集中を避ける)。"""
    request_timeout_seconds: float = 20.0

    def delay_seconds(self, attempt: int) -> float:
        """`attempt`回目(1始まり)の失敗後に待つ基準秒数(ジッターなし)。"""
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        return self.base_delay_seconds * (self.backoff_multiplier ** (attempt - 1))


DEFAULT_RETRY_POLICY = RetryPolicy()

RETRYABLE_ERRORS: tuple[type[Exception], ...] = (AIRateLimitError, AIServiceUnavailableError)
"""リトライ対象は429/503のみ。Timeout・Validationは即失敗させる。"""
