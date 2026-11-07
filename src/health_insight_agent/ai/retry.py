"""リトライ付き呼び出しの共通処理(LLM生成・埋め込みの両方で同じ方針を使う)。"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from health_insight_agent.ai.retry_policy import RETRYABLE_ERRORS, RetryPolicy
from health_insight_agent.logging_config import get_logger

logger = get_logger(__name__)


def call_with_retry[T](
    func: Callable[[], T],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    label: str = "ai call",
) -> T:
    """429/503(RETRYABLE_ERRORS)だけを、指数バックオフ+ジッターで再試行する。

    それ以外の例外(タイムアウト・検証エラー等)は即座に送出する(無条件には再試行しない)。
    """
    rng = rng or random.Random()
    last: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return func()
        except RETRYABLE_ERRORS as e:
            last = e
            if attempt == policy.max_attempts:
                break
            delay = policy.delay_seconds(attempt) * rng.uniform(
                1 - policy.jitter_ratio, 1 + policy.jitter_ratio
            )
            logger.warning(
                "%s: retryable error (%s); attempt %d/%d, retrying in %.2fs",
                label,
                type(e).__name__,
                attempt,
                policy.max_attempts,
                delay,
            )
            sleep(delay)
    assert last is not None
    raise last
