"""LLM呼び出しのエラー型。リトライ対象かどうかを型で表す(既存アプリ ai/errors.py と同方針)。"""

from __future__ import annotations


class AIProviderError(Exception):
    """LLM呼び出し全般の失敗(リトライ対象外)。"""


class AIRateLimitError(AIProviderError):
    """429。リトライ対象。"""


class AIServiceUnavailableError(AIProviderError):
    """503。リトライ対象。"""


class AITimeoutError(AIProviderError):
    """1回の試行のタイムアウト。無条件には再試行しない。"""


class AIValidationError(AIProviderError):
    """応答が期待した形式でない。再試行しない。"""
