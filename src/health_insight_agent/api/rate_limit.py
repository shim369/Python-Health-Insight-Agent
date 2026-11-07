"""簡易レート制限(プロセス内・クライアントIPごと)。

公開デモでLLM APIの費用が暴走しないための最低限の防御。複数プロセス構成にするなら
Redis等の共有ストアに置き換える(既知の制限としてREADMEに記載)。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._max:
            retry_after = max(1, int(self._window - (now - hits[0])))
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)

    def dependency(self) -> Callable[[Request], None]:
        def _dep(request: Request) -> None:
            self.check(request.client.host if request.client else "unknown")

        return _dep
