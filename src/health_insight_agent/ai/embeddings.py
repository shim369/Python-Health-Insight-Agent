"""埋め込みプロバイダ(EmbeddingProvider Protocolの実装)。

- HashingEmbeddingProvider: キー不要・決定的な文字n-gramハッシュ埋め込み(Fake)。意味理解はしないが、
  文字の重なりに基づく類似度は出るので、テスト・デモ・評価の「LLM抜き」構成で使える。
- GeminiEmbeddingProvider: Gemini埋め込みAPI。リトライ/タイムアウトはRetryPolicyに従う。

送るのはメモ本文のみ(user_id・メール等は送らない)。
"""

from __future__ import annotations

import hashlib
import math
import random
import time
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any

from google import genai
from google.genai import errors, types

from health_insight_agent.ai.errors import (
    AIProviderError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
    AIValidationError,
)
from health_insight_agent.ai.retry import call_with_retry
from health_insight_agent.ai.retry_policy import DEFAULT_RETRY_POLICY, RetryPolicy

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
"""公式モデル一覧(2026-11)の最新の埋め込みモデル。GEMINI_EMBEDDING_MODELで変更可。"""
_BATCH_SIZE = 100


class HashingEmbeddingProvider:
    """文字1〜3-gramを符号付きハッシュで固定次元に落とし、L2正規化したベクトル。"""

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        norm = unicodedata.normalize("NFKC", text).lower()
        vec = [0.0] * self._dim
        for n in (1, 2, 3):
            for i in range(len(norm) - n + 1):
                gram = norm[i : i + n]
                if gram.isspace():
                    continue
                # Pythonのhash()は起動ごとに変わるため、md5で決定的にする。
                h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:8], "big")
                # 長いn-gramほど情報量が多いので重みを大きくする。
                vec[h % self._dim] += (1.0 if (h >> 63) & 1 else -1.0) * n
        length = math.sqrt(sum(v * v for v in vec))
        return [v / length for v in vec] if length > 0 else vec


def _is_timeout(e: BaseException) -> bool:
    return isinstance(e, TimeoutError) or any("Timeout" in c.__name__ for c in type(e).__mro__)


class GeminiEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        import os

        self._model = model or os.environ.get("GEMINI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self._policy = retry_policy
        self._sleep = sleep
        self._rng = rng
        if client is not None:
            self._client = client
            return
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY が未設定です(.env.example を参照)。")
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1),
                timeout=int(retry_policy.request_timeout_seconds * 1000),
            ),
        )

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[i : i + _BATCH_SIZE])
            out.extend(
                call_with_retry(
                    lambda b=batch: self._embed_once(b),  # type: ignore[misc]
                    policy=self._policy,
                    sleep=self._sleep,
                    rng=self._rng,
                    label="Gemini embed",
                )
            )
        return out

    def _embed_once(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = self._client.models.embed_content(model=self._model, contents=batch)
        except errors.APIError as e:
            if e.code == 429:
                raise AIRateLimitError(str(e)) from e
            if e.code == 503:
                raise AIServiceUnavailableError(str(e)) from e
            raise AIProviderError(f"Gemini embedding error {e.code}: {e}") from e
        except Exception as e:
            if _is_timeout(e):
                raise AITimeoutError(str(e)) from e
            raise AIProviderError(f"Gemini embedding failed: {e}") from e
        vectors = [list(e.values or []) for e in (resp.embeddings or [])]
        if len(vectors) != len(batch) or any(not v for v in vectors):
            raise AIValidationError("embedding response does not match the request")
        return vectors
