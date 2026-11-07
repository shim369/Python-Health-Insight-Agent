"""VectorStoreのProtocolとNumPy実装。

Protocolで抽象化しているのは、後でpgvector(PostgreSQL拡張)に差し替えるため。
どの実装でも **検索は必ず user_id で絞る**(他ユーザーのメモは返さない)。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

import numpy as np


@dataclass(frozen=True, slots=True)
class VectorItem:
    item_id: str
    user_id: UUID
    vector: Sequence[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    item_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def upsert(self, items: Sequence[VectorItem]) -> None: ...

    def search(
        self,
        user_id: UUID,
        query_vector: Sequence[float],
        *,
        top_k: int,
        where: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[SearchHit]: ...

    def count(self, user_id: UUID) -> int: ...

    def delete_user(self, user_id: UUID) -> None: ...


class NumpyVectorStore:
    """メモリ上のコサイン類似度検索。件数が数千程度までならこれで十分(pgvector差し替え口あり)。"""

    def __init__(self) -> None:
        self._items: dict[UUID, dict[str, VectorItem]] = {}
        self._lock = threading.Lock()  # APIのスレッドプールから同時に呼ばれるため

    def upsert(self, items: Sequence[VectorItem]) -> None:
        with self._lock:
            for it in items:
                self._items.setdefault(it.user_id, {})[it.item_id] = it

    def search(
        self,
        user_id: UUID,
        query_vector: Sequence[float],
        *,
        top_k: int,
        where: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[SearchHit]:
        with self._lock:
            # user_idのバケットだけを見る。これが他ユーザー分離の仕組み。
            candidates = [
                it
                for it in self._items.get(user_id, {}).values()
                if where is None or where(it.metadata)
            ]
        if not candidates or top_k <= 0:
            return []
        matrix = np.asarray([c.vector for c in candidates], dtype=np.float64)
        q = np.asarray(query_vector, dtype=np.float64)
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        # ゼロベクトルとの類似度は0とする(0除算を避ける)。
        scores = np.divide(matrix @ q, denom, out=np.zeros(len(candidates)), where=denom > 0)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchHit(
                candidates[i].item_id, candidates[i].text, float(scores[i]), candidates[i].metadata
            )
            for i in order
        ]

    def count(self, user_id: UUID) -> int:
        with self._lock:
            return len(self._items.get(user_id, {}))

    def delete_user(self, user_id: UUID) -> None:
        with self._lock:
            self._items.pop(user_id, None)
