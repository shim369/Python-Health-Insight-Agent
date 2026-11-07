"""振り返りメモ(DailyJournal)のRAG検索。

流れ: 初回(またはメモが増えたとき)にメモを埋め込んでVectorStoreに入れる → 質問を埋め込んで
類似メモをtop_k件返す。LLMはこの結果(日付つきの原文)だけを根拠に述べる。

コスト管理: 埋め込みはテキストのハッシュでキャッシュし、同じメモを再度埋め込まない。
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import UUID

from health_insight_agent.ai.protocols import EmbeddingProvider
from health_insight_agent.domain.models import DailyJournal
from health_insight_agent.local_time import local_today
from health_insight_agent.rag.vector_store import SearchHit, VectorItem, VectorStore
from health_insight_agent.repositories.protocols import UnitOfWork

MAX_NOTE_CHARS = 1000
"""1メモあたりの最大文字数(埋め込みコスト・LLMへの送信量の上限)。"""


def journal_text(j: DailyJournal) -> str:
    """検索対象にするテキスト列(振り返り系のみ。買い物メモ等は含めない)。"""
    parts = [j.reflection_memo, j.improvement_point, j.diet_policy_note, j.cooking_shopping_note]
    return "\n".join(p.strip() for p in parts if p and p.strip())[:MAX_NOTE_CHARS]


class EmbeddingCache:
    """テキスト→ベクトルのキャッシュ(任意でJSONファイルに永続化。.cache/はgit管理外)。"""

    def __init__(self, namespace: str, path: Path | None = None) -> None:
        self._ns = namespace
        self._path = path
        self._data: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        if path and path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self._ns}\0{text}".encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        with self._lock:
            return self._data.get(self.key(text))

    def put_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        with self._lock:
            for t, v in zip(texts, vectors, strict=True):
                self._data[self.key(t)] = v
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(self._data), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class NoteHit:
    note_date: date
    text: str
    score: float


class NoteSearchService:
    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        cache: EmbeddingCache | None = None,
        clock: Callable[[], date] = local_today,
    ) -> None:
        self._uow_factory = uow_factory
        self._embedder = embedder
        self._store = store
        self._cache = cache
        self._clock = clock
        self._signatures: dict[UUID, tuple[int, object]] = {}
        self._lock = threading.Lock()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._cache is None:
            return self._embedder.embed(texts)
        vectors: list[list[float] | None] = [self._cache.get(t) for t in texts]
        missing = [t for t, v in zip(texts, vectors, strict=True) if v is None]
        if missing:
            fresh = self._embedder.embed(missing)
            self._cache.put_many(missing, fresh)
            by_text = dict(zip(missing, fresh, strict=True))
            vectors = [
                v if v is not None else by_text[t] for t, v in zip(texts, vectors, strict=True)
            ]
        return [v for v in vectors if v is not None]

    def ensure_indexed(self, user_id: UUID) -> int:
        """メモの件数・最終日が前回と変わっていれば作り直す。戻り値は索引中の件数。"""
        with self._uow_factory() as uow:
            b = uow.daily_journals.bounds(user_id)
        signature = (b.count, b.last)
        with self._lock:
            if self._signatures.get(user_id) == signature:
                return self._store.count(user_id)
            if b.count == 0:
                self._store.delete_user(user_id)
                self._signatures[user_id] = signature
                return 0
            with self._uow_factory() as uow:
                journals = uow.daily_journals.list_between_dates(
                    user_id, date(2000, 1, 1), date(2999, 12, 31)
                )
            docs = [(j, journal_text(j)) for j in journals]
            docs = [(j, t) for j, t in docs if t]
            vectors = self._embed([t for _, t in docs]) if docs else []
            self._store.delete_user(user_id)
            self._store.upsert(
                [
                    VectorItem(
                        item_id=str(j.daily_journal_id),
                        user_id=user_id,
                        vector=v,
                        text=t,
                        metadata={"date": j.journal_date.isoformat()},
                    )
                    for (j, t), v in zip(docs, vectors, strict=True)
                ]
            )
            self._signatures[user_id] = signature
            return len(docs)

    def search(
        self,
        user_id: UUID,
        query: str,
        *,
        top_k: int = 5,
        start: date | None = None,
        end: date | None = None,
    ) -> tuple[list[NoteHit], int]:
        """(ヒット, 索引中のメモ総数)。総数0は「検索できるメモが無い」ことを意味する。"""
        total = self.ensure_indexed(user_id)
        if total == 0:
            return [], 0
        [qv] = self._embedder.embed([query])

        def in_range(meta: dict[str, object]) -> bool:
            d = date.fromisoformat(str(meta["date"]))
            return (start is None or d >= start) and (end is None or d <= end)

        hits: list[SearchHit] = self._store.search(user_id, qv, top_k=top_k, where=in_range)
        return (
            [NoteHit(date.fromisoformat(str(h.metadata["date"])), h.text, h.score) for h in hits],
            total,
        )
