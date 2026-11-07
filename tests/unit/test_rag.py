from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from google.genai import errors

from health_insight_agent.ai.embeddings import GeminiEmbeddingProvider, HashingEmbeddingProvider
from health_insight_agent.ai.errors import AIRateLimitError, AIValidationError
from health_insight_agent.rag.note_search import EmbeddingCache, journal_text
from health_insight_agent.rag.vector_store import NumpyVectorStore, VectorItem

A, B = UUID(int=1), UUID(int=2)


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_normalized_and_lexically_meaningful() -> None:
    e = HashingEmbeddingProvider()
    v1, v2, other = e.embed(["夜遅くに食べてしまった", "夜遅くに食べた", "今日は仕事が忙しかった"])
    assert e.embed(["夜遅くに食べてしまった"])[0] == v1  # プロセスをまたいでも同じ(md5)
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)
    assert _cos(v1, v2) > _cos(v1, other) + 0.2
    assert e.embed([""])[0] == [0.0] * 512  # 空文字でも落ちない


def test_store_ranks_by_cosine_and_respects_top_k_and_filter() -> None:
    s = NumpyVectorStore()
    s.upsert(
        [
            VectorItem("x", A, [1.0, 0.0], "x", {"date": "2026-01-01"}),
            VectorItem("y", A, [0.7, 0.7], "y", {"date": "2026-01-02"}),
            VectorItem("z", A, [0.0, 1.0], "z", {"date": "2026-01-03"}),
        ]
    )
    hits = s.search(A, [1.0, 0.1], top_k=2)
    assert [h.item_id for h in hits] == ["x", "y"] and hits[0].score > hits[1].score
    only_late = s.search(A, [1.0, 0.1], top_k=5, where=lambda m: m["date"] >= "2026-01-02")
    assert [h.item_id for h in only_late] == ["y", "z"]
    assert s.search(A, [1.0, 0.0], top_k=0) == []


def test_store_never_returns_another_users_items() -> None:
    s = NumpyVectorStore()
    s.upsert([VectorItem("a1", A, [1.0, 0.0], "A's note"), VectorItem("b1", B, [1.0, 0.0], "B's")])
    assert [h.item_id for h in s.search(A, [1.0, 0.0], top_k=10)] == ["a1"]
    assert [h.item_id for h in s.search(B, [1.0, 0.0], top_k=10)] == ["b1"]
    assert s.search(UUID(int=3), [1.0, 0.0], top_k=10) == []
    s.delete_user(A)
    assert s.count(A) == 0 and s.count(B) == 1


def test_store_upsert_replaces_same_id_and_handles_zero_vectors() -> None:
    s = NumpyVectorStore()
    s.upsert([VectorItem("a", A, [1.0, 0.0], "old")])
    s.upsert([VectorItem("a", A, [0.0, 1.0], "new"), VectorItem("zero", A, [0.0, 0.0], "z")])
    assert s.count(A) == 2
    hits = s.search(A, [0.0, 1.0], top_k=2)
    assert hits[0].text == "new" and hits[1].score == 0.0


def test_embedding_cache_avoids_repeat_and_persists(tmp_path: Any) -> None:
    path = tmp_path / "c.json"
    c = EmbeddingCache("ns", path)
    assert c.get("hello") is None
    c.put_many(["hello"], [[0.1, 0.2]])
    assert EmbeddingCache("ns", path).get("hello") == [0.1, 0.2]
    assert EmbeddingCache("other-model", path).get("hello") is None  # モデルが違えば別物


def test_journal_text_uses_reflection_fields_only_and_truncates() -> None:
    from datetime import date
    from uuid import uuid4

    from health_insight_agent.domain.models import DailyJournal

    j = DailyJournal(
        daily_journal_id=uuid4(),
        user_id=A,
        journal_date=date(2026, 1, 1),
        reflection_memo="振り返り",
        improvement_point="改善",
        shopping_memo="牛乳を買う",
    )
    t = journal_text(j)
    assert "振り返り" in t and "改善" in t and "牛乳" not in t
    long = DailyJournal(
        daily_journal_id=uuid4(),
        user_id=A,
        journal_date=date(2026, 1, 1),
        reflection_memo="あ" * 5000,
    )
    assert len(journal_text(long)) == 1000


class _EmbedModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.batches: list[list[str]] = []
        self.configs: list[Any] = []

    def embed_content(self, *, model: str, contents: list[Any], config: Any = None) -> Any:
        self.configs.append(config)
        self.batches.append(contents)
        out = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(out, BaseException):
            raise out
        n = len(contents) if out is None else out
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[float(i), 1.0]) for i in range(n)]
        )


def _gemini(outcomes: list[Any]) -> tuple[GeminiEmbeddingProvider, _EmbedModels, list[float]]:
    models = _EmbedModels(outcomes)
    sleeps: list[float] = []
    p = GeminiEmbeddingProvider(
        client=SimpleNamespace(models=models), sleep=sleeps.append, rng=random.Random(0), model="m"
    )
    return p, models, sleeps


def test_gemini_embeddings_batch_by_100_and_keep_order() -> None:
    p, models, _ = _gemini([])
    vectors = p.embed([f"t{i}" for i in range(250)])
    assert [len(b) for b in models.batches] == [100, 100, 50] and len(vectors) == 250


def test_gemini_embeddings_retry_429_then_succeed() -> None:
    p, models, sleeps = _gemini([errors.APIError(429, {})])
    assert len(p.embed(["a", "b"])) == 2
    assert len(models.batches) == 2 and len(sleeps) == 1


def test_gemini_embeddings_give_up_and_validate_response() -> None:
    p, _, sleeps = _gemini([errors.APIError(429, {})] * 3)
    with pytest.raises(AIRateLimitError):
        p.embed(["a"])
    assert len(sleeps) == 2
    p2, _, _ = _gemini([1])  # 2件要求に1件だけ返る
    with pytest.raises(AIValidationError):
        p2.embed(["a", "b"])


def test_each_text_is_sent_as_its_own_content_with_a_reduced_dimension() -> None:
    """gemini-embedding-2はリストを1つの入力に合成する(実キーで確認)ので、1件ずつContentで包む。"""
    from google.genai import types

    p, models, _ = _gemini([])
    p.embed(["夜遅くに食べた", "朝は忙しかった"])
    [batch] = models.batches
    assert all(isinstance(c, types.Content) for c in batch)
    assert [c.parts[0].text for c in batch] == ["夜遅くに食べた", "朝は忙しかった"]
    assert models.configs[0].output_dimensionality == 768
