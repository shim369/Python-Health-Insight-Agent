"""メモ検索を実DB(合成データ)で検証。仕込んだ「遅い食事」のメモが上位に来ること、ユーザー分離。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest

from health_insight_agent.ai.embeddings import HashingEmbeddingProvider
from health_insight_agent.rag.note_search import NoteSearchService
from health_insight_agent.rag.vector_store import NumpyVectorStore
from tests.conftest import FIXED_END_DATE

pytestmark = pytest.mark.integration


class _CountingEmbedder(HashingEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.texts_embedded = 0

    def embed(self, texts: Any) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return super().embed(texts)


@pytest.fixture
def svc(seeded: tuple[Any, UUID, UUID]) -> tuple[NoteSearchService, _CountingEmbedder, UUID, UUID]:
    factory, uid, other = seeded
    emb = _CountingEmbedder()
    return NoteSearchService(factory, emb, NumpyVectorStore()), emb, uid, other


def test_late_meal_query_surfaces_late_meal_notes(svc: Any) -> None:
    s, _, uid, _ = svc
    hits, total = s.search(uid, "夜遅くに食べてしまった日の振り返り", top_k=5)
    assert total > 30 and len(hits) == 5
    relevant = [h for h in hits if "遅" in h.text and ("食べ" in h.text or "夕食" in h.text)]
    assert len(relevant) >= 4  # 上位5件のほとんどが、仕込んだ遅い食事のメモ
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_date_filter_and_user_isolation(svc: Any) -> None:
    s, _, uid, other = svc
    since = FIXED_END_DATE - timedelta(days=9)
    hits, _ = s.search(uid, "睡眠が短くて眠かった", top_k=10, start=since)
    assert hits and all(h.note_date >= since for h in hits)
    # 別ユーザーのデータは20日分だけ。そのユーザーの検索結果に、最初のユーザー固有のメモは混ざらない
    other_hits, other_total = s.search(other, "睡眠が短くて眠かった", top_k=50)
    assert other_total < 20 and all(
        h.note_date > FIXED_END_DATE - timedelta(days=20) for h in other_hits
    )


def test_unknown_user_has_no_notes(svc: Any) -> None:
    s, *_ = svc
    assert s.search(UUID(int=777), "何でも", top_k=5) == ([], 0)


def test_indexing_is_cached_until_notes_change(svc: Any) -> None:
    s, emb, uid, _ = svc
    s.search(uid, "q1")
    first = emb.texts_embedded
    s.search(uid, "q2")  # 索引は再作成されず、質問の埋め込み1件だけ増える
    assert emb.texts_embedded == first + 1


def test_search_notes_tool_end_to_end(svc: Any, seeded: Any) -> None:
    from health_insight_agent.services.health_insight_service import LocalHealthInsightService
    from health_insight_agent.services.insight_data_service import InsightDataService
    from health_insight_agent.tools.builtin import build_default_registry

    s, _, uid, _ = svc
    factory = seeded[0]
    reg = build_default_registry(
        InsightDataService(factory, clock=lambda: FIXED_END_DATE),
        LocalHealthInsightService(factory),
        s,
    )
    assert "search_notes" in reg.names
    ex = reg.execute(uid, "search_notes", {"query": "夜に激しく運動したら目が冴えた", "top_k": 3})
    assert ex.ok and len(ex.content["data"]["hits"]) == 3
    assert ex.content["evidence"]["rows_used"] == 3
    bad = reg.execute(uid, "search_notes", {"query": "", "top_k": 3})
    assert not bad.ok
    rev = reg.execute(
        uid,
        "search_notes",
        {"query": "x", "start_date": "2026-02-01", "end_date": "2026-01-01"},
    )
    assert not rev.ok and "end_date" in rev.content["error"]
    # ツール無し構成ではsearch_notesは公開されない
    plain = build_default_registry(InsightDataService(factory), LocalHealthInsightService(factory))
    assert "search_notes" not in plain.names
