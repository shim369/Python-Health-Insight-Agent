"""Rule-based Recommendation。出典: 既存アプリ recommendation/rule_based.py を移植。

各領域の所見(findings)を改善提案として並べ、スコアが低い領域ほど優先度を高くする。
"""

from __future__ import annotations

from dataclasses import dataclass

from health_insight_agent.analysis.health_score import HealthScore


@dataclass(frozen=True, kw_only=True, slots=True)
class Recommendation:
    domain: str
    message: str
    priority: int
    """大きいほど優先度が高い(その領域のスコアが低いほど大きい)。"""


def generate_recommendations(score: HealthScore) -> list[Recommendation]:
    by_domain = {
        "nutrition": score.nutrition,
        "weight": score.weight,
        "sleep": score.sleep,
        "workout": score.workout,
    }
    recs = [
        Recommendation(domain=d, message=f, priority=round(100 - r.score))
        for d, r in by_domain.items()
        for f in r.findings
    ]
    recs.sort(key=lambda r: r.priority, reverse=True)
    return recs
