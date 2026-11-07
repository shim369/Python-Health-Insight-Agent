"""採用判断の保存(JSON)。出典: 既存アプリ ml/model_registry.py の考え方。

保存するのは「採用判断とその根拠」(指標・日数・重要度)だけで、学習済みモデル自体は保存しない
(数十件の学習は一瞬なので、必要なときに再学習する。再現性のため乱数は固定)。
ユーザーごとに別ファイル。ファイル名にはuser_idのハッシュを使う。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import UUID


def user_key(user_id: UUID) -> str:
    """user_idそのものをファイル名・ログに出さないための短いハッシュ。"""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ModelDecision:
    domain: str
    adopted: bool
    reason: str
    total_days: int
    usable_samples: int
    missing_rate: float
    data_signature: str
    trained_at: str
    mae_minutes: float | None = None
    baseline_name: str | None = None
    baseline_mae_minutes: float | None = None
    improvement_rate: float | None = None
    overfit_gap: float | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    reason_en: str = ""
    """reasonの英語版(英語UI向け)。古い保存ファイルには無いので既定は空。"""

    def localized_reason(self, locale: str) -> str:
        return self.reason_en if locale == "en" and self.reason_en else self.reason


class ModelRegistry:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, user_id: UUID, domain: str) -> Path:
        return self._dir / f"{domain}_{user_key(user_id)}.json"

    def save(self, user_id: UUID, decision: ModelDecision) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(user_id, decision.domain)
        path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), "utf-8")
        return path

    def load(self, user_id: UUID, domain: str = "sleep") -> ModelDecision | None:
        path = self._path(user_id, domain)
        if not path.exists():
            return None
        return ModelDecision(**json.loads(path.read_text("utf-8")))
