"""比較する設定のプリセット(プロンプト・ツール構成・モデルを変えて比べる)。

fake-*  : キー不要のルールベース(LLM抜きのベースライン)。再現性が高く、CIでも回せる。
gemini-*: 実際のGemini。GEMINI_API_KEY が必要(README「実キーで評価する」参照)。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from health_insight_agent.agent.factory import build_registry
from health_insight_agent.agent.loop import AgentConfig, HealthInsightAgent
from health_insight_agent.agent.prompts import NOTES_RULE, SQL_RULE
from health_insight_agent.ai.factory import build_llm_provider
from health_insight_agent.config import Settings


@dataclass(frozen=True, slots=True)
class EvalConfig:
    name: str
    description: str
    provider: str  # fake | gemini
    model: str | None = None
    prompt_variant: str = "full"
    enable_notes: bool = True
    enable_sql: bool = True
    enable_model: bool = True
    max_tool_calls: int = 8


PRESETS: dict[str, EvalConfig] = {
    c.name: c
    for c in (
        EvalConfig("fake-all", "ルールベース(LLM抜き)・全ツール", "fake"),
        EvalConfig(
            "fake-no-model",
            "ルールベース・自作モデルなし(通常ツールのみ。RAG/SQLはあり)",
            "fake",
            enable_model=False,
        ),
        EvalConfig(
            "fake-typed-only",
            "ルールベース・型付きツールのみ(RAG/SQL/モデルなし)",
            "fake",
            enable_notes=False,
            enable_sql=False,
            enable_model=False,
        ),
        EvalConfig("gemini-flash-all", "Gemini(既定モデル)・全ツール・fullプロンプト", "gemini"),
        EvalConfig(
            "gemini-flash-minimal-prompt",
            "Gemini・最小プロンプト(ルール無し)",
            "gemini",
            prompt_variant="minimal",
        ),
        EvalConfig(
            "gemini-flash-no-model",
            "Gemini・自作モデルなし(通常ツールのみ)",
            "gemini",
            enable_model=False,
        ),
        EvalConfig(
            "gemini-flash-typed-only",
            "Gemini・型付きツールのみ",
            "gemini",
            enable_notes=False,
            enable_sql=False,
            enable_model=False,
        ),
        EvalConfig(
            "gemini-flash-lite-all",
            "Gemini Flash-Lite・全ツール(モデル比較)",
            "gemini",
            model="gemini-3.5-flash-lite",
        ),
    )
}


def build_eval_agent(
    settings: Settings, cfg: EvalConfig, *, today: date, **registry_flags: bool
) -> HealthInsightAgent:
    s = replace(settings, ai_provider=cfg.provider, gemini_model=cfg.model)
    registry = build_registry(
        s,
        today=today,
        enable_notes=cfg.enable_notes,
        enable_sql=cfg.enable_sql,
        enable_model=cfg.enable_model,
        **registry_flags,
    )
    rules = tuple(
        r for r, enabled in ((NOTES_RULE, cfg.enable_notes), (SQL_RULE, cfg.enable_sql)) if enabled
    )
    return HealthInsightAgent(
        build_llm_provider(cfg.provider, model=cfg.model),
        registry,
        config=AgentConfig(
            max_tool_calls=cfg.max_tool_calls, extra_rules=rules, prompt_variant=cfg.prompt_variant
        ),
        today=lambda: today,
        sink=lambda _log: None,
    )
