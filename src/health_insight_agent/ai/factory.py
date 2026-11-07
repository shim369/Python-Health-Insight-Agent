"""AI_PROVIDER=fake|gemini の切替(既存アプリの config.get_ai_provider と同じ方式)。"""

from __future__ import annotations

from health_insight_agent.ai.protocols import EmbeddingProvider, LLMProvider


def build_llm_provider(provider: str, *, model: str | None = None) -> LLMProvider:
    """`fake`はキー不要のルールベース、それ以外(既定)はGemini。"""
    if provider.lower() == "fake":
        from health_insight_agent.ai.rule_based_fake import RuleBasedFakeLLM

        return RuleBasedFakeLLM()
    from health_insight_agent.ai.gemini_provider import GeminiProvider

    return GeminiProvider(model=model)


def build_embedding_provider(provider: str) -> EmbeddingProvider:
    """AI_PROVIDER=fake はキー不要のハッシュ埋め込み、それ以外はGemini埋め込み。"""
    if provider.lower() == "fake":
        from health_insight_agent.ai.embeddings import HashingEmbeddingProvider

        return HashingEmbeddingProvider()
    from health_insight_agent.ai.embeddings import GeminiEmbeddingProvider

    return GeminiEmbeddingProvider()
