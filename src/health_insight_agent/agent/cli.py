"""CLIでエージェントに質問する。

uv run python -m health_insight_agent.agent.cli "先週、睡眠が悪かった日の共通点は?"
uv run python -m health_insight_agent.agent.cli --provider gemini --show-tools "質問"
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

from health_insight_agent.agent.factory import build_agent
from health_insight_agent.config import load_settings
from health_insight_agent.domain.enums import Locale


def main() -> None:
    parser = argparse.ArgumentParser(description="Health Insight Agent CLI")
    parser.add_argument("question")
    parser.add_argument("--provider", choices=["fake", "gemini"], default=None)
    parser.add_argument("--lang", choices=["ja", "en"], default="ja")
    parser.add_argument("--today", type=date.fromisoformat, default=None, help="YYYY-MM-DD")
    parser.add_argument("--show-tools", action="store_true", help="使ったツールと結果も表示")
    args = parser.parse_args()

    settings = load_settings()
    if args.provider:
        os.environ["AI_PROVIDER"] = args.provider
        settings = load_settings()
    agent = build_agent(settings, today=args.today)
    answer = agent.ask(settings.demo_user_id, args.question, locale=Locale(args.lang))

    print(answer.text)
    print(
        f"\n[status={answer.status} model={answer.model} llm_calls={answer.llm_calls} "
        f"tool_calls={answer.tool_calls} tokens={answer.usage.total_tokens} "
        f"({answer.usage.input_tokens} in / {answer.usage.output_tokens} out) "
        f"{answer.duration_ms:.0f}ms]"
    )
    if args.show_tools:
        for t in answer.tool_trace:
            print(f"\n--- {t.name} {json.dumps(t.arguments, ensure_ascii=False)} ok={t.ok}")
            print(json.dumps(t.content, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
