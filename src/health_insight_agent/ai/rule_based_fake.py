"""キー無しで動くルールベースのLLM(AI_PROVIDER=fake)。

本物のLLMではない。キーワードでツールを選び、ツール結果のJSONから決まった文章を組み立てる。
目的: (1) 実APIなしでCLI/API/UIを通しで動かす、(2) 評価の「LLM抜きベースライン」にする。
数値は必ずツール結果からそのまま写す(LLMに計算させない方針を、Fakeでも守る)。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from health_insight_agent.ai.protocols import (
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

_TODAY_RE = re.compile(r"today[^0-9]*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


def _today(system: str) -> date:
    m = _TODAY_RE.search(system)
    return date.fromisoformat(m.group(1)) if m else date.today()


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


def _period(question: str, today: date) -> dict[str, str]:
    days = 7 if _has(question, "先週", "今週", "week", "直近") else 89
    return {
        "start_date": (today - timedelta(days=days - 1)).isoformat(),
        "end_date": today.isoformat(),
    }


def _plan(question: str, today: date) -> list[ToolCall]:
    q = question.lower()
    period = _period(question, today)
    if _has(q, "スコア", "score", "おすすめ", "recommend", "改善"):
        return [
            ToolCall("c1", "get_health_score", {}),
            ToolCall("c2", "get_recommendations", {}),
        ]
    if _has(q, "共通点", "悪かった", "原因", "common", "why"):
        return [
            ToolCall(
                "c1",
                "compare_days",
                {
                    **period,
                    "condition": {"metric": "sleep_duration_min", "op": "<", "value": 390},
                    "compare_metrics": [
                        "prev_day_late_meal",
                        "prev_day_evening_high_workout",
                        "prev_day_breakfast_kcal",
                        "is_weekend",
                    ],
                },
            )
        ]
    if _has(q, "相関", "関係", "影響", "correlat", "どうなる", "遅め", "夜"):
        x = (
            "prev_day_breakfast_kcal"  # 「朝食」は「食」より先に判定する(部分一致の取り違え防止)
            if _has(q, "朝食", "breakfast")
            else "prev_day_late_meal"
            if _has(q, "食", "meal", "eat")
            else "prev_day_evening_high_workout"
            if _has(q, "運動", "workout")
            else "same_day_weight_kg"
            if _has(q, "体重")
            else "is_weekend"
        )
        y = "sleep_bedtime_hour" if _has(q, "就寝", "寝る時間", "bedtime") else "sleep_duration_min"
        return [ToolCall("c1", "correlate", {**period, "x_metric": x, "y_metric": y})]
    return [ToolCall("c1", "get_data_overview", {})]


def _summarize(results: list[tuple[str, dict[str, Any]]]) -> str:
    lines: list[str] = []
    for name, content in results:
        if not content.get("ok"):
            lines.append(f"- {name}: エラー({content.get('error')})のため確認できませんでした。")
            continue
        d, ev = content["data"], content["evidence"]
        n = ev.get("rows_used")
        if name == "compare_days":
            lines.append(
                f"- {ev.get('period')} の{d['n_days_with_condition_data']}日を、条件に当てはまる"
                f"{d['n_group_a']}日(A)とそれ以外{d['n_group_b']}日(B)に分けて比較しました。"
            )
            for metric, c in d["metrics"].items():
                lines.append(
                    f"  - {metric}: A平均={c['group_a']['mean']}, B平均={c['group_b']['mean']}, "
                    f"判定={c['interpretation']}(信頼度={c['reliability']}, p={c['p_value']})"
                )
        elif name == "correlate":
            lines.append(
                f"- {d['x_metric']} と {d['y_metric']}: {d['n_pairs']}件, r={d['pearson_r']}, "
                f"判定={d['interpretation']}(信頼度={d['reliability']})。相関は因果を意味しません。"
            )
        elif name == "get_health_score":
            lines.append(f"- 総合スコア {d['overall']}/100(独自の目安)。")
        elif name == "get_recommendations":
            for r in d["recommendations"][:3]:
                lines.append(f"- 提案({r['domain']}): {r['message']}")
        elif name == "get_data_overview":
            for t, b in d["tables"].items():
                lines.append(f"- {t}: {b['count']}件 ({b['first']} 〜 {b['last']})")
        else:
            lines.append(f"- {name}: {n}件を使用しました。")
    lines.append("わからないこと: 因果関係の断定はできません。件数が少ない結果は参考程度です。")
    return "\n".join(lines)


class RuleBasedFakeLLM:
    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        question = next((m.text for m in messages if m.role == "user"), "")
        tool_msgs = [m for m in messages if m.role == "tool"]
        usage = TokenUsage(input_tokens=len(question) + 50, output_tokens=40)
        if not tool_msgs:
            calls = _plan(question, _today(system))
            return LLMResponse(Message("assistant", tool_calls=tuple(calls)), usage, "fake-rule")
        results = [(r.name, r.content) for m in tool_msgs for r in m.tool_results]
        return LLMResponse(Message("assistant", text=_summarize(results)), usage, "fake-rule")
