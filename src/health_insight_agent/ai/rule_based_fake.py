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
    days = 7 if _has(question, "先週", "今週", "week", "直近") else 90  # 既定は全期間(90日)
    return {
        "start_date": (today - timedelta(days=days - 1)).isoformat(),
        "end_date": today.isoformat(),
    }


# (キーワード, 指標)。長いキーワードを先に書く(「朝食」を「食」より先に拾うため)。
_METRIC_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("最後の食事", "prev_day_meal_to_bed_gap_h"),
    ("寝るまでの時間", "prev_day_meal_to_bed_gap_h"),
    ("夜遅い食事", "prev_day_late_meal"),
    ("夕食が遅", "prev_day_late_meal"),
    ("遅い食事", "prev_day_late_meal"),
    ("夜遅くに食", "prev_day_late_meal"),
    ("夜に激しい運動", "prev_day_evening_high_workout"),
    ("夜の運動", "prev_day_evening_high_workout"),
    ("運動で消費", "prev_day_workout_kcal"),
    ("消費", "prev_day_workout_kcal"),
    ("朝食", "prev_day_breakfast_kcal"),
    ("疲労", "same_day_fatigue_score"),
    ("疲れ", "same_day_fatigue_score"),
    ("ストレス", "same_day_stress_score"),
    ("体重", "same_day_weight_kg"),
    ("平日", "is_weekend"),
    ("休日", "is_weekend"),
    ("週末", "is_weekend"),
    ("睡眠時間", "sleep_duration_min"),
    ("睡眠の質", "sleep_quality_score"),
    ("就寝", "sleep_bedtime_hour"),
    ("寝る時間", "sleep_bedtime_hour"),
    ("食事", "prev_day_late_meal"),
    ("運動", "prev_day_evening_high_workout"),
)


def _mentioned_metrics(question: str) -> list[str]:
    """質問文に出てくる指標を、出現順・重複なしで返す(重なる範囲は長いキーワードを優先)。"""
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for kw, metric in _METRIC_KEYWORDS:
        for m in re.finditer(re.escape(kw), question):
            span = (m.start(), m.end())
            if any(span[0] < t1 and t0 < span[1] for t0, t1 in taken):
                continue
            taken.append(span)
            found.append((m.start(), metric))
    out: list[str] = []
    for _, metric in sorted(found):
        if metric not in out:
            out.append(metric)
    return out


_NOTE_FILLERS = re.compile(
    r"(振り返り)?メモ(に|には|で|を)?|書いて(あっ|い)?た?|書いた|何と|について|日の|日は|はある|あった|"
    r"ある\?|教えて|だっ(た)?|[?？。、]|[はにをがでの]"
)


def _note_query(question: str) -> str:
    """質問から定型句を除いた検索語(LLMなら自分で簡潔なクエリを作る部分の代わり)。"""
    cleaned = _NOTE_FILLERS.sub("", question).strip()
    return cleaned or question


def _plan(question: str, today: date) -> list[ToolCall]:
    q = question.lower()
    period = _period(question, today)
    if _has(q, "メモ", "日記", "振り返り", "書いて", "note", "diary"):
        return [ToolCall("c1", "search_notes", {"query": _note_query(question), "top_k": 5})]
    if _has(q, "平日", "休日", "曜日", "sql", "集計"):
        sql = (
            "SELECT CASE WHEN EXTRACT(ISODOW FROM wake_time AT TIME ZONE 'Asia/Tokyo') >= 6 "
            "THEN 'weekend' ELSE 'weekday' END AS day_type, "
            "ROUND(AVG(EXTRACT(EPOCH FROM (wake_time - bedtime)) / 60)::numeric, 1) "
            "AS avg_sleep_min, COUNT(*) AS n FROM sleep_logs GROUP BY 1 ORDER BY 1"
        )
        return [ToolCall("c1", "run_readonly_sql", {"sql": sql})]
    if _has(q, "高強度", "何回") and _has(q, "運動", "workout"):
        return [
            ToolCall(
                "c1",
                "get_workout_logs",
                {**_period("", today), "limit": 200},
            )
        ]
    mentioned = _mentioned_metrics(question)
    asks_relation = _has(
        q, "相関", "関係", "影響", "correlat", "どうなる", "遅め", "夜", "違", "差"
    )
    if len(mentioned) >= 2 or (mentioned and asks_relation):
        if len(mentioned) == 1:  # 1つだけなら、結果側は睡眠時間(就寝の話なら就寝時刻)
            bedtime_topic = _has(q, "就寝", "寝る時間", "bedtime")
            y = "sleep_bedtime_hour" if bedtime_topic else "sleep_duration_min"
            x = mentioned[0]
            if x == y:
                y = "sleep_duration_min"
        else:
            x, y = mentioned[0], mentioned[1]
        return [ToolCall("c1", "correlate", {**period, "x_metric": x, "y_metric": y})]
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
    if _has(q, "スコア", "score", "おすすめ", "recommend", "改善"):
        return [
            ToolCall("c1", "get_health_score", {}),
            ToolCall("c2", "get_recommendations", {}),
        ]
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
        elif name == "get_workout_logs":
            rows = d["rows"]
            high = sum(1 for r in rows if r["intensity"] == "high")
            lines.append(f"- 運動{d['total']}回のうち、高強度は{high}回です(派生指標)。")
        elif name == "run_readonly_sql":
            lines.append(f"- SQL結果({d['row_count']}行, 列={d['columns']}):")
            for r in d["rows"][:5]:
                lines.append(f"  - {r}")
        elif name == "search_notes":
            lines.append(
                f"- 振り返りメモ{d['total_notes_indexed']}件から類似{len(d['hits'])}件を検索:"
            )
            for h in d["hits"][:3]:
                lines.append(f"  - {h['date']}(類似度{h['score']}): {h['text'][:60]}")
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
