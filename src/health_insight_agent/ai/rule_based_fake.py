"""キー無しで動くルールベースのLLM(AI_PROVIDER=fake)。

本物のLLMではない。キーワードでツールを選び、ツール結果のJSONから決まった文章を組み立てる。
目的: (1) 実APIなしでCLI/API/UIを通しで動かす、(2) 評価の「LLM抜きベースライン」にする。
数値は必ずツール結果からそのまま写す(LLMに計算させない方針を、Fakeでも守る)。
日本語・英語の質問に対応し、システムプロンプトの回答言語(Answer in ...)に合わせて回答する。
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
from health_insight_agent.services.findings import (
    comparison_finding,
    correlation_finding,
)

_TODAY_RE = re.compile(r"today[^0-9]*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


def _today(system: str) -> date:
    m = _TODAY_RE.search(system)
    return date.fromisoformat(m.group(1)) if m else date.today()


def _is_english(system: str) -> bool:
    """システムプロンプトの「Answer in English」から回答言語を判定する(agent/prompts.py)。"""
    return "Answer in English" in system


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


def _period(question: str, today: date) -> dict[str, str]:
    days = (
        7
        if _has(question.lower(), "先週", "今週", "last week", "this week", "直近", "recent")
        else 90
    )  # 既定は全期間(90日)
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
    # --- English(質問は小文字化して照合する)---
    ("time between my last meal", "prev_day_meal_to_bed_gap_h"),
    ("last meal", "prev_day_meal_to_bed_gap_h"),
    ("eating late", "prev_day_late_meal"),
    ("eat late", "prev_day_late_meal"),
    ("late dinner", "prev_day_late_meal"),
    ("late meal", "prev_day_late_meal"),
    ("evening workout", "prev_day_evening_high_workout"),
    ("calories burned", "prev_day_workout_kcal"),
    ("breakfast", "prev_day_breakfast_kcal"),
    ("fatigue", "same_day_fatigue_score"),
    ("tired", "same_day_fatigue_score"),
    ("stress", "same_day_stress_score"),
    ("weight", "same_day_weight_kg"),
    ("weekend", "is_weekend"),
    ("weekday", "is_weekend"),
    ("sleep quality", "sleep_quality_score"),
    ("sleep duration", "sleep_duration_min"),
    ("how long i sleep", "sleep_duration_min"),
    ("bedtime", "sleep_bedtime_hour"),
    ("dinner", "prev_day_late_meal"),
    ("meal", "prev_day_late_meal"),
    ("eating", "prev_day_late_meal"),
    ("workout", "prev_day_evening_high_workout"),
    ("exercise", "prev_day_evening_high_workout"),
    ("sleep", "sleep_duration_min"),
)


def _mentioned_metrics(question: str) -> list[str]:
    """質問文に出てくる指標を、出現順・重複なしで返す(重なる範囲は長いキーワードを優先)。"""
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    question = question.lower()
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


_EN_NOTE_FILLERS = re.compile(
    r"\b(what|did|do|does|i|my|me|write|wrote|written|say|said|notes?|diary|journal|about|in|the|on|"
    r"days?|a|an|have|has|there|any|is|are|was|were|of|to|when|which)\b|[?.,!]",
    re.IGNORECASE,
)


def _note_query(question: str) -> str:
    """質問から定型句を除いた検索語(LLMなら自分で簡潔なクエリを作る部分の代わり)。"""
    cleaned = _NOTE_FILLERS.sub("", question)
    cleaned = _EN_NOTE_FILLERS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or question


_HOUR_RE = re.compile(r"(\d{1,2})時(半)?")


def _predict_args(question: str) -> dict[str, float]:
    """「22時半に夕食を終えて0時に寝たら」→ 最後の食事22.5時・就寝24時。"""
    hours = [int(h) + (0.5 if half else 0.0) for h, half in _HOUR_RE.findall(question)]
    meal = hours[0] if hours else 22.5
    bed = hours[1] if len(hours) > 1 else max(24.0, meal + 1.5)
    if bed < 12:  # 「0時」「1時」は日付をまたいだ深夜
        bed += 24
    if bed <= meal:
        bed += 24
    return {"last_meal_hour": meal, "bedtime_hour": bed}


def _plan(question: str, today: date) -> list[ToolCall]:
    q = question.lower()
    period = _period(question, today)
    if _has(q, "メモ", "日記", "振り返り", "書いて", "note", "diary", "journal", "wrote"):
        return [ToolCall("c1", "search_notes", {"query": _note_query(question), "top_k": 5})]
    if _has(q, "平日", "休日", "曜日", "sql", "集計", "day of week"):
        sql = (
            "SELECT CASE WHEN EXTRACT(ISODOW FROM wake_time AT TIME ZONE 'Asia/Tokyo') >= 6 "
            "THEN 'weekend' ELSE 'weekday' END AS day_type, "
            "ROUND(AVG(EXTRACT(EPOCH FROM (wake_time - bedtime)) / 60)::numeric, 1) "
            "AS avg_sleep_min, COUNT(*) AS n FROM sleep_logs GROUP BY 1 ORDER BY 1"
        )
        return [ToolCall("c1", "run_readonly_sql", {"sql": sql})]
    if _has(q, "高強度", "何回", "high-intensity", "how many") and _has(
        q, "運動", "workout", "exercise"
    ):
        return [
            ToolCall(
                "c1",
                "get_workout_logs",
                {**_period("", today), "limit": 200},
            )
        ]
    asks_forecast = (
        _has(q, "なりそう", "予測", "寝たら", "predict", "will i sleep", "would i sleep")
        and _has(q, "明日", "今夜", "tomorrow", "tonight")
        and _has(q, "食", "dinner", "meal", "eat")
    )
    if asks_forecast:
        return [ToolCall("c1", "predict_sleep_duration", _predict_args(question))]
    if (
        _has(q, "要因", "factor")
        and _has(q, "最も", "一番", "影響している", "most", "main")
        and _has(q, "睡眠", "sleep")
    ):
        return [ToolCall("c1", "explain_sleep_factors", {})]
    mentioned = _mentioned_metrics(question)
    asks_relation = _has(
        q,
        "相関", "関係", "影響", "どうなる", "遅め", "夜", "違", "差",
        "correlat", "affect", "related", "relationship", "influence", "impact", "differ",
    )  # fmt: skip
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
    if _has(q, "共通点", "悪かった", "原因", "common", "why", "bad-sleep", "bad sleep"):
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
    if _has(q, "スコア", "score", "おすすめ", "recommend", "改善", "improve"):
        return [
            ToolCall("c1", "get_health_score", {}),
            ToolCall("c2", "get_recommendations", {}),
        ]
    return [ToolCall("c1", "get_data_overview", {})]


def _summarize(results: list[tuple[str, dict[str, Any]]], *, en: bool = False) -> str:
    def tr(ja: str, en_text: str) -> str:
        return en_text if en else ja

    lines: list[str] = []
    for name, content in results:
        if not content.get("ok"):
            err = content.get("error")
            lines.append(
                tr(
                    f"- {name}: エラー({err})のため確認できませんでした。",
                    f"- {name}: could not be checked because of an error ({err}).",
                )
            )
            continue
        d, ev = content["data"], content["evidence"]
        n = ev.get("rows_used")
        if name == "compare_days":
            cf = comparison_finding(d, "en" if en else "ja")
            lines.append(
                tr(
                    f"- {cf.title}を比較しました(グループA: {cf.n_a}日、グループB: {cf.n_b}日)。",
                    f"- Compared {cf.title} (group A: {cf.n_a} days, group B: {cf.n_b} days).",
                )
            )
            for row in cf.rows:
                lines.append(
                    tr(
                        f"  - {row.label}: A={row.mean_a}{row.unit} / B={row.mean_b}{row.unit}"
                        f" → {row.verdict_text}",
                        f"  - {row.label}: A={row.mean_a} {row.unit} / B={row.mean_b} {row.unit}"
                        f" → {row.verdict_text}",
                    )
                )
        elif name == "correlate":
            f = correlation_finding(d, "en" if en else "ja")
            r = "-" if f.r is None else f"{f.r:.2f}"
            lines.append(
                tr(
                    f"- {f.verdict_text}(r={r}、{f.n}件、{f.reliability_text})。"
                    "相関は因果を意味しません。",
                    f"- {f.verdict_text} (r={r}, n={f.n}; {f.reliability_text}). "
                    "Correlation does not imply causation.",
                )
            )
        elif name == "get_health_score":
            lines.append(
                tr(
                    f"- 総合スコア {d['overall']}/100(独自の目安)。",
                    f"- Overall score {d['overall']}/100 (this app's own guide).",
                )
            )
        elif name == "get_recommendations":
            for r in d["recommendations"][:3]:
                lines.append(
                    tr(
                        f"- 提案({r['domain']}): {r['message']}",
                        f"- Suggestion ({r['domain']}): {r['message']}",
                    )
                )
        elif name == "predict_sleep_duration":
            m = d["model"]
            if d["model_used"]:
                hours, minutes = d["predicted_sleep_hours"], d["predicted_sleep_minutes"]
                gain = round(m["improvement_vs_baseline"] * 100, 1)
                lines.append(
                    tr(
                        f"- 明日の睡眠時間の予測(モデルによる参考値): 約{hours}時間({minutes}分)。"
                        f"モデルの誤差(MAE)は約{m['mae_minutes']}分で、"
                        f"ベースライン(直近平均など)より{gain}%良い結果でした(Baseline比)。"
                        "参考値であり確定ではありません。",
                        f"- Predicted sleep for tomorrow (model-based reference value): about "
                        f"{hours} h ({minutes} min). The model's error (MAE) is about "
                        f"{m['mae_minutes']} min, {gain}% better than the baseline "
                        "(recent average etc.). This is a reference value, not a certainty.",
                    )
                )
            else:
                fb = d.get("fallback") or {}
                lines.append(
                    tr(
                        f"- モデルは使っていません({m['reason']})。",
                        f"- No model was used ({m['reason']}).",
                    )
                )
                if fb.get("recent_mean_minutes") is not None:
                    avg = fb["recent_mean_minutes"]
                    lines.append(
                        tr(
                            f"  参考: 直近の平均睡眠は約{avg}分です。",
                            f"  For reference, your recent average sleep is about {avg} min.",
                        )
                    )
        elif name == "explain_sleep_factors":
            if d["model_used"]:
                top = ", ".join(
                    tr(
                        f"{f['label']}(重要度{f['importance']})",
                        f"{f['feature']} (importance {f['importance']})",
                    )
                    for f in d["factors"][:3]
                )
                lines.append(
                    tr(
                        f"- モデルが睡眠時間の予測に使った主な要因: {top}。"
                        "重要度は予測への寄与で、因果の強さではありません(参考値)。",
                        f"- Main factors the model used to predict sleep: {top}. Importance is "
                        "contribution to the prediction, not a causal effect (reference value).",
                    )
                )
            else:
                lines.append(
                    tr(
                        f"- モデルは使っていません({d['model']['reason']})。",
                        f"- No model was used ({d['model']['reason']}).",
                    )
                )
        elif name == "get_workout_logs":
            rows = d["rows"]
            high = sum(1 for r in rows if r["intensity"] == "high")
            lines.append(
                tr(
                    f"- 運動{d['total']}回のうち、高強度は{high}回です(派生指標)。",
                    f"- Of {d['total']} workouts, {high} were high intensity (derived metric).",
                )
            )
        elif name == "run_readonly_sql":
            lines.append(
                tr(
                    f"- SQL結果({d['row_count']}行, 列={d['columns']}):",
                    f"- SQL result ({d['row_count']} rows, columns={d['columns']}):",
                )
            )
            for r in d["rows"][:5]:
                lines.append(f"  - {r}")
        elif name == "search_notes":
            lines.append(
                tr(
                    f"- 振り返りメモ{d['total_notes_indexed']}件から類似{len(d['hits'])}件を検索:",
                    f"- Searched {d['total_notes_indexed']} reflection notes; "
                    f"{len(d['hits'])} similar notes:",
                )
            )
            for h in d["hits"][:3]:
                lines.append(
                    tr(
                        f"  - {h['date']}(類似度{h['score']}): {h['text'][:60]}",
                        f"  - {h['date']} (similarity {h['score']}): {h['text'][:60]}",
                    )
                )
        elif name == "get_data_overview":
            for t, b in d["tables"].items():
                lines.append(
                    tr(
                        f"- {t}: {b['count']}件 ({b['first']} 〜 {b['last']})",
                        f"- {t}: {b['count']} rows ({b['first']} to {b['last']})",
                    )
                )
        else:
            lines.append(tr(f"- {name}: {n}件を使用しました。", f"- {name}: used {n} rows."))
    lines.append(
        tr(
            "わからないこと: 因果関係の断定はできません。件数が少ない結果は参考程度です。",
            "What is unknown: causation cannot be established; results from few records "
            "are only indicative.",
        )
    )
    return "\n".join(lines)


class RuleBasedFakeLLM:
    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        # 会話履歴が付くことがあるので、「今の質問」=最後のユーザー発話を使う。
        question = next((m.text for m in reversed(messages) if m.role == "user"), "")
        tool_msgs = [m for m in messages if m.role == "tool"]
        usage = TokenUsage(input_tokens=len(question) + 50, output_tokens=40)
        if not tool_msgs:
            calls = _plan(question, _today(system))
            return LLMResponse(Message("assistant", tool_calls=tuple(calls)), usage, "fake-rule")
        results = [(r.name, r.content) for m in tool_msgs for r in m.tool_results]
        text = _summarize(results, en=_is_english(system))
        return LLMResponse(Message("assistant", text=text), usage, "fake-rule")
