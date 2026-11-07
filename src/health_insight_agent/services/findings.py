"""統計ツールの結果を、人が読める「所見」に直す(決定的。LLMは使わない)。

`correlate` / `compare_days` のツール結果(interpretationなどのコード)から、
- 指標のラベル(「遅い食事(就寝3時間以内)」など。内部名は出さない)
- 平易な1文(「『遅い食事』の日は『睡眠時間』が短い傾向」)
- 強さ・向き・信頼度
を日英で作る。UIの結果カードと、キー無しFake LLMの文面が同じ表現を使う。
数値はツール結果をそのまま写し、再計算しない。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from health_insight_agent.services.metrics import METRICS, mean_display, metric_display
from health_insight_agent.tools.base import ToolExecution

# 指標ごとの「大きい/小さい」の言い方: (大: 形容詞, 副詞形), (小: 形容詞, 副詞形)。英語は比較級。
_WORDS_JA: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "sleep_duration_min": (("長い", "長く"), ("短い", "短く")),
    "sleep_bedtime_hour": (("遅い", "遅く"), ("早い", "早く")),
    "sleep_quality_score": (("良い", "良く"), ("悪い", "悪く")),
    "prev_day_total_kcal": (("多い", "多く"), ("少ない", "少なく")),
    "prev_day_breakfast_kcal": (("多い", "多く"), ("少ない", "少なく")),
    "prev_day_last_meal_hour": (("遅い", "遅く"), ("早い", "早く")),
    "prev_day_meal_to_bed_gap_h": (("長い", "長く"), ("短い", "短く")),
    "prev_day_workout_kcal": (("多い", "多く"), ("少ない", "少なく")),
    "same_day_weight_kg": (("重い", "重く"), ("軽い", "軽く")),
}
_WORDS_EN: dict[str, tuple[str, str]] = {
    "sleep_duration_min": ("longer", "shorter"),
    "sleep_bedtime_hour": ("later", "earlier"),
    "sleep_quality_score": ("better", "worse"),
    "prev_day_total_kcal": ("higher", "lower"),
    "prev_day_breakfast_kcal": ("higher", "lower"),
    "prev_day_last_meal_hour": ("later", "earlier"),
    "prev_day_meal_to_bed_gap_h": ("longer", "shorter"),
    "prev_day_workout_kcal": ("higher", "lower"),
    "same_day_weight_kg": ("heavier", "lighter"),
}
_DEFAULT_JA = (("高い", "高く"), ("低い", "低く"))
_DEFAULT_EN = ("higher", "lower")

_STRENGTH = {
    "strong": ("強い", "strong"),
    "moderate": ("中程度の", "moderate"),
    "weak": ("弱い", "weak"),
}
_OPS_JA = {
    ">": "より大きい日",
    ">=": "以上の日",
    "<": "より小さい日",
    "<=": "以下の日",
    "==": "と等しい日",
    "!=": "以外の日",
}


def _en(locale: str) -> bool:
    return locale == "en"


def metric_label(name: str, locale: str) -> str:
    m = METRICS.get(name)
    if m is None:
        return name
    return m.label_en if _en(locale) else m.label_ja


def _quote(text: str, locale: str) -> str:
    return f"“{text}”" if _en(locale) else f"「{text}」"


def _is_flag(name: str) -> bool:
    m = METRICS.get(name)
    return m is not None and m.kind == "flag"


def reliability_text(reliability: str, locale: str) -> str:
    table = {
        "ok": ("信頼度: 高", "Reliability: high"),
        "low": ("信頼度: 低(件数が少なめ)", "Reliability: low (few records)"),
        "insufficient": (
            "信頼度: 判断不可(件数不足)",
            "Reliability: cannot judge (too few records)",
        ),
    }
    ja, en = table.get(reliability, (reliability, reliability))
    return en if _en(locale) else ja


def _direction_phrase(x: str, y: str, sign: str, locale: str) -> str:
    """「Xが大きい(Xの日)ほど Yは ○○ なる傾向」の文。"""
    qx, qy = _quote(metric_label(x, locale), locale), _quote(metric_label(y, locale), locale)
    positive = sign == "positive"
    if _en(locale):
        word = _WORDS_EN.get(y, _DEFAULT_EN)[0 if positive else 1]
        if _is_flag(x):
            return f"on days with {qx}, {qy} tends to be {word}"
        return f"the higher {qx} is, the {word} {qy} tends to be"
    (adj_hi, adv_hi), (adj_lo, adv_lo) = _WORDS_JA.get(y, _DEFAULT_JA)
    if _is_flag(x):
        return f"{qx}の日は{qy}が{adj_hi if positive else adj_lo}傾向"
    return f"{qx}が大きいほど、{qy}は{adv_hi if positive else adv_lo}なる傾向"


@dataclass(frozen=True, slots=True)
class CorrelationFinding:
    title: str
    verdict: str
    """ツールのinterpretation(コード)。"""
    verdict_text: str
    """平易な1文。"""
    tone: str
    """negative / positive / none / unknown(色分け用)。"""
    strength: float | None
    """|r|(0〜1)。判定不能ならNone。"""
    r: float | None
    n: int
    reliability: str
    reliability_text: str


def correlation_finding(data: dict[str, Any], locale: str) -> CorrelationFinding:
    x, y = data["x_metric"], data["y_metric"]
    interp: str = data["interpretation"]
    r = data.get("pearson_r")
    title = f"{metric_label(x, locale)} × {metric_label(y, locale)}"
    qx, qy = _quote(metric_label(x, locale), locale), _quote(metric_label(y, locale), locale)
    if interp == "insufficient_data":
        text = (
            f"Too few records to judge the relationship between {qx} and {qy}"
            if _en(locale)
            else f"{qx}と{qy}の関係は、件数が少なく判断できません"
        )
        tone = "unknown"
    elif interp == "no_clear_relationship":
        text = (
            f"No clear relationship between {qx} and {qy}"
            if _en(locale)
            else f"{qx}と{qy}に明確な関係は見られません"
        )
        tone = "none"
    else:
        level, sign = interp.split("_", 1)
        ja_s, en_s = _STRENGTH[level]
        kind_ja = "正の相関" if sign == "positive" else "負の相関"
        kind_en = "positive" if sign == "positive" else "negative"
        head = f"{en_s.capitalize()} {kind_en} correlation" if _en(locale) else f"{ja_s}{kind_ja}"
        text = f"{head}: {_direction_phrase(x, y, sign, locale)}"
        tone = sign
    return CorrelationFinding(
        title=title,
        verdict=interp,
        verdict_text=text,
        tone=tone,
        strength=None if r is None else abs(float(r)),
        r=None if r is None else float(r),
        n=int(data["n_pairs"]),
        reliability=data["reliability"],
        reliability_text=reliability_text(data["reliability"], locale),
    )


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    label: str
    mean_a: float | None
    mean_b: float | None
    n_a: int
    n_b: int
    verdict: str
    verdict_text: str
    tone: str
    unit: str
    """平均値の表示単位(睡眠時間は「時間」)。"""


@dataclass(frozen=True, slots=True)
class ComparisonFinding:
    title: str
    """どの群を比べたか(例: 「遅い食事」の日 vs それ以外)。"""
    n_a: int
    n_b: int
    rows: list[ComparisonRow] = field(default_factory=list)


def _condition_title(cond: dict[str, Any], locale: str) -> str:
    metric, op, value = cond["metric"], cond["op"], cond["value"]
    label = _quote(metric_label(metric, locale), locale)
    if _is_flag(metric) and op == "==" and float(value) == 1:
        return f"{label} days vs other days" if _en(locale) else f"{label}の日 vs それ以外の日"
    divisor, unit = metric_display(metric, locale)
    number = float(value) / divisor if isinstance(value, int | float) else value
    shown = f"{number:g}{'' if _en(locale) else unit}"
    if _en(locale):
        return f"Days where {label} {op} {shown}{' ' + unit if unit else ''} vs other days"
    return f"{label}が{shown}{_OPS_JA.get(op, op)} vs それ以外の日"


def comparison_finding(data: dict[str, Any], locale: str) -> ComparisonFinding:
    rows: list[ComparisonRow] = []
    for metric, c in data["metrics"].items():
        interp = c["interpretation"]
        label = metric_label(metric, locale)
        if _en(locale):
            hi, lo = _WORDS_EN.get(metric, _DEFAULT_EN)
        else:
            (hi, _), (lo, _) = _WORDS_JA.get(metric, _DEFAULT_JA)
        if interp == "insufficient_data":
            text = "Too few records to judge" if _en(locale) else "件数不足で判断できません"
            tone = "unknown"
        elif interp == "no_clear_difference":
            text = "No clear difference" if _en(locale) else "差は見られません"
            tone = "none"
        else:
            higher = interp == "a_higher"
            word = hi if higher else lo
            text = f"Group A is {word}" if _en(locale) else f"グループAの方が{word}"
            tone = "positive" if higher else "negative"
        mean_a, unit = mean_display(metric, c["group_a"]["mean"], locale)
        mean_b, _ = mean_display(metric, c["group_b"]["mean"], locale)
        rows.append(
            ComparisonRow(
                label=label,
                mean_a=mean_a,
                mean_b=mean_b,
                unit=unit,
                n_a=int(c["group_a"]["n"]),
                n_b=int(c["group_b"]["n"]),
                verdict=interp,
                verdict_text=text,
                tone=tone,
            )
        )
    return ComparisonFinding(
        title=_condition_title(data["condition"], locale),
        n_a=int(data["n_group_a"]),
        n_b=int(data["n_group_b"]),
        rows=rows,
    )


Finding = CorrelationFinding | ComparisonFinding


def build_findings(trace: Sequence[ToolExecution], locale: str) -> list[Finding]:
    """成功した correlate / compare_days の結果だけを所見にする。"""
    out: list[Finding] = []
    for t in trace:
        if not t.ok:
            continue
        data = t.content["data"]
        if t.name == "correlate":
            out.append(correlation_finding(data, locale))
        elif t.name == "compare_days":
            out.append(comparison_finding(data, locale))
    return out
