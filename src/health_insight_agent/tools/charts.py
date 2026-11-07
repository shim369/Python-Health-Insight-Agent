"""グラフ仕様(JSON)の生成。ツール結果から決定的に作る。

回答の数値グラフはLLMに作らせない。ツールが返した値そのものから作るので、本文の数値と
グラフが食い違わない。LLMには渡さない(`ToolOutput.charts`は`ToolExecution.charts`に入り、
UI/APIにだけ届く)ため、トークンも増えない。

文言(タイトル・軸・ラベル)は `tr(ja, en)` で日英の両方を持たせ、APIが利用者の言語で解決する
(`localize`)。ツールは言語を知らなくてよく、MCP経由でも両方の言語がそのまま運ばれる。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

Chart = dict[str, Any]
Text = str | dict[str, str]
"""そのままの文字列、または {"ja": ..., "en": ...}。"""


def tr(ja: str, en: str) -> dict[str, str]:
    return {"ja": ja, "en": en}


def localize(obj: Any, locale: str) -> Any:
    """入れ子のデータ中の {"ja","en"} を、指定した言語の文字列に置き換える。"""
    if isinstance(obj, dict):
        if set(obj) == {"ja", "en"} and all(isinstance(v, str) for v in obj.values()):
            return obj.get(locale, obj["en"])
        return {k: localize(v, locale) for k, v in obj.items()}
    if isinstance(obj, list):
        return [localize(v, locale) for v in obj]
    return obj


def _name(title: Text, y_label: Text) -> Text:
    return y_label or title


def line_chart(title: Text, points: Sequence[tuple[Text, float]], *, y_label: Text = "") -> Chart:
    return {
        "kind": "line",
        "title": title,
        "y_label": y_label,
        "series": [
            {"name": _name(title, y_label), "points": [{"x": x, "y": y} for x, y in points]}
        ],
    }


def bar_chart(title: Text, points: Sequence[tuple[Text, float]], *, y_label: Text = "") -> Chart:
    return {
        "kind": "bar",
        "title": title,
        "y_label": y_label,
        "series": [
            {"name": _name(title, y_label), "points": [{"x": x, "y": y} for x, y in points]}
        ],
    }


def scatter_chart(
    title: Text, points: list[tuple[float, float]], *, x_label: Text, y_label: Text
) -> Chart:
    return {
        "kind": "scatter",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "series": [{"name": title, "points": [{"x": x, "y": y} for x, y in points]}],
    }


def value_card(title: Text, value: float, *, unit: Text, label: Text, note: Text = "") -> Chart:
    """単一の値(予測値など)。`label`は「モデルによる参考値」のような区別用ラベル。"""
    return {
        "kind": "value",
        "title": title,
        "value": value,
        "unit": unit,
        "label": label,
        "note": note,
    }
