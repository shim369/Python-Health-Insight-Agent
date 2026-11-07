"""グラフ仕様(JSON)の生成。ツール結果から決定的に作る。

回答の数値グラフはLLMに作らせない。ツールが返した値そのものから作るので、本文の数値と
グラフが食い違わない。LLMには渡さない(`ToolOutput.charts`は`ToolExecution.charts`に入り、
UI/APIにだけ届く)ため、トークンも増えない。
"""

from __future__ import annotations

from typing import Any

Chart = dict[str, Any]


def line_chart(title: str, points: list[tuple[str, float]], *, y_label: str = "") -> Chart:
    return {
        "kind": "line",
        "title": title,
        "y_label": y_label,
        "series": [{"name": y_label or title, "points": [{"x": x, "y": y} for x, y in points]}],
    }


def bar_chart(title: str, points: list[tuple[str, float]], *, y_label: str = "") -> Chart:
    return {
        "kind": "bar",
        "title": title,
        "y_label": y_label,
        "series": [{"name": y_label or title, "points": [{"x": x, "y": y} for x, y in points]}],
    }


def scatter_chart(
    title: str, points: list[tuple[float, float]], *, x_label: str, y_label: str
) -> Chart:
    return {
        "kind": "scatter",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "series": [{"name": title, "points": [{"x": x, "y": y} for x, y in points]}],
    }


def value_card(title: str, value: float, *, unit: str, label: str, note: str = "") -> Chart:
    """単一の値(予測値など)。`label`は「モデルによる参考値」のような区別用ラベル。"""
    return {
        "kind": "value",
        "title": title,
        "value": value,
        "unit": unit,
        "label": label,
        "note": note,
    }
