"""自動採点の部品。LLMによる採点は使わず、(1)ツール結果(trace)と(2)回答テキストを決定的に検査する。

- relation_sign: traceから「2指標の関係」(負/正/なし/不明)を、ツールが計算した解釈から読み取る。
  これは「エージェントが正しい証拠を取りに行けたか」の判定で、LLMの言い回しに左右されない。
- 回答テキストは正規表現で検査(言い換えに強いよう、同義語をグループにする)。
- grounded_ratio: 回答中の数値のうち、ツール結果に根拠がある割合(でっち上げた数値の検出)。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from health_insight_agent.tools.base import ToolExecution

# 「関係がない/差がない/判断できない」を表す言い回し(日本語・英語・ツールのコード名)
NEGATION_RE = (
    r"(関係|相関|影響|差|違い|傾向)[^。\n]{0,24}(見られ|ありません|ない|認められ|確認でき|小さ|わずか)"
    r"|明確な[^。\n]{0,12}(ない|見られ)|ほぼ(同じ|変わら)|同程度"
    r"|判断(でき|は難し)|断定(でき|は)|有意(な)?(差|関係)?[^。\n]{0,6}(ない|見られ|ありません)"
    r"|no[_ ](clear[_ ])?(relationship|difference|correlation|pattern)"
    r"|not (found|detected|observed)|no (significant|clear)|insufficient"
)
# 回答に「どのデータ・何件・わからないこと」が含まれるか(ルール4)
LIMITS_RE = (
    r"わから(ない|なかっ)|判断でき|断定でき|限界|制限|参考|注意|not (determine|clear)|cannot|can't"
)


def _flatten_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, str):
        out.extend(float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", obj))


_DATE_OR_TIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2})?)?"  # ISO日時
    r"|\d{1,2}月\d{1,2}日"  # 8月10日
    r"|\d{1,2}[:：]\d{2}"  # 12:30
    r"|\d{1,2}時(?!間)(?:\d{1,2}分)?"  # 22時 / 22時30分(「3時間」は時刻でなく量なので除外しない)
)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def tool_numbers(trace: Iterable[ToolExecution]) -> list[float]:
    """ツール結果に現れる全ての数値(根拠として許容する値)。分⇔時間の換算値も含める。"""
    nums: list[float] = []
    for t in trace:
        if t.ok:
            _flatten_numbers(t.content, nums)
    derived = [n / 60 for n in nums] + [n * 60 for n in nums] + [abs(n) for n in nums]
    return nums + derived


def grounded_ratio(answer: str, trace: Iterable[ToolExecution]) -> float | None:
    """回答中の「統計らしい数値」のうち根拠のある割合。数値が無ければNone(判定対象外)。

    日付・時刻、1桁の整数(「3か月」「2回」等)、注意書き中の数値は対象外にする。
    許容誤差は相対3%か絶対0.6(四捨五入のずれ)。
    """
    body = answer.split("※")[0]
    body = _DATE_OR_TIME_RE.sub(" ", body)
    candidates = []
    for m in _NUM_RE.finditer(body):
        text = m.group(0)
        value = float(text)
        if "." not in text and abs(value) < 10:
            continue
        candidates.append(value)
    if not candidates:
        return None
    allowed = tool_numbers(trace)
    hits = sum(1 for v in candidates if any(abs(v - a) <= max(0.6, abs(a) * 0.03) for a in allowed))
    return hits / len(candidates)


def _results(trace: Iterable[ToolExecution], name: str) -> list[dict[str, Any]]:
    return [t.content["data"] for t in trace if t.ok and t.name == name]


def relation_sign(trace: Iterable[ToolExecution], xs: Iterable[str], ys: Iterable[str]) -> str:
    """traceから「x(原因側)とy(結果側)の関係」を読む: negative / positive / none / unknown。

    correlate は xy どちら向きでも、compare_days は「xが条件」または「yの小さい群で比較」の
    形を認める。
    ツールが計算した interpretation を読むだけで、数値を再計算しない。
    """
    xs, ys = set(xs), set(ys)
    signs: list[str] = []
    for d in _results(trace, "correlate"):
        a, b = d["x_metric"], d["y_metric"]
        if (a in xs and b in ys) or (b in xs and a in ys):  # どちら向きでも相関は同じ
            interp = d["interpretation"]
            signs.append(
                "none"
                if interp == "no_clear_relationship"
                else "unknown"
                if interp == "insufficient_data"
                else "negative"
                if interp.endswith("negative")
                else "positive"
            )
    for d in _results(trace, "compare_days"):
        cond = d["condition"]
        metrics = d["metrics"]
        if cond["metric"] in xs:  # 例: 遅い食事==1 の日 vs それ以外 で睡眠を比較
            for y in ys & set(metrics):
                signs.append(_compare_sign(metrics[y]["interpretation"], flip=False))
        if cond["metric"] in ys and cond["op"] in ("<", "<="):  # 例: 睡眠が短い日 vs それ以外
            for x in xs & set(metrics):
                signs.append(_compare_sign(metrics[x]["interpretation"], flip=True))
        if cond["metric"] in ys and cond["op"] in (">", ">="):
            for x in xs & set(metrics):
                signs.append(_compare_sign(metrics[x]["interpretation"], flip=False))
    if not signs:
        return "unknown"
    for preferred in ("negative", "positive", "none"):
        if preferred in signs:
            return preferred
    return "unknown"


def _compare_sign(interp: str, *, flip: bool) -> str:
    if interp == "no_clear_difference":
        return "none"
    if interp == "insufficient_data":
        return "unknown"
    higher = interp == "a_higher"
    if flip:
        higher = not higher
    return "positive" if higher else "negative"


def tools_ok(trace: Iterable[ToolExecution], names: Iterable[str]) -> bool:
    wanted = set(names)
    return any(t.ok and t.name in wanted for t in trace)


def matches_all(text: str, patterns: Iterable[str]) -> list[str]:
    """マッチしなかった正規表現のリスト(空なら全てマッチ)。"""
    return [p for p in patterns if not re.search(p, text, flags=re.IGNORECASE | re.DOTALL)]


def matches_any(text: str, patterns: Iterable[str]) -> list[str]:
    """マッチしてしまった(禁止)正規表現のリスト(空なら禁止表現なし)。"""
    return [p for p in patterns if re.search(p, text, flags=re.IGNORECASE | re.DOTALL)]
