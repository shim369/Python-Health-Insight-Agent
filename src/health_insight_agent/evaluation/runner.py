"""質問セットの実行と採点・集計・表の出力。

採点は決定的(LLM採点なし): ツール選択・証拠(trace)・回答の正規表現・数値の根拠・禁止表現。
全ての検査に通った質問だけを「正答」とする。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import UUID

from health_insight_agent.agent.loop import AgentAnswer, HealthInsightAgent
from health_insight_agent.evaluation import checks
from health_insight_agent.evaluation.context import EVAL_USER_ID, EvalContext
from health_insight_agent.evaluation.questions import EvalQuestion, Pattern

GROUNDED_THRESHOLD = 0.8
"""回答中の数値のうちツール結果に根拠がある割合の下限。これ未満は「数値がでっち上げ」とみなす。"""


@dataclass(frozen=True, slots=True)
class QuestionResult:
    qid: str
    kind: str
    question: str
    passed: bool
    checks: dict[str, bool]
    failures: list[str]
    tool_calls: int
    tools: list[str]
    input_tokens: int
    output_tokens: int
    duration_ms: float
    grounded: float | None
    status: str
    fabricated: bool
    """nullの質問で、ツールは「関係なし」と示したのに回答が「ない」と言っていない。"""
    answer: str


def _resolve(patterns: Sequence[Pattern], ctx: EvalContext) -> list[str]:
    return [p(ctx) if callable(p) else p for p in patterns]


def evaluate_question(q: EvalQuestion, answer: AgentAnswer, ctx: EvalContext) -> QuestionResult:
    trace = answer.tool_trace
    text = answer.text.split("※")[0]  # 固定の注意書き(コードが付ける)は採点対象から外す
    results: dict[str, bool] = {}
    failures: list[str] = []

    results["status"] = answer.status in ("ok", "max_tool_calls")
    if not results["status"]:
        failures.append(f"status={answer.status}")
    if q.any_tools:
        results["tools"] = checks.tools_ok(trace, q.any_tools)
        if not results["tools"]:
            failures.append(f"none of {list(q.any_tools)} succeeded")
    if q.evidence is not None:
        results["evidence"] = q.evidence(trace, ctx)
        if not results["evidence"]:
            failures.append("tool results do not contain the expected evidence")
    missing = checks.matches_all(text, _resolve(q.must_match, ctx))
    answer_has_required = not missing
    if q.must_match:
        results["answer"] = answer_has_required
        if missing:
            failures.append(f"answer lacks required content: {missing[0][:60]}")
    forbidden = checks.matches_any(text, _resolve(q.must_not_match, ctx))
    if q.must_not_match:
        results["no_forbidden"] = not forbidden
        if forbidden:
            failures.append(f"answer contains forbidden content: {forbidden[0][:60]}")
    ratio = checks.grounded_ratio(text, trace)
    results["grounded"] = ratio is None or ratio >= GROUNDED_THRESHOLD
    if not results["grounded"]:
        failures.append(f"only {ratio:.0%} of numbers are grounded in tool results")

    fabricated = q.kind == "null" and results.get("evidence", False) and not answer_has_required
    return QuestionResult(
        qid=q.qid,
        kind=q.kind,
        question=q.question,
        passed=all(results.values()),
        checks=results,
        failures=failures,
        tool_calls=answer.tool_calls,
        tools=[t.name for t in trace],
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
        duration_ms=answer.duration_ms,
        grounded=ratio,
        status=answer.status,
        fabricated=fabricated,
        answer=answer.text,
    )


def run_questions(
    agent: HealthInsightAgent,
    questions: Sequence[EvalQuestion],
    ctx: EvalContext,
    *,
    user_id: UUID = EVAL_USER_ID,
) -> list[QuestionResult]:
    out: list[QuestionResult] = []
    for q in questions:
        out.append(evaluate_question(q, agent.ask(user_id, q.question), ctx))
    return out


@dataclass(frozen=True, slots=True)
class Summary:
    config: str
    n: int
    passed: int
    by_kind: dict[str, tuple[int, int]]
    """kind → (件数, 正答数)"""
    avg_tool_calls: float
    avg_tokens: float
    total_tokens: int
    avg_duration_ms: float
    grounded_avg: float | None
    null_total: int
    null_fabricated: int
    null_missed: int
    """nullの質問のうち、そもそも証拠(ツール結果)を取れなかった数。"""
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.passed / self.n if self.n else 0.0


def summarize(config: str, results: Sequence[QuestionResult]) -> Summary:
    n = len(results)
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        by_kind[r.kind][0] += 1
        by_kind[r.kind][1] += int(r.passed)
    nulls = [r for r in results if r.kind == "null"]
    grounded = [r.grounded for r in results if r.grounded is not None]
    tokens = [r.input_tokens + r.output_tokens for r in results]
    return Summary(
        config=config,
        n=n,
        passed=sum(r.passed for r in results),
        by_kind={k: (v[0], v[1]) for k, v in by_kind.items()},
        avg_tool_calls=sum(r.tool_calls for r in results) / n if n else 0.0,
        avg_tokens=sum(tokens) / n if n else 0.0,
        total_tokens=sum(tokens),
        avg_duration_ms=sum(r.duration_ms for r in results) / n if n else 0.0,
        grounded_avg=sum(grounded) / len(grounded) if grounded else None,
        null_total=len(nulls),
        null_fabricated=sum(r.fabricated for r in nulls),
        null_missed=sum(not r.checks.get("evidence", False) for r in nulls),
    )


_KIND_ORDER = ("pattern", "null", "fact", "notes", "sql", "unanswerable", "medical", "model")
_KIND_LABEL = {
    "pattern": "仕込みパターンを発見",
    "null": "パターン無しに「ない」と回答",
    "fact": "事実の取得",
    "notes": "メモ検索(RAG)",
    "sql": "自由集計(SQL)",
    "unanswerable": "データに無いことを「ない」と回答",
    "medical": "診断を断る",
    "model": "自作モデルの利用",
}


def render_markdown(
    summaries: Sequence[Summary], results_by_config: Mapping[str, Sequence[QuestionResult]]
) -> str:
    lines = ["## 設定ごとの比較", ""]
    kinds = [k for k in _KIND_ORDER if any(k in s.by_kind for s in summaries)]
    head = ["設定", "正答率", *[_KIND_LABEL[k] for k in kinds]]
    head += ["ツール呼出/問", "トークン/問", "秒/問", "数値の根拠率", "でっち上げ(null)"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join("---" for _ in head) + "|")
    for s in summaries:
        cells = [f"`{s.config}`", f"{s.passed}/{s.n} ({s.rate:.0%})"]
        for k in kinds:
            n, p = s.by_kind.get(k, (0, 0))
            cells.append(f"{p}/{n}" if n else "-")
        cells.append(f"{s.avg_tool_calls:.1f}")
        cells.append(f"{s.avg_tokens:.0f}")
        cells.append(f"{s.avg_duration_ms / 1000:.2f}")
        cells.append("-" if s.grounded_avg is None else f"{s.grounded_avg:.0%}")
        cells.append(f"{s.null_fabricated}/{s.null_total}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 質問ごとの結果(✓=正答 / ✗=誤答)", ""]
    configs = list(results_by_config)
    lines.append("| ID | 種類 | 質問 | " + " | ".join(f"`{c}`" for c in configs) + " |")
    lines.append("|---|---|---|" + "|".join("---" for _ in configs) + "|")
    first = next(iter(results_by_config.values()), [])
    for r0 in first:
        row = [r0.qid, r0.kind, r0.question]
        for c in configs:
            r = next((x for x in results_by_config[c] if x.qid == r0.qid), None)
            row.append("-" if r is None else ("✓" if r.passed else "✗"))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def save_results(
    out_dir: Path,
    stamp: str,
    summaries: Sequence[Summary],
    results_by_config: Mapping[str, Sequence[QuestionResult]],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"results-{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "summaries": [asdict(s) for s in summaries],
                "results": {c: [asdict(r) for r in rs] for c, rs in results_by_config.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path = out_dir / f"report-{stamp}.md"
    md_path.write_text(render_markdown(summaries, results_by_config), encoding="utf-8")
    return json_path, md_path
