"""評価の実行:

    uv run python -m health_insight_agent.evaluation --config fake-all --config fake-typed-only
    uv run python -m health_insight_agent.evaluation --config gemini-flash-all   # 要APIキー
    uv run python -m health_insight_agent.evaluation --list

評価用の合成データ(固定ユーザー・固定シード)をDBに作り直してから、質問セットを実行する。
結果は evaluation/results/(git管理外)に保存し、--write-docs で表をdocs/に書き出せる。
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from health_insight_agent.config import PROJECT_ROOT, load_settings
from health_insight_agent.evaluation.configs import PRESETS, build_eval_agent
from health_insight_agent.evaluation.context import EvalContext, build_context, seed_eval_data
from health_insight_agent.evaluation.questions import get_questions
from health_insight_agent.evaluation.runner import (
    QuestionResult,
    Summary,
    render_markdown,
    run_questions,
    save_results,
    summarize,
)


def _preamble(ctx: EvalContext, configs: list[str], n_questions: int) -> str:
    cmd = "uv run python -m health_insight_agent.evaluation"
    flags = " ".join(f"--config {c}" for c in configs)
    ds = ctx.dataset
    return (
        "# 評価結果\n\n"
        f"- 生成日: {datetime.now():%Y-%m-%d}\n"
        f"- 再現コマンド: `{cmd} {flags} --write-docs docs/evaluation-results.md`\n"
        f"- データ: 合成データ(食事{len(ds.meal_logs)}件・睡眠{len(ds.sleep_logs)}件、固定シード)。"
        "実データは使っていません。\n"
        f"- 質問: {n_questions}問。採点は決定的(LLM採点なし): ツール選択・ツール結果の証拠・"
        "回答の正規表現・数値の根拠・禁止表現。\n"
        "- `fake-*` はキー不要のルールベース(LLM抜きのベースライン)。Geminiの結果は"
        "README「実キーで評価する」の手順で測定できます(この表には含まれません)。\n\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Health Insight Agent evaluation")
    parser.add_argument("--config", action="append", default=[], help="プリセット名(複数可)")
    parser.add_argument("--only", default="", help="質問IDをカンマ区切りで絞る(例: P1,N2)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    parser.add_argument("--write-docs", type=Path, default=None, help="表をMarkdownで書き出す先")
    parser.add_argument("--include-model", action="store_true", help="自作モデルの質問も含める")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="質問の間の待機秒数(無料枠のレート制限対策。例: 20)",
    )
    parser.add_argument("--repeat", type=int, default=3, help="モデル質問の繰り返し回数(一貫性)")
    parser.add_argument(
        "--min-rate",
        action="append",
        default=[],
        metavar="CONFIG=RATE",
        help="正答率がRATE未満なら終了コード1(CIの回帰ガード。例: fake-all=0.85)",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for c in PRESETS.values():
            print(f"{c.name:32s} {c.description}")
        return
    configs = args.config or ["fake-all"]
    unknown = [c for c in configs if c not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown config: {unknown}. --list で一覧を表示")

    settings = load_settings()
    ctx = seed_eval_data(settings.database_url, build_context())
    questions = get_questions(include_model=args.include_model)
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q.qid in wanted]

    summaries: list[Summary] = []
    by_config: dict[str, list[QuestionResult]] = {}
    for name in configs:
        cfg = PRESETS[name]
        if cfg.provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            print(f"[skip] {name}: GEMINI_API_KEY が未設定のため実行しません")
            continue
        print(f"[run] {name}: {len(questions)} questions")
        agent = build_eval_agent(settings, cfg, today=ctx.today)
        results = run_questions(agent, questions, ctx, repeat=args.repeat, pause_seconds=args.pause)
        by_config[name] = results
        summaries.append(summarize(name, results))
        s = summaries[-1]
        print(
            f"      {s.passed}/{s.n} passed ({s.rate:.0%}), avg {s.avg_tokens:.0f} tokens/question"
        )
    if not summaries:
        raise SystemExit("実行できる設定がありませんでした")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path, md_path = save_results(args.out, stamp, summaries, by_config)
    print(f"saved: {json_path} / {md_path}")
    failed: list[str] = []
    for spec in args.min_rate:
        name, _, rate = spec.partition("=")
        found = next((x for x in summaries if x.config == name), None)
        if found is not None and found.rate < float(rate):
            failed.append(f"{name}: {found.rate:.0%} < {float(rate):.0%}")
    if args.write_docs:
        preamble = _preamble(ctx, list(by_config), len(questions))
        args.write_docs.write_text(
            preamble + render_markdown(summaries, by_config), encoding="utf-8"
        )
        print(f"wrote: {args.write_docs}")
    if failed:
        raise SystemExit("評価の正答率が基準を下回りました: " + "; ".join(failed))


if __name__ == "__main__":
    main()
