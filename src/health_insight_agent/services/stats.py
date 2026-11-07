"""統計の純粋関数。数値・判定はここ(コード)で出し、LLMには解釈させない。

件数が少ないときは結論を弱める(reliability)。閾値は目安で、医学的な根拠はない。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from scipy import stats as sp

MIN_GROUP_N = 5
"""比較する各グループの最低件数。これ未満は「判断できない」とする。"""
MIN_PAIRS_N = 10
"""相関を評価する最低ペア数。これ未満は「判断できない」とする。"""
LOW_PAIRS_N = 20
"""この件数未満は信頼度low(結論を弱めて伝える)。"""
ALPHA = 0.05


@dataclass(frozen=True, slots=True)
class GroupStats:
    n: int
    mean: float | None
    std: float | None
    median: float | None


def summarize(values: list[float]) -> GroupStats:
    if not values:
        return GroupStats(0, None, None, None)
    return GroupStats(
        n=len(values),
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if len(values) >= 2 else 0.0,
        median=statistics.median(values),
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    group_a: GroupStats
    group_b: GroupStats
    difference: float | None
    """mean(a) - mean(b)。"""
    cohens_d: float | None
    p_value: float | None
    reliability: str  # insufficient | low | ok
    interpretation: str  # insufficient_data | no_clear_difference | a_higher | a_lower


def compare_groups(a: list[float], b: list[float]) -> Comparison:
    ga, gb = summarize(a), summarize(b)
    if ga.n < MIN_GROUP_N or gb.n < MIN_GROUP_N:
        diff = (ga.mean - gb.mean) if ga.mean is not None and gb.mean is not None else None
        return Comparison(ga, gb, diff, None, None, "insufficient", "insufficient_data")
    assert ga.mean is not None and gb.mean is not None
    assert ga.std is not None and gb.std is not None
    diff = ga.mean - gb.mean
    pooled = math.sqrt(((ga.n - 1) * ga.std**2 + (gb.n - 1) * gb.std**2) / (ga.n + gb.n - 2))
    d = diff / pooled if pooled > 0 else None
    if ga.std == 0 and gb.std == 0:
        p = 0.0 if diff != 0 else 1.0
    else:
        p = float(sp.ttest_ind(a, b, equal_var=False).pvalue)  # Welchのt検定
    reliability = "low" if min(ga.n, gb.n) < 10 else "ok"
    if p >= ALPHA:
        interp = "no_clear_difference"
    else:
        interp = "a_higher" if diff > 0 else "a_lower"
    return Comparison(ga, gb, diff, d, p, reliability, interp)


@dataclass(frozen=True, slots=True)
class Correlation:
    n: int
    pearson_r: float | None
    spearman_rho: float | None
    p_value: float | None
    reliability: str  # insufficient | low | ok
    strength: str  # none | weak | moderate | strong
    interpretation: str
    """insufficient_data | no_clear_relationship | {weak,moderate,strong}_{positive,negative}"""


def _strength(r: float) -> str:
    a = abs(r)
    return "none" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else "strong"


def correlate_pairs(xs: list[float], ys: list[float]) -> Correlation:
    n = len(xs)
    if n < MIN_PAIRS_N:
        return Correlation(n, None, None, None, "insufficient", "none", "insufficient_data")
    if len(set(xs)) < 2 or len(set(ys)) < 2:  # 定数列では相関が定義できない
        return Correlation(n, None, None, None, "ok", "none", "no_clear_relationship")
    pearson = sp.pearsonr(xs, ys)
    spearman = sp.spearmanr(xs, ys)
    r, p = float(pearson.statistic), float(pearson.pvalue)
    rho = float(spearman.statistic)
    strength = _strength(r)
    if p >= ALPHA or strength == "none":
        interp = "no_clear_relationship"
    else:
        interp = f"{strength}_{'positive' if r > 0 else 'negative'}"
    return Correlation(n, r, rho, p, "low" if n < LOW_PAIRS_N else "ok", strength, interp)
