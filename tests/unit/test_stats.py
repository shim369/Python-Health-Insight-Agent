import random

from health_insight_agent.services import stats


def test_summarize_empty_and_single() -> None:
    assert stats.summarize([]).n == 0
    s = stats.summarize([5.0])
    assert (s.mean, s.std) == (5.0, 0.0)


def test_compare_groups_detects_clear_difference() -> None:
    rng = random.Random(1)
    a = [rng.gauss(360, 20) for _ in range(20)]
    b = [rng.gauss(420, 20) for _ in range(40)]
    c = stats.compare_groups(a, b)
    assert c.interpretation == "a_lower" and c.reliability == "ok"
    assert c.difference is not None and -80 < c.difference < -40


def test_compare_groups_no_difference_for_same_distribution() -> None:
    rng = random.Random(2)
    a = [rng.gauss(400, 30) for _ in range(30)]
    b = [rng.gauss(400, 30) for _ in range(30)]
    assert stats.compare_groups(a, b).interpretation == "no_clear_difference"


def test_small_groups_are_insufficient_not_a_conclusion() -> None:
    c = stats.compare_groups([1.0, 2.0, 3.0], [10.0] * 30)
    assert c.interpretation == "insufficient_data" and c.p_value is None


def test_correlation_thresholds() -> None:
    xs = [float(i) for i in range(30)]
    strong = stats.correlate_pairs(xs, [2 * x + random.Random(3).gauss(0, 1) for x in xs])
    assert strong.interpretation == "strong_positive" and strong.reliability == "ok"
    rng = random.Random(4)
    noise = stats.correlate_pairs(xs, [rng.gauss(0, 1) for _ in xs])
    assert noise.interpretation == "no_clear_relationship"
    assert stats.correlate_pairs(xs[:5], xs[:5]).interpretation == "insufficient_data"
    assert stats.correlate_pairs(xs[:15], [2.0 * x for x in xs[:15]]).reliability == "low"


def test_constant_series_has_no_relationship() -> None:
    assert stats.correlate_pairs([1.0] * 20, [float(i) for i in range(20)]).interpretation == (
        "no_clear_relationship"
    )
