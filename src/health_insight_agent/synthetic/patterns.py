"""合成データに仕込む「正解」の因果パターン。

評価(Phase 8)ではこれが正解になる: エージェントが見つけられるか、仕込んでいない関係に
「ない」と答えられるか。生成側(generator.py)と評価側が同じ定数を参照し、ズレを防ぐ。

時間の対応: ある日Dの夕方〜夜の行動が、その夜の睡眠(起床日はD+1)に効く。
睡眠は「起床日」に属するので、factor_date=D の要因は wake_date=D+1 の睡眠と対応する。
"""

from __future__ import annotations

from dataclasses import dataclass

LATE_MEAL_GAP_HOURS = 3.0
"""就寝の何時間前までに最後の食事を終えていれば「遅い食事」でないか。"""
LATE_MEAL_SLEEP_PENALTY_MIN = 60
"""遅い食事の夜は、睡眠時間がこれだけ短くなる(平均)。"""
EVENING_WORKOUT_START_HOUR = 20
"""この時刻(JST)以降に始めた運動を「夜の運動」とみなす。"""
EVENING_HIGH_WORKOUT_BEDTIME_DELAY_MIN = 50
"""夜の高強度運動の日は、就寝がこれだけ遅れる(入眠の遅れの代理指標。決定1=案A)。"""
SHORT_SLEEP_FATIGUE_PER_HOUR = 1.0
"""睡眠が1時間短いと、翌日(起床日)の疲労スコアが約1上がる。"""


@dataclass(frozen=True, slots=True)
class CausalPattern:
    pattern_id: str
    factor: str
    outcome: str
    description: str


PATTERNS: tuple[CausalPattern, ...] = (
    CausalPattern(
        "late_meal_short_sleep",
        "最後の食事が就寝の3時間以内",
        "睡眠時間",
        f"就寝3時間以内に食べた夜は睡眠が約{LATE_MEAL_SLEEP_PENALTY_MIN}分短い",
    ),
    CausalPattern(
        "evening_high_workout_late_bedtime",
        "20時以降の高強度運動",
        "就寝時刻",
        f"夜に高強度運動をした日は就寝が約{EVENING_HIGH_WORKOUT_BEDTIME_DELAY_MIN}分遅い(睡眠時間は変わらない)",
    ),
    CausalPattern(
        "short_sleep_fatigue",
        "睡眠時間",
        "同日の疲労スコア",
        "睡眠が短いほど、その日の疲労スコアが高い",
    ),
)

NULL_PATTERNS: tuple[CausalPattern, ...] = (
    CausalPattern(
        "breakfast_calories_no_effect",
        "朝食カロリー",
        "睡眠時間",
        "朝食カロリーと睡眠時間に関係は仕込んでいない(「ない」と答えるべき)",
    ),
    CausalPattern(
        "weight_no_effect",
        "体重",
        "睡眠時間",
        "体重と睡眠時間に関係は仕込んでいない",
    ),
    CausalPattern(
        "weekend_no_effect",
        "週末かどうか",
        "睡眠時間",
        "平日と休日で睡眠時間の差は仕込んでいない",
    ),
)
