"""合成データ生成(既存スキーマ準拠、既知の因果パターン入り)。

- 乱数はシード固定(同じseedなら同じデータ)。実データは一切使わない。
- 仕込みの定義は patterns.py。生成後に「その夜の正解ラベル」(NightTruth)も返し、
  テストと評価で使う。
- 欠損も少し入れる(睡眠3%・体調10%・体重10%の日が欠ける)。実データに近づけ、
  Phase 9の欠損率チェックを意味のあるものにするため。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from uuid import UUID

from health_insight_agent.domain.enums import (
    ExerciseCategory,
    Locale,
    MealType,
    SleepQuality,
    WorkoutIntensity,
)
from health_insight_agent.domain.intensity import workout_intensity
from health_insight_agent.domain.models import (
    DailyJournal,
    HealthLog,
    MealLog,
    SleepLog,
    User,
    WeightLog,
    WorkoutExercise,
    WorkoutLog,
    WorkoutSet,
)
from health_insight_agent.local_time import APP_TIMEZONE, local_datetime
from health_insight_agent.synthetic import patterns as p

_KCAL_BANDS = {
    WorkoutIntensity.LOW: (120.0, 190.0),
    WorkoutIntensity.MODERATE: (230.0, 380.0),
    WorkoutIntensity.HIGH: (430.0, 620.0),
}
"""強度ごとの消費kcal帯。domain/intensity.py のしきい値(200/400)に合わせてある。"""

_STRENGTH_MOVES = ("レッグプレス", "チェストプレス", "ラットプルダウン", "ショルダープレス")

_GENERIC_REFLECTIONS = (
    "今日は仕事が忙しく、あっという間に終わった。",
    "昼食後に少し散歩できて気分転換になった。",
    "夕方に集中できて、作業がはかどった。",
    "友人と食事に行って楽しかった。",
    "特に変わったことはなく、穏やかな一日だった。",
)
_IMPROVEMENTS = (
    "明日は早めに作業を切り上げる。",
    "水分をもう少しこまめにとる。",
    "昼休みに少し歩く。",
    "夕食の時間を意識する。",
)
_LATE_MEAL_NOTES = (
    "夜遅くに食べてしまった。寝つきが悪くなりそう。",
    "夕食が遅くなり、お腹が重いまま布団に入った。",
)
_SHORT_SLEEP_NOTES = (
    "昨夜は睡眠が短く、日中ずっと眠かった。",
    "寝不足で集中力が続かなかった。",
)
_NIGHT_WORKOUT_NOTES = (
    "夜に激しく運動したら、目が冴えてしまった。",
    "遅い時間のトレーニングで体が火照っている。",
)


@dataclass(frozen=True, slots=True)
class NightTruth:
    """ある夜の「仕込みの正解」ラベル。実際に生成した値から計算する。"""

    factor_date: date
    wake_date: date
    bedtime: datetime
    last_meal_at: datetime
    late_meal: bool
    evening_high_workout: bool


@dataclass(kw_only=True)
class SyntheticDataset:
    user: User
    sleep_logs: list[SleepLog] = field(default_factory=list)
    meal_logs: list[MealLog] = field(default_factory=list)
    workout_logs: list[WorkoutLog] = field(default_factory=list)
    health_logs: list[HealthLog] = field(default_factory=list)
    weight_logs: list[WeightLog] = field(default_factory=list)
    daily_journals: list[DailyJournal] = field(default_factory=list)
    truths: list[NightTruth] = field(default_factory=list)
    """全夜分の正解(睡眠ログが欠損した夜も含む)。"""


def _uuid(ids: random.Random) -> UUID:
    """行IDの生成。IDはuser_id由来の別の乱数から作る。データの値はseedだけで決まるので、
    同じseedなら別ユーザーでも同じ履歴になり、かつ主キー(全ユーザー共通)は衝突しない。"""
    return UUID(int=ids.getrandbits(128), version=4)


def _clamp(value: float, low: int, high: int) -> int:
    return max(low, min(high, round(value)))


def _at(d: date, hour: float, rng: random.Random, jitter_min: float = 0.0) -> datetime:
    """JSTのd日 hour時(小数可)を基準に、±jitterのガウスノイズを加えた時刻。"""
    minutes = hour * 60 + rng.gauss(0, jitter_min)
    return local_datetime(d, time.min) + timedelta(minutes=minutes)


def _meal(
    rng: random.Random,
    ids: random.Random,
    user_id: UUID,
    when: datetime,
    meal_type: MealType,
    kcal: float,
) -> MealLog:
    kcal = max(80.0, kcal)
    return MealLog(
        meal_log_id=_uuid(ids),
        user_id=user_id,
        consumed_at=when,
        meal_type=meal_type,
        calories=round(kcal),
        protein_g=round(kcal * 0.15 / 4 * rng.uniform(0.9, 1.1), 1),
        fat_g=round(kcal * 0.30 / 9 * rng.uniform(0.9, 1.1), 1),
        carbohydrate_g=round(kcal * 0.55 / 4 * rng.uniform(0.9, 1.1), 1),
    )


def _workout(
    rng: random.Random,
    ids: random.Random,
    user_id: UUID,
    when: datetime,
    intensity: WorkoutIntensity,
) -> WorkoutLog:
    low, high = _KCAL_BANDS[intensity]
    kcal = round(rng.uniform(low, high))
    kind = rng.choice(("cardio", "strength", "both"))
    exercises: list[WorkoutExercise] = []
    if kind in ("cardio", "both"):
        minutes = round(kcal / 7) if kind == "cardio" else 10
        exercises.append(
            WorkoutExercise(
                exercise_name="有酸素運動",
                category=ExerciseCategory.CARDIO,
                duration_minutes=minutes,
            )
        )
    if kind in ("strength", "both"):
        for move in rng.sample(_STRENGTH_MOVES, 3):
            base = rng.choice((20.0, 30.0, 40.0, 60.0))
            exercises.append(
                WorkoutExercise(
                    exercise_name=move,
                    category=ExerciseCategory.STRENGTH,
                    # セットごとに重さが変わるケース(既存アプリの3階層設計)を再現する。
                    sets=[
                        WorkoutSet(reps=10, weight_kg=base),
                        WorkoutSet(reps=10, weight_kg=base - rng.choice((0.0, 5.0))),
                        WorkoutSet(reps=8, weight_kg=base - rng.choice((0.0, 5.0, 10.0))),
                    ],
                )
            )
    workout = WorkoutLog(
        workout_log_id=_uuid(ids),
        user_id=user_id,
        performed_at=when,
        exercises=exercises,
        calories_burned=float(kcal),
    )
    # 生成した強度が狙い通りか(派生指標と整合しているか)を常に検証する。
    assert workout_intensity(workout) == intensity
    return workout


def _quality(rng: random.Random, duration_min: int) -> SleepQuality:
    order = [SleepQuality.POOR, SleepQuality.FAIR, SleepQuality.GOOD, SleepQuality.EXCELLENT]
    idx = 0 if duration_min < 360 else 1 if duration_min < 420 else 2 if duration_min < 480 else 3
    if rng.random() < 0.15:  # 主観評価のぶれ
        idx = max(0, min(3, idx + rng.choice((-1, 1))))
    return order[idx]


def generate_dataset(
    user_id: UUID,
    *,
    end_date: date,
    days: int = 90,
    seed: int = 42,
    email: str = "demo@example.com",
) -> SyntheticDataset:
    """`end_date`を最終日とする`days`日分(睡眠は最初の1日分を除く)を生成する。

    最初の日の前夜の食事・運動は記録されていないため、その夜(最初の起床日)の睡眠は作らない
    (要因が見えない睡眠ログを作ると、パターン検証が不公平になるため)。
    """
    rng = random.Random(seed)
    ids = random.Random(f"ids:{user_id}")
    start = end_date - timedelta(days=days - 1)
    ds = SyntheticDataset(
        user=User(
            user_id=user_id,
            email=email,  # 既定は架空のアドレス(実在の個人情報ではない)
            display_name="Demo User",
            created_at=local_datetime(start, time.min),
            locale=Locale.JA,
        )
    )
    prev_duration = 420.0
    weight = 68.0

    for offset in range(days):
        day = start + timedelta(days=offset)
        weekend = day.weekday() >= 5

        # --- 食事(朝・昼は固定的、夕食は30%が遅い。夜食が少しある) ---
        meals = [
            _meal(
                rng, ids, user_id, _at(day, 7.5, rng, 25), MealType.BREAKFAST, rng.gauss(420, 80)
            ),
            _meal(rng, ids, user_id, _at(day, 12.5, rng, 25), MealType.LUNCH, rng.gauss(700, 110)),
        ]
        dinner_hour = rng.uniform(21.25, 22.75) if rng.random() < 0.30 else rng.uniform(18.5, 20.0)
        meals.append(
            _meal(
                rng, ids, user_id, _at(day, dinner_hour, rng), MealType.DINNER, rng.gauss(720, 140)
            )
        )
        if rng.random() < 0.25:
            meals.append(
                _meal(
                    rng, ids, user_id, _at(day, 15.0, rng, 20), MealType.SNACK, rng.gauss(160, 40)
                )
            )
        if rng.random() < 0.08:
            meals.append(
                _meal(
                    rng, ids, user_id, _at(day, rng.uniform(22.5, 23.2), rng), MealType.SNACK, 180
                )
            )
        ds.meal_logs.extend(meals)

        # --- 運動(夜枠は高強度になりやすい) ---
        evening_high = False
        if rng.random() < (0.65 if weekend else 0.55):
            slot = rng.choices(("morning", "evening", "night"), weights=(35, 35, 30))[0]
            if slot == "morning":
                when, weights = _at(day, 7.0, rng, 15), (30, 50, 20)
            elif slot == "evening":
                when, weights = _at(day, rng.uniform(18.0, 19.5), rng), (30, 50, 20)
            else:
                when, weights = _at(day, rng.uniform(20.25, 21.75), rng), (0, 30, 70)
            intensity = rng.choices(
                (WorkoutIntensity.LOW, WorkoutIntensity.MODERATE, WorkoutIntensity.HIGH),
                weights=weights,
            )[0]
            ds.workout_logs.append(_workout(rng, ids, user_id, when, intensity))
            evening_high = (
                slot == "night" and intensity == WorkoutIntensity.HIGH
            )  # 20時以降の高強度

        # --- 体重(睡眠とは無関係なランダムウォーク。「関係なし」の対照) ---
        weight += rng.gauss(0, 0.15)
        if rng.random() < 0.9:
            ds.weight_logs.append(
                WeightLog(
                    weight_log_id=_uuid(ids),
                    user_id=user_id,
                    recorded_at=_at(day, 7.2, rng, 5),
                    weight_kg=round(weight, 1),
                )
            )

        # --- その夜の睡眠(起床日 = day+1)。最終日の夜は未来なので作らない ---
        if day < end_date:
            last_meal = max(m.consumed_at for m in meals)
            bedtime = _at(day, 23.25, rng, 30)
            if evening_high:
                bedtime += timedelta(minutes=p.EVENING_HIGH_WORKOUT_BEDTIME_DELAY_MIN)
            late_meal = (bedtime - last_meal) < timedelta(hours=p.LATE_MEAL_GAP_HOURS)
            duration = 420 + rng.gauss(0, 25) - (p.LATE_MEAL_SLEEP_PENALTY_MIN if late_meal else 0)
            duration = max(240.0, min(600.0, duration))
            ds.truths.append(
                NightTruth(
                    factor_date=day,
                    wake_date=day + timedelta(days=1),
                    bedtime=bedtime,
                    last_meal_at=last_meal,
                    late_meal=late_meal,
                    evening_high_workout=evening_high,
                )
            )
            if rng.random() > 0.03:
                ds.sleep_logs.append(
                    SleepLog(
                        sleep_log_id=_uuid(ids),
                        user_id=user_id,
                        bedtime=bedtime,
                        wake_time=bedtime + timedelta(minutes=round(duration)),
                        quality=_quality(rng, round(duration)),
                    )
                )
            next_duration = duration
        else:
            late_meal = False
            next_duration = prev_duration

        # --- 体調(前夜の睡眠が短いほど疲労が高い) ---
        if rng.random() < 0.90:
            short_hours = (420 - prev_duration) / 60
            fatigue = _clamp(
                3 + p.SHORT_SLEEP_FATIGUE_PER_HOUR * short_hours + rng.gauss(0, 0.6), 1, 5
            )
            morning = _clamp(3.5 - 0.6 * short_hours + rng.gauss(0, 0.7), 1, 5)
            evening = _clamp(3.2 + rng.gauss(0, 0.8), 1, 5)
            mental = _clamp(3.3 + rng.gauss(0, 0.8), 1, 5)
            stress = _clamp((3.0 if weekend else 3.4) + rng.gauss(0, 0.8), 1, 5)
            ds.health_logs.append(
                HealthLog(
                    health_log_id=_uuid(ids),
                    user_id=user_id,
                    recorded_at=_at(day, 21.0, rng, 20),
                    condition_score=_clamp((morning + evening + mental + (6 - fatigue)) / 4, 1, 5),
                    morning_condition=morning,
                    evening_condition=evening,
                    mental_score=mental,
                    fatigue_score=fatigue,
                    stress_score=stress,
                    symptoms=["眠気"] if fatigue >= 4 else [],
                )
            )

        # --- 振り返りメモ(RAG用)。一部は仕込みと対応し、残りは雑多な内容 ---
        if rng.random() < 0.6:
            sentences = [rng.choice(_GENERIC_REFLECTIONS)]
            if late_meal and rng.random() < 0.7:
                sentences.append(rng.choice(_LATE_MEAL_NOTES))
            if evening_high and rng.random() < 0.6:
                sentences.append(rng.choice(_NIGHT_WORKOUT_NOTES))
            if prev_duration < 390 and rng.random() < 0.6:
                sentences.append(rng.choice(_SHORT_SLEEP_NOTES))
            ds.daily_journals.append(
                DailyJournal(
                    daily_journal_id=_uuid(ids),
                    user_id=user_id,
                    journal_date=day,
                    wake_target_time=time(6, 30),
                    sleep_target_time=time(23, 0),
                    reflection_memo="".join(sentences),
                    improvement_point=rng.choice(_IMPROVEMENTS),
                )
            )

        prev_duration = next_duration

    return ds


def jst_wake_date(log: SleepLog) -> date:
    """睡眠ログの所属日(JSTの起床日)。テスト・評価で正解ラベルと突合するための補助。"""
    return log.wake_time.astimezone(APP_TIMEZONE).date()
