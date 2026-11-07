"""ORM ⇔ Domain Model の変換。ORMの型をRepositoryの外へ漏らさないための境界。"""

from __future__ import annotations

from uuid import uuid4

from health_insight_agent.database import schema as orm
from health_insight_agent.domain.enums import (
    ExerciseCategory,
    Locale,
    MealType,
    SleepQuality,
)
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


def user_to_orm(u: User) -> orm.UserORM:
    return orm.UserORM(
        id=u.user_id,
        email=u.email,
        display_name=u.display_name,
        created_at=u.created_at,
        locale=u.locale.value,
    )


def user_to_domain(o: orm.UserORM) -> User:
    return User(
        user_id=o.id,
        email=o.email,
        display_name=o.display_name,
        created_at=o.created_at,
        locale=Locale(o.locale),
    )


def sleep_to_orm(s: SleepLog) -> orm.SleepLogORM:
    return orm.SleepLogORM(
        id=s.sleep_log_id,
        user_id=s.user_id,
        bedtime=s.bedtime,
        wake_time=s.wake_time,
        quality=s.quality.value if s.quality else None,
    )


def sleep_to_domain(o: orm.SleepLogORM) -> SleepLog:
    return SleepLog(
        sleep_log_id=o.id,
        user_id=o.user_id,
        bedtime=o.bedtime,
        wake_time=o.wake_time,
        quality=SleepQuality(o.quality) if o.quality else None,
    )


def meal_to_orm(m: MealLog) -> orm.MealLogORM:
    return orm.MealLogORM(
        id=m.meal_log_id,
        user_id=m.user_id,
        consumed_at=m.consumed_at,
        meal_type=m.meal_type.value,
        calories=m.calories,
        protein_g=m.protein_g,
        fat_g=m.fat_g,
        carbohydrate_g=m.carbohydrate_g,
        memo=m.memo,
    )


def meal_to_domain(o: orm.MealLogORM) -> MealLog:
    return MealLog(
        meal_log_id=o.id,
        user_id=o.user_id,
        consumed_at=o.consumed_at,
        meal_type=MealType(o.meal_type),
        calories=o.calories,
        protein_g=o.protein_g,
        fat_g=o.fat_g,
        carbohydrate_g=o.carbohydrate_g,
        memo=o.memo,
    )


def workout_to_orm(w: WorkoutLog) -> orm.WorkoutLogORM:
    exercises = []
    for ex_i, ex in enumerate(w.exercises):
        sets = [
            orm.WorkoutSetORM(id=uuid4(), order_index=set_i, reps=s.reps, weight_kg=s.weight_kg)
            for set_i, s in enumerate(ex.sets)
        ]
        exercises.append(
            orm.WorkoutExerciseORM(
                id=uuid4(),
                order_index=ex_i,
                exercise_name=ex.exercise_name,
                category=ex.category.value,
                duration_minutes=ex.duration_minutes,
                sets=sets,
            )
        )
    return orm.WorkoutLogORM(
        id=w.workout_log_id,
        user_id=w.user_id,
        performed_at=w.performed_at,
        calories_burned=w.calories_burned,
        memo=w.memo,
        exercises=exercises,
    )


def workout_to_domain(o: orm.WorkoutLogORM) -> WorkoutLog:
    return WorkoutLog(
        workout_log_id=o.id,
        user_id=o.user_id,
        performed_at=o.performed_at,
        calories_burned=o.calories_burned,
        memo=o.memo,
        exercises=[
            WorkoutExercise(
                exercise_name=e.exercise_name,
                category=ExerciseCategory(e.category),
                duration_minutes=e.duration_minutes,
                sets=[WorkoutSet(reps=s.reps, weight_kg=s.weight_kg) for s in e.sets],
            )
            for e in o.exercises
        ],
    )


def health_to_orm(h: HealthLog) -> orm.HealthLogORM:
    return orm.HealthLogORM(
        id=h.health_log_id,
        user_id=h.user_id,
        recorded_at=h.recorded_at,
        condition_score=h.condition_score,
        morning_condition=h.morning_condition,
        evening_condition=h.evening_condition,
        mental_score=h.mental_score,
        fatigue_score=h.fatigue_score,
        stress_score=h.stress_score,
        symptoms=list(h.symptoms),
        memo=h.memo,
    )


def health_to_domain(o: orm.HealthLogORM) -> HealthLog:
    return HealthLog(
        health_log_id=o.id,
        user_id=o.user_id,
        recorded_at=o.recorded_at,
        condition_score=o.condition_score,
        morning_condition=o.morning_condition,
        evening_condition=o.evening_condition,
        mental_score=o.mental_score,
        fatigue_score=o.fatigue_score,
        stress_score=o.stress_score,
        symptoms=list(o.symptoms),
        memo=o.memo,
    )


def weight_to_orm(w: WeightLog) -> orm.WeightLogORM:
    return orm.WeightLogORM(
        id=w.weight_log_id, user_id=w.user_id, recorded_at=w.recorded_at, weight_kg=w.weight_kg
    )


def weight_to_domain(o: orm.WeightLogORM) -> WeightLog:
    return WeightLog(
        weight_log_id=o.id, user_id=o.user_id, recorded_at=o.recorded_at, weight_kg=o.weight_kg
    )


def journal_to_orm(j: DailyJournal) -> orm.DailyJournalORM:
    return orm.DailyJournalORM(
        id=j.daily_journal_id,
        user_id=j.user_id,
        journal_date=j.journal_date,
        diet_policy_note=j.diet_policy_note,
        wake_target_time=j.wake_target_time,
        sleep_target_time=j.sleep_target_time,
        shopping_memo=j.shopping_memo,
        food_expense_yen=j.food_expense_yen,
        reflection_memo=j.reflection_memo,
        improvement_point=j.improvement_point,
        cooking_shopping_note=j.cooking_shopping_note,
        study_prep_note=j.study_prep_note,
    )


def journal_to_domain(o: orm.DailyJournalORM) -> DailyJournal:
    return DailyJournal(
        daily_journal_id=o.id,
        user_id=o.user_id,
        journal_date=o.journal_date,
        diet_policy_note=o.diet_policy_note,
        wake_target_time=o.wake_target_time,
        sleep_target_time=o.sleep_target_time,
        shopping_memo=o.shopping_memo,
        food_expense_yen=o.food_expense_yen,
        reflection_memo=o.reflection_memo,
        improvement_point=o.improvement_point,
        cooking_shopping_note=o.cooking_shopping_note,
        study_prep_note=o.study_prep_note,
    )
