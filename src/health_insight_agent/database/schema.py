"""SQLAlchemy ORMモデル。既存アプリ(database/schema.py)のスキーマと同形(統合時にそのまま使える)。

対象は本エージェントが読むテーブルのみ(users / health / meal / workout×3 / sleep /
weight / daily_journals)。Domain ModelとORMは分離し、変換はRepository内で行う。
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time

from sqlalchemy import DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserORM(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    display_name: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    locale: Mapped[str] = mapped_column(default="ja")


class HealthLogORM(Base):
    __tablename__ = "health_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    condition_score: Mapped[int | None]
    morning_condition: Mapped[int | None]
    evening_condition: Mapped[int | None]
    mental_score: Mapped[int | None]
    fatigue_score: Mapped[int | None]
    stress_score: Mapped[int | None]
    symptoms: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    memo: Mapped[str | None] = mapped_column(Text)


class MealLogORM(Base):
    __tablename__ = "meal_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    consumed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    meal_type: Mapped[str]
    calories: Mapped[float]
    protein_g: Mapped[float]
    fat_g: Mapped[float]
    carbohydrate_g: Mapped[float]
    memo: Mapped[str | None] = mapped_column(Text)


class WorkoutLogORM(Base):
    __tablename__ = "workout_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    performed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    calories_burned: Mapped[float | None]
    memo: Mapped[str | None] = mapped_column(Text)

    exercises: Mapped[list[WorkoutExerciseORM]] = relationship(
        back_populates="workout_log",
        cascade="all, delete-orphan",
        order_by="WorkoutExerciseORM.order_index",
    )


class WorkoutExerciseORM(Base):
    __tablename__ = "workout_exercises"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    workout_log_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workout_logs.id"), index=True)
    order_index: Mapped[int]
    exercise_name: Mapped[str]
    category: Mapped[str]
    duration_minutes: Mapped[int | None]

    workout_log: Mapped[WorkoutLogORM] = relationship(back_populates="exercises")
    sets: Mapped[list[WorkoutSetORM]] = relationship(
        back_populates="exercise",
        cascade="all, delete-orphan",
        order_by="WorkoutSetORM.order_index",
    )


class WorkoutSetORM(Base):
    __tablename__ = "workout_sets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    workout_exercise_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workout_exercises.id"), index=True
    )
    order_index: Mapped[int]
    reps: Mapped[int]
    weight_kg: Mapped[float | None]

    exercise: Mapped[WorkoutExerciseORM] = relationship(back_populates="sets")


class SleepLogORM(Base):
    __tablename__ = "sleep_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    bedtime: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    wake_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    quality: Mapped[str | None]


class WeightLogORM(Base):
    __tablename__ = "weight_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    weight_kg: Mapped[float]


class DailyJournalORM(Base):
    __tablename__ = "daily_journals"
    __table_args__ = (
        UniqueConstraint("user_id", "journal_date", name="uq_daily_journals_user_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    journal_date: Mapped[date]
    diet_policy_note: Mapped[str | None] = mapped_column(Text)
    wake_target_time: Mapped[time | None]
    sleep_target_time: Mapped[time | None]
    shopping_memo: Mapped[str | None] = mapped_column(Text)
    food_expense_yen: Mapped[int | None]
    reflection_memo: Mapped[str | None] = mapped_column(Text)
    improvement_point: Mapped[str | None] = mapped_column(Text)
    cooking_shopping_note: Mapped[str | None] = mapped_column(Text)
    study_prep_note: Mapped[str | None] = mapped_column(Text)
