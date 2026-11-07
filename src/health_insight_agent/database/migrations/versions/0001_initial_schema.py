"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-11-07 23:58:38.953741
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("locale", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "daily_journals",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("journal_date", sa.Date(), nullable=False),
        sa.Column("diet_policy_note", sa.Text(), nullable=True),
        sa.Column("wake_target_time", sa.Time(), nullable=True),
        sa.Column("sleep_target_time", sa.Time(), nullable=True),
        sa.Column("shopping_memo", sa.Text(), nullable=True),
        sa.Column("food_expense_yen", sa.Integer(), nullable=True),
        sa.Column("reflection_memo", sa.Text(), nullable=True),
        sa.Column("improvement_point", sa.Text(), nullable=True),
        sa.Column("cooking_shopping_note", sa.Text(), nullable=True),
        sa.Column("study_prep_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "journal_date", name="uq_daily_journals_user_date"),
    )
    op.create_index(op.f("ix_daily_journals_user_id"), "daily_journals", ["user_id"], unique=False)
    op.create_table(
        "health_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("condition_score", sa.Integer(), nullable=True),
        sa.Column("morning_condition", sa.Integer(), nullable=True),
        sa.Column("evening_condition", sa.Integer(), nullable=True),
        sa.Column("mental_score", sa.Integer(), nullable=True),
        sa.Column("fatigue_score", sa.Integer(), nullable=True),
        sa.Column("stress_score", sa.Integer(), nullable=True),
        sa.Column("symptoms", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("memo", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_health_logs_user_id"), "health_logs", ["user_id"], unique=False)
    op.create_table(
        "meal_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("meal_type", sa.String(), nullable=False),
        sa.Column("calories", sa.Double(), nullable=False),
        sa.Column("protein_g", sa.Double(), nullable=False),
        sa.Column("fat_g", sa.Double(), nullable=False),
        sa.Column("carbohydrate_g", sa.Double(), nullable=False),
        sa.Column("memo", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_meal_logs_user_id"), "meal_logs", ["user_id"], unique=False)
    op.create_table(
        "sleep_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("bedtime", sa.DateTime(timezone=True), nullable=False),
        sa.Column("wake_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quality", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_sleep_logs_user_id"), "sleep_logs", ["user_id"], unique=False)
    op.create_table(
        "weight_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("weight_kg", sa.Double(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_weight_logs_user_id"), "weight_logs", ["user_id"], unique=False)
    op.create_table(
        "workout_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("performed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("calories_burned", sa.Double(), nullable=True),
        sa.Column("memo", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_workout_logs_user_id"), "workout_logs", ["user_id"], unique=False)
    op.create_table(
        "workout_exercises",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("workout_log_id", sa.UUID(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("exercise_name", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("duration_minutes", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["workout_log_id"],
            ["workout_logs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_workout_exercises_workout_log_id"),
        "workout_exercises",
        ["workout_log_id"],
        unique=False,
    )
    op.create_table(
        "workout_sets",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("workout_exercise_id", sa.UUID(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("reps", sa.Integer(), nullable=False),
        sa.Column("weight_kg", sa.Double(), nullable=True),
        sa.ForeignKeyConstraint(
            ["workout_exercise_id"],
            ["workout_exercises.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_workout_sets_workout_exercise_id"),
        "workout_sets",
        ["workout_exercise_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_workout_sets_workout_exercise_id"), table_name="workout_sets")
    op.drop_table("workout_sets")
    op.drop_index(op.f("ix_workout_exercises_workout_log_id"), table_name="workout_exercises")
    op.drop_table("workout_exercises")
    op.drop_index(op.f("ix_workout_logs_user_id"), table_name="workout_logs")
    op.drop_table("workout_logs")
    op.drop_index(op.f("ix_weight_logs_user_id"), table_name="weight_logs")
    op.drop_table("weight_logs")
    op.drop_index(op.f("ix_sleep_logs_user_id"), table_name="sleep_logs")
    op.drop_table("sleep_logs")
    op.drop_index(op.f("ix_meal_logs_user_id"), table_name="meal_logs")
    op.drop_table("meal_logs")
    op.drop_index(op.f("ix_health_logs_user_id"), table_name="health_logs")
    op.drop_table("health_logs")
    op.drop_index(op.f("ix_daily_journals_user_id"), table_name="daily_journals")
    op.drop_table("daily_journals")
    op.drop_table("users")
