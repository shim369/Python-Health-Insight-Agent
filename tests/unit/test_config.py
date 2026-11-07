import pytest

from health_insight_agent.config import assert_test_database


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://u:p@localhost:5433/health_insight_test",
        "postgresql+psycopg://u:p@h/other_test?sslmode=require",
    ],
)
def test_test_database_names_are_accepted(url: str) -> None:
    assert_test_database(url)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://u:p@localhost:5433/health_insight",
        "postgresql+psycopg://u:p@ep-xxx.neon.tech/neondb?sslmode=require",
        "postgresql+psycopg://u:p@localhost:5433/",
    ],
)
def test_non_test_database_names_are_rejected(url: str) -> None:
    with pytest.raises(RuntimeError, match="_test"):
        assert_test_database(url)
