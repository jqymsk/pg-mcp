"""Security tests for blocked table and column access.

Tests verify that the SQLValidator correctly prevents access to sensitive
tables and columns, including case-insensitive matching, JOINs, and subqueries.
"""

import pytest

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.services.sql_validator import SQLValidator


@pytest.fixture
def validator_with_blocked_tables() -> SQLValidator:
    """Create a validator with blocked tables configured."""
    return SQLValidator(
        config=SecurityConfig(),
        blocked_tables=["secrets", "credentials", "api_keys"],
    )


@pytest.fixture
def validator_with_blocked_columns() -> SQLValidator:
    """Create a validator with blocked columns configured."""
    return SQLValidator(
        config=SecurityConfig(),
        blocked_columns=["password", "ssn", "credit_card"],
    )


class TestBlockedTableAccess:
    """Test that access to blocked tables is prevented."""

    def test_blocked_table_direct_access(
        self, validator_with_blocked_tables: SQLValidator
    ) -> None:
        """Direct SELECT from a blocked table should be rejected."""
        sql = "SELECT * FROM secrets"
        with pytest.raises(SecurityViolationError) as exc_info:
            validator_with_blocked_tables.validate_or_raise(sql)
        assert "secrets" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM SECRETS",
            "SELECT * FROM Secrets",
        ],
        ids=["uppercase", "mixed_case"],
    )
    def test_blocked_table_case_insensitive(
        self, validator_with_blocked_tables: SQLValidator, sql: str
    ) -> None:
        """Table blocking should be case-insensitive."""
        with pytest.raises(SecurityViolationError) as exc_info:
            validator_with_blocked_tables.validate_or_raise(sql)
        assert "secrets" in str(exc_info.value).lower()

    def test_blocked_table_in_join(
        self, validator_with_blocked_tables: SQLValidator
    ) -> None:
        """Blocked table referenced in a JOIN should be rejected."""
        sql = "SELECT u.* FROM users u JOIN secrets s ON u.id = s.user_id"
        with pytest.raises(SecurityViolationError) as exc_info:
            validator_with_blocked_tables.validate_or_raise(sql)
        assert "secrets" in str(exc_info.value).lower()

    def test_blocked_table_in_subquery(
        self, validator_with_blocked_tables: SQLValidator
    ) -> None:
        """Blocked table referenced in a subquery should be rejected."""
        sql = "SELECT * FROM (SELECT * FROM secrets) AS t"
        with pytest.raises(SecurityViolationError) as exc_info:
            validator_with_blocked_tables.validate_or_raise(sql)
        assert "secrets" in str(exc_info.value).lower()

    def test_allowed_table_access(
        self, validator_with_blocked_tables: SQLValidator
    ) -> None:
        """Non-blocked tables should be accessible."""
        sql = "SELECT * FROM users"
        is_valid, error = validator_with_blocked_tables.validate(sql)
        assert is_valid
        assert error is None


class TestBlockedColumnAccess:
    """Test that access to blocked columns is prevented."""

    def test_blocked_column_direct(
        self, validator_with_blocked_columns: SQLValidator
    ) -> None:
        """Direct reference to a blocked column should be rejected."""
        sql = "SELECT password FROM users"
        with pytest.raises(SecurityViolationError) as exc_info:
            validator_with_blocked_columns.validate_or_raise(sql)
        assert "password" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT PASSWORD FROM users",
            "SELECT Password FROM users",
        ],
        ids=["uppercase", "mixed_case"],
    )
    def test_blocked_column_case_insensitive(
        self, validator_with_blocked_columns: SQLValidator, sql: str
    ) -> None:
        """Column blocking should be case-insensitive."""
        with pytest.raises(SecurityViolationError) as exc_info:
            validator_with_blocked_columns.validate_or_raise(sql)
        assert "password" in str(exc_info.value).lower()

    def test_blocked_column_qualified(
        self, validator_with_blocked_columns: SQLValidator
    ) -> None:
        """Qualified column reference (table.column) should still be blocked."""
        sql = "SELECT users.password FROM users"
        with pytest.raises(SecurityViolationError) as exc_info:
            validator_with_blocked_columns.validate_or_raise(sql)
        assert "password" in str(exc_info.value).lower()

    def test_allowed_column_access(
        self, validator_with_blocked_columns: SQLValidator
    ) -> None:
        """Non-blocked columns should be accessible."""
        sql = "SELECT name, email FROM users"
        is_valid, error = validator_with_blocked_columns.validate(sql)
        assert is_valid
        assert error is None
