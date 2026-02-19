"""Security tests for EXPLAIN statement policy enforcement.

Tests verify that the SQLValidator correctly allows or denies EXPLAIN
statements based on the allow_explain configuration flag.
"""

import pytest

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.services.sql_validator import SQLValidator


@pytest.fixture
def validator_explain_allowed() -> SQLValidator:
    """Create a validator that allows EXPLAIN statements."""
    return SQLValidator(config=SecurityConfig(), allow_explain=True)


@pytest.fixture
def validator_explain_denied() -> SQLValidator:
    """Create a validator that denies EXPLAIN statements."""
    return SQLValidator(config=SecurityConfig(), allow_explain=False)


class TestExplainAllowed:
    """Test EXPLAIN behavior when allow_explain=True."""

    def test_explain_allowed(self, validator_explain_allowed: SQLValidator) -> None:
        """EXPLAIN SELECT should pass when allow_explain is True."""
        sql = "EXPLAIN SELECT * FROM users"
        is_valid, error = validator_explain_allowed.validate(sql)
        assert is_valid
        assert error is None

    def test_explain_analyze_allowed(
        self, validator_explain_allowed: SQLValidator
    ) -> None:
        """EXPLAIN ANALYZE should pass when allow_explain is True."""
        sql = "EXPLAIN ANALYZE SELECT * FROM users"
        is_valid, error = validator_explain_allowed.validate(sql)
        assert is_valid
        assert error is None


class TestExplainDenied:
    """Test EXPLAIN behavior when allow_explain=False."""

    def test_explain_denied(self, validator_explain_denied: SQLValidator) -> None:
        """EXPLAIN SELECT should be rejected when allow_explain is False."""
        sql = "EXPLAIN SELECT * FROM users"
        with pytest.raises(SecurityViolationError) as exc_info:
            validator_explain_denied.validate_or_raise(sql)
        assert "explain" in str(exc_info.value).lower()

    def test_explain_denied_by_default(self) -> None:
        """Default validator (allow_explain=False) should reject EXPLAIN."""
        validator = SQLValidator(config=SecurityConfig(), allow_explain=False)
        sql = "EXPLAIN SELECT * FROM users"
        with pytest.raises(SecurityViolationError) as exc_info:
            validator.validate_or_raise(sql)
        assert "explain" in str(exc_info.value).lower()
