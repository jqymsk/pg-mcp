"""Security tests for SQL injection prevention.

Tests verify that the SQLValidator correctly blocks various SQL injection
techniques including multi-statement injection, comment truncation,
dangerous function calls, DDL/DML injection, and subquery abuse.
"""

import pytest

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.models.errors import SecurityViolationError, SQLParseError
from pg_mcp.services.sql_validator import SQLValidator


@pytest.fixture
def validator() -> SQLValidator:
    """Create a default validator with EXPLAIN disabled."""
    return SQLValidator(config=SecurityConfig(), allow_explain=False)


class TestUnionInjection:
    """Test UNION-based injection attempts."""

    def test_union_injection_allowed(self, validator: SQLValidator) -> None:
        """UNION of SELECTs is valid SQL and should be allowed by the validator."""
        sql = "SELECT * FROM users UNION SELECT * FROM passwords"
        is_valid, error = validator.validate(sql)
        assert is_valid
        assert error is None


class TestCommentTruncation:
    """Test comment-based truncation attacks."""

    def test_comment_truncation(self, validator: SQLValidator) -> None:
        """Multi-statement with trailing comment should be rejected."""
        sql = "SELECT * FROM users; DROP TABLE users;--"
        with pytest.raises(SecurityViolationError) as exc_info:
            validator.validate_or_raise(sql)
        assert "multiple" in str(exc_info.value).lower()


class TestMultiStatementInjection:
    """Test multi-statement injection attempts."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; DELETE FROM users",
            "SELECT 1; INSERT INTO logs VALUES(1)",
            "SELECT 1; UPDATE users SET admin=true",
            "SELECT 1; DROP TABLE users",
        ],
        ids=[
            "select_then_delete",
            "select_then_insert",
            "select_then_update",
            "select_then_drop",
        ],
    )
    def test_multi_statement_injection(self, validator: SQLValidator, sql: str) -> None:
        """All multi-statement queries should be rejected."""
        with pytest.raises(SecurityViolationError) as exc_info:
            validator.validate_or_raise(sql)
        assert "multiple" in str(exc_info.value).lower()


class TestSubqueryWithWriteOperation:
    """Test write operations hidden inside subqueries."""

    def test_subquery_with_write_operation(self, validator: SQLValidator) -> None:
        """DELETE inside a subquery should be rejected."""
        sql = "SELECT * FROM (DELETE FROM users RETURNING *) AS t"
        with pytest.raises((SecurityViolationError, SQLParseError)):
            validator.validate_or_raise(sql)


class TestFunctionInjection:
    """Test dangerous function injection attempts."""

    @pytest.mark.parametrize(
        "sql,blocked_func",
        [
            ("SELECT pg_sleep(100)", "pg_sleep"),
            ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
            ("SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(id int)", "dblink"),
        ],
        ids=["pg_sleep", "pg_read_file", "dblink"],
    )
    def test_function_injection(
        self, validator: SQLValidator, sql: str, blocked_func: str
    ) -> None:
        """Dangerous PostgreSQL functions should be blocked."""
        with pytest.raises(SecurityViolationError) as exc_info:
            validator.validate_or_raise(sql)
        assert blocked_func in str(exc_info.value).lower()


class TestValidSelectQueries:
    """Test that legitimate SELECT queries pass validation."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM users",
            "SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'",
            "WITH cte AS (SELECT 1) SELECT * FROM cte",
            "SELECT a.id, b.name FROM a JOIN b ON a.id = b.id",
        ],
        ids=["simple_select", "count_with_where", "cte", "join"],
    )
    def test_valid_select_queries(self, validator: SQLValidator, sql: str) -> None:
        """Valid SELECT queries should pass validation."""
        is_valid, error = validator.validate(sql)
        assert is_valid
        assert error is None


class TestDDLInjection:
    """Test DDL statement injection attempts."""

    @pytest.mark.parametrize(
        "sql,keyword",
        [
            ("DROP TABLE users", "DROP"),
            ("CREATE TABLE evil(id int)", "CREATE"),
            ("ALTER TABLE users ADD COLUMN hack text", "ALTER"),
        ],
        ids=["drop", "create", "alter"],
    )
    def test_ddl_injection(
        self, validator: SQLValidator, sql: str, keyword: str
    ) -> None:
        """DDL statements should be rejected."""
        with pytest.raises(SecurityViolationError) as exc_info:
            validator.validate_or_raise(sql)
        assert keyword in str(exc_info.value).upper()


class TestDMLInjection:
    """Test DML statement injection attempts."""

    @pytest.mark.parametrize(
        "sql,keyword",
        [
            ("INSERT INTO users VALUES(1, 'hack')", "INSERT"),
            ("UPDATE users SET name='hack'", "UPDATE"),
            ("DELETE FROM users", "DELETE"),
        ],
        ids=["insert", "update", "delete"],
    )
    def test_dml_injection(
        self, validator: SQLValidator, sql: str, keyword: str
    ) -> None:
        """DML statements should be rejected."""
        with pytest.raises(SecurityViolationError) as exc_info:
            validator.validate_or_raise(sql)
        assert keyword in str(exc_info.value).upper()


class TestEmptyAndWhitespaceSQL:
    """Test handling of empty, whitespace-only, and comment-only SQL."""

    @pytest.mark.parametrize(
        "sql",
        [
            "",
            "   ",
            "-- just a comment",
        ],
        ids=["empty", "whitespace_only", "comment_only"],
    )
    def test_empty_and_whitespace_sql(self, validator: SQLValidator, sql: str) -> None:
        """Empty, whitespace-only, and comment-only SQL should raise SQLParseError."""
        with pytest.raises(SQLParseError):
            validator.validate_or_raise(sql)
