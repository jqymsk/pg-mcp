"""Integration tests for multi-database routing and per-database security isolation.

This module tests that the orchestrator correctly routes queries to the right
database and enforces per-database security policies using real SQLValidator
instances with mocked LLM and execution components.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from pg_mcp.config.settings import ResilienceConfig, SecurityConfig, ValidationConfig
from pg_mcp.models.errors import DatabaseError, SecurityViolationError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator


def _make_schema(db_name: str) -> DatabaseSchema:
    """Create a minimal DatabaseSchema for testing."""
    return DatabaseSchema(
        database_name=db_name,
        tables=[
            TableInfo(
                schema_name="public",
                table_name="users",
                columns=[
                    ColumnInfo(
                        name="id",
                        data_type="integer",
                        is_nullable=False,
                        is_primary_key=True,
                    ),
                ],
            ),
        ],
        version="15.0",
    )


def _make_orchestrator(
    pools: dict[str, MagicMock],
    sql_validators: dict[str, SQLValidator] | None = None,
    sql_executors: dict[str, AsyncMock] | None = None,
    sql_generator: AsyncMock | None = None,
    schemas: dict[str, DatabaseSchema] | None = None,
) -> QueryOrchestrator:
    """Build an orchestrator with sensible defaults for integration tests."""
    generator = sql_generator or AsyncMock()
    if sql_generator is None:
        generator.generate.return_value = "SELECT * FROM users;"

    validators = sql_validators or {
        name: MagicMock() for name in pools
    }
    # Default validators pass everything
    for v in validators.values():
        if isinstance(v, MagicMock):
            v.validate_or_raise.return_value = None

    executors = sql_executors or {
        name: AsyncMock() for name in pools
    }
    for ex in executors.values():
        if isinstance(ex, AsyncMock) and not ex.execute.side_effect:
            ex.execute.return_value = ([{"id": 1}], 1)

    cache = MagicMock()
    if schemas:
        cache.get.side_effect = lambda db: schemas.get(db)
    else:
        cache.get.side_effect = lambda db: _make_schema(db)

    return QueryOrchestrator(
        sql_generator=generator,
        sql_validators=validators,
        sql_executors=executors,
        result_validator=AsyncMock(),
        schema_cache=cache,
        pools=pools,
        resilience_config=ResilienceConfig(),
        validation_config=ValidationConfig(enabled=False),
    )


class TestMultiDatabaseRouting:
    """Test that queries are routed to the correct database."""

    @pytest.mark.asyncio
    async def test_route_to_correct_database(self) -> None:
        """Sending database='db2' should invoke db2's executor, not db1's."""
        db1_executor = AsyncMock()
        db1_executor.execute.return_value = ([{"src": "db1"}], 1)
        db2_executor = AsyncMock()
        db2_executor.execute.return_value = ([{"src": "db2"}], 1)

        orchestrator = _make_orchestrator(
            pools={"db1": MagicMock(), "db2": MagicMock()},
            sql_executors={"db1": db1_executor, "db2": db2_executor},
        )

        request = QueryRequest(
            question="Get all users",
            database="db2",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        db2_executor.execute.assert_called_once()
        db1_executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_select_single_database(self) -> None:
        """With one database configured, database=None should auto-select it."""
        orchestrator = _make_orchestrator(
            pools={"only_db": MagicMock()},
        )

        request = QueryRequest(
            question="Get all users",
            database=None,
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.generated_sql is not None

    @pytest.mark.asyncio
    async def test_multiple_databases_require_explicit_selection(self) -> None:
        """With multiple databases, database=None should return an error."""
        orchestrator = _make_orchestrator(
            pools={"db1": MagicMock(), "db2": MagicMock()},
        )

        request = QueryRequest(
            question="Get all users",
            database=None,
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "multiple databases" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_database_error(self) -> None:
        """Requesting a nonexistent database should return an error."""
        orchestrator = _make_orchestrator(
            pools={"db1": MagicMock()},
        )

        request = QueryRequest(
            question="Get all users",
            database="nonexistent",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "not found" in response.error.message.lower()


class TestPerDatabaseSecurityIsolation:
    """Test that per-database security policies are enforced independently."""

    @pytest.mark.asyncio
    async def test_per_database_security_isolation(self) -> None:
        """db1 blocks 'secrets' table; db2 does not. Same SQL should be
        rejected for db1 and accepted for db2."""
        security_config = SecurityConfig()

        db1_validator = SQLValidator(
            config=security_config,
            blocked_tables=["secrets"],
        )
        db2_validator = SQLValidator(
            config=security_config,
            blocked_tables=[],
        )

        generator = AsyncMock()
        generator.generate.return_value = "SELECT * FROM secrets;"

        db2_executor = AsyncMock()
        db2_executor.execute.return_value = ([{"data": "ok"}], 1)

        orchestrator = _make_orchestrator(
            pools={"db1": MagicMock(), "db2": MagicMock()},
            sql_validators={"db1": db1_validator, "db2": db2_validator},
            sql_generator=generator,
            sql_executors={"db1": AsyncMock(), "db2": db2_executor},
        )

        # db1: should be rejected (blocked table)
        req_db1 = QueryRequest(
            question="Show secrets",
            database="db1",
            return_type=ReturnType.SQL,
        )
        resp_db1 = await orchestrator.execute_query(req_db1)
        assert resp_db1.success is False
        assert resp_db1.error is not None
        assert "security_violation" in resp_db1.error.code

        # db2: should succeed (no blocked tables)
        req_db2 = QueryRequest(
            question="Show secrets",
            database="db2",
            return_type=ReturnType.RESULT,
        )
        resp_db2 = await orchestrator.execute_query(req_db2)
        assert resp_db2.success is True

    @pytest.mark.asyncio
    async def test_per_database_explain_policy(self) -> None:
        """db1 allows EXPLAIN; db2 does not. Same EXPLAIN query should be
        accepted for db1 and rejected for db2."""
        security_config = SecurityConfig()

        db1_validator = SQLValidator(
            config=security_config,
            allow_explain=True,
        )
        db2_validator = SQLValidator(
            config=security_config,
            allow_explain=False,
        )

        generator = AsyncMock()
        generator.generate.return_value = "EXPLAIN SELECT * FROM users;"

        orchestrator = _make_orchestrator(
            pools={"db1": MagicMock(), "db2": MagicMock()},
            sql_validators={"db1": db1_validator, "db2": db2_validator},
            sql_generator=generator,
        )

        # db1: EXPLAIN allowed
        req_db1 = QueryRequest(
            question="Explain query plan",
            database="db1",
            return_type=ReturnType.SQL,
        )
        resp_db1 = await orchestrator.execute_query(req_db1)
        assert resp_db1.success is True

        # db2: EXPLAIN not allowed
        req_db2 = QueryRequest(
            question="Explain query plan",
            database="db2",
            return_type=ReturnType.SQL,
        )
        resp_db2 = await orchestrator.execute_query(req_db2)
        assert resp_db2.success is False
        assert resp_db2.error is not None
        assert "security_violation" in resp_db2.error.code
