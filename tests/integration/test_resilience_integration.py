"""Integration tests for resilience components with the orchestrator.

This module tests that rate limiting, circuit breaker, and retry backoff
work correctly when integrated into the full query orchestration pipeline.
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from pg_mcp.config.settings import ResilienceConfig, SecurityConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, SecurityViolationError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.resilience.circuit_breaker import CircuitBreaker, CircuitState
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


def _make_schema(db_name: str = "test_db") -> DatabaseSchema:
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


def _build_orchestrator(
    *,
    resilience_config: ResilienceConfig | None = None,
    sql_generator: AsyncMock | None = None,
    rate_limiter: MultiRateLimiter | None = None,
    sql_validator_mock: MagicMock | None = None,
) -> QueryOrchestrator:
    """Build an orchestrator with sensible defaults for resilience tests."""
    config = resilience_config or ResilienceConfig()

    generator = sql_generator or AsyncMock()
    if sql_generator is None:
        generator.generate.return_value = "SELECT * FROM users;"

    validator = sql_validator_mock or MagicMock()
    if sql_validator_mock is None:
        validator.validate_or_raise.return_value = None

    executor = AsyncMock()
    executor.execute.return_value = ([{"id": 1}], 1)

    cache = MagicMock()
    cache.get.return_value = _make_schema()

    return QueryOrchestrator(
        sql_generator=generator,
        sql_validators={"test_db": validator},
        sql_executors={"test_db": executor},
        result_validator=AsyncMock(),
        schema_cache=cache,
        pools={"test_db": MagicMock()},
        resilience_config=config,
        validation_config=ValidationConfig(enabled=False),
        rate_limiter=rate_limiter,
    )


class TestRateLimiterIntegration:
    """Test rate limiter behaviour when integrated with the orchestrator."""

    @pytest.mark.asyncio
    async def test_rate_limiter_limits_concurrent_queries(self) -> None:
        """With query_limit=2, launching 5 concurrent queries should queue them.
        All should eventually complete, and the rate limiter stats should reflect
        the total requests processed."""
        rate_limiter = MultiRateLimiter(query_limit=2, llm_limit=5)

        # Add a small delay to the generator so concurrency is observable
        async def slow_generate(**kwargs):
            await asyncio.sleep(0.05)
            return "SELECT * FROM users;"

        generator = AsyncMock()
        generator.generate.side_effect = slow_generate

        orchestrator = _build_orchestrator(
            sql_generator=generator,
            rate_limiter=rate_limiter,
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )

        # Launch 5 concurrent queries
        tasks = [orchestrator.execute_query(request) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(responses) == 5
        for resp in responses:
            assert resp.success is True

        # Rate limiter should have processed all requests
        stats = rate_limiter.get_all_stats()
        assert stats["queries"]["total_requests"] == 5


class TestCircuitBreakerIntegration:
    """Test circuit breaker behaviour when integrated with the orchestrator."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self) -> None:
        """When the generator always fails, the circuit breaker should open
        after enough failures, and subsequent queries should fail immediately."""
        generator = AsyncMock()
        generator.generate.side_effect = RuntimeError("LLM is down")

        orchestrator = _build_orchestrator(
            resilience_config=ResilienceConfig(
                max_retries=0,
                circuit_breaker_threshold=2,
                circuit_breaker_timeout=60.0,
            ),
            sql_generator=generator,
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )

        # First two queries should fail and trip the breaker
        for _ in range(2):
            resp = await orchestrator.execute_query(request)
            assert resp.success is False

        # Circuit should now be open
        assert orchestrator.circuit_breaker.state == CircuitState.OPEN

        # Next query should fail immediately with circuit breaker message
        resp = await orchestrator.execute_query(request)
        assert resp.success is False
        assert resp.error is not None
        assert "circuit breaker" in resp.error.message.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self) -> None:
        """After the circuit breaker opens, waiting for recovery_timeout should
        allow it to transition to HALF_OPEN, and a successful request should
        close it again."""
        generator = AsyncMock()
        generator.generate.side_effect = RuntimeError("LLM is down")

        orchestrator = _build_orchestrator(
            resilience_config=ResilienceConfig(
                max_retries=0,
                circuit_breaker_threshold=1,
                circuit_breaker_timeout=10.0,  # Will be overridden below
            ),
            sql_generator=generator,
        )

        # Override recovery timeout to something very short for testing
        orchestrator.circuit_breaker._recovery_timeout = 0.1

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )

        # Trip the circuit breaker
        resp = await orchestrator.execute_query(request)
        assert resp.success is False
        assert orchestrator.circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Now make the generator succeed
        generator.generate.side_effect = None
        generator.generate.return_value = "SELECT * FROM users;"

        # Next query should succeed and close the circuit
        resp = await orchestrator.execute_query(request)
        assert resp.success is True
        assert orchestrator.circuit_breaker.state == CircuitState.CLOSED


class TestRetryBackoff:
    """Test retry with exponential backoff delay."""

    @pytest.mark.asyncio
    async def test_retry_backoff_delay(self) -> None:
        """With retry_delay=0.1 and backoff_factor=2.0, the first retry should
        wait at least 0.1s. We verify by measuring elapsed time."""
        generator = AsyncMock()
        generator.generate.side_effect = [
            "DELETE FROM users;",  # First attempt: will fail validation
            "SELECT * FROM users;",  # Second attempt: will pass
        ]

        validator = MagicMock()
        validator.validate_or_raise.side_effect = [
            SecurityViolationError("DELETE statements are not allowed"),
            None,  # Success on second attempt
        ]

        orchestrator = _build_orchestrator(
            resilience_config=ResilienceConfig(
                max_retries=2,
                retry_delay=0.1,
                backoff_factor=2.0,
            ),
            sql_generator=generator,
            sql_validator_mock=validator,
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )

        start = time.monotonic()
        resp = await orchestrator.execute_query(request)
        elapsed = time.monotonic() - start

        assert resp.success is True
        # First retry delay = retry_delay * backoff_factor^0 = 0.1s
        assert elapsed >= 0.09, f"Expected >= 0.09s backoff delay, got {elapsed:.3f}s"
        assert generator.generate.call_count == 2
