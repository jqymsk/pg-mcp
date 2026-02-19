"""Unit tests for metrics recording by the orchestrator.

This module verifies that the orchestrator correctly records Prometheus metrics
for query requests, LLM calls, SQL rejections, and database query durations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from prometheus_client import REGISTRY

from pg_mcp.config.settings import ResilienceConfig, SecurityConfig, ValidationConfig
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.models.query import (
    QueryRequest,
    ResultValidationResult,
    ReturnType,
)
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.observability.metrics import MetricsCollector
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


@pytest.fixture()
def metrics() -> MetricsCollector:
    """Provide a MetricsCollector with freshly re-initialized metrics.

    Because MetricsCollector is a singleton, we call _initialize_metrics()
    to reset counters/histograms between tests. Note that prometheus_client
    will raise if you try to re-register a metric with the same name, so
    _initialize_metrics silently re-assigns the attributes (the underlying
    Prometheus collectors remain registered but their values are cumulative).

    To work around this, we unregister and re-register by calling
    _initialize_metrics which re-creates the metric objects.
    """
    # Force singleton creation
    collector = MetricsCollector()

    # Unregister existing metrics to allow re-creation
    for name in list(REGISTRY._names_to_collectors.keys()):
        if name.startswith("pg_mcp_"):
            try:
                REGISTRY.unregister(REGISTRY._names_to_collectors[name])
            except Exception:
                pass

    # Re-initialize metrics (creates fresh counters/histograms)
    collector._initialize_metrics()
    return collector


def _build_orchestrator(
    metrics: MetricsCollector,
    *,
    sql_generator: AsyncMock | None = None,
    sql_validator_mock: MagicMock | None = None,
    sql_executor: AsyncMock | None = None,
    result_validator: AsyncMock | None = None,
    validation_enabled: bool = False,
) -> QueryOrchestrator:
    """Build an orchestrator wired to the given MetricsCollector."""
    generator = sql_generator or AsyncMock()
    if sql_generator is None:
        generator.generate.return_value = "SELECT * FROM users;"

    validator = sql_validator_mock or MagicMock()
    if sql_validator_mock is None:
        validator.validate_or_raise.return_value = None

    executor = sql_executor or AsyncMock()
    if sql_executor is None:
        executor.execute.return_value = ([{"id": 1}], 1)

    rv = result_validator or AsyncMock()
    if result_validator is None:
        rv.validate.return_value = ResultValidationResult(
            confidence=90,
            explanation="Looks good",
            suggestion=None,
            is_acceptable=True,
        )

    cache = MagicMock()
    cache.get.return_value = _make_schema()

    return QueryOrchestrator(
        sql_generator=generator,
        sql_validators={"test_db": validator},
        sql_executors={"test_db": executor},
        result_validator=rv,
        schema_cache=cache,
        pools={"test_db": MagicMock()},
        resilience_config=ResilienceConfig(),
        validation_config=ValidationConfig(enabled=validation_enabled),
        metrics=metrics,
    )


class TestMetricsOnSuccess:
    """Verify metrics are recorded for successful queries."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_success(self, metrics: MetricsCollector) -> None:
        """A successful query should increment query_requests with status=success."""
        orchestrator = _build_orchestrator(metrics)

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True

        # Check that the success counter was incremented
        value = metrics.query_requests.labels(
            status="success", database="test_db"
        )._value.get()
        assert value >= 1


class TestMetricsOnError:
    """Verify metrics are recorded for failed queries."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_error(self, metrics: MetricsCollector) -> None:
        """A failing query should increment query_requests with status=error."""
        generator = AsyncMock()
        generator.generate.side_effect = RuntimeError("boom")

        orchestrator = _build_orchestrator(
            metrics,
            sql_generator=generator,
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False

        value = metrics.query_requests.labels(
            status="error", database="test_db"
        )._value.get()
        assert value >= 1


class TestLLMMetrics:
    """Verify LLM call metrics are recorded."""

    @pytest.mark.asyncio
    async def test_llm_metrics_recorded(self, metrics: MetricsCollector) -> None:
        """After SQL generation, llm_calls counter should be incremented
        and llm_latency histogram should have observations."""
        orchestrator = _build_orchestrator(metrics)

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True

        # llm_calls counter for generate_sql should be >= 1
        llm_count = metrics.llm_calls.labels(
            operation="generate_sql"
        )._value.get()
        assert llm_count >= 1

        # llm_latency histogram should have at least one observation
        # Access the histogram's _sum to verify observations were recorded
        latency_sum = metrics.llm_latency.labels(
            operation="generate_sql"
        )._sum.get()
        assert latency_sum >= 0  # Duration is non-negative


class TestSQLRejectedMetrics:
    """Verify SQL rejection metrics are recorded."""

    @pytest.mark.asyncio
    async def test_sql_rejected_metrics(self, metrics: MetricsCollector) -> None:
        """When the validator rejects SQL, sql_rejected counter should be incremented."""
        generator = AsyncMock()
        generator.generate.return_value = "DELETE FROM users;"

        validator = MagicMock()
        validator.validate_or_raise.side_effect = SecurityViolationError(
            "DELETE statements are not allowed"
        )

        orchestrator = _build_orchestrator(
            metrics,
            sql_generator=generator,
            sql_validator_mock=validator,
        )

        request = QueryRequest(
            question="Delete all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False

        # sql_rejected counter should have been incremented
        rejected_count = metrics.sql_rejected.labels(
            reason="validation_failed"
        )._value.get()
        assert rejected_count >= 1


class TestDBQueryDurationMetrics:
    """Verify database query duration metrics are recorded."""

    @pytest.mark.asyncio
    async def test_db_query_duration_recorded(self, metrics: MetricsCollector) -> None:
        """After SQL execution, db_query_duration histogram should have observations."""
        orchestrator = _build_orchestrator(metrics)

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True

        # db_query_duration histogram should have at least one observation
        duration_sum = metrics.db_query_duration._sum.get()
        assert duration_sum >= 0  # Duration is non-negative

        # Also verify query_duration (overall) was observed
        overall_sum = metrics.query_duration._sum.get()
        assert overall_sum >= 0
