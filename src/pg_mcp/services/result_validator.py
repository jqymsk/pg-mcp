"""Result validation service using Google Gemini for query result verification.

This module provides the ResultValidator class that uses Google's Gemini LLM to validate
whether query results correctly match the user's original question.
"""

import json
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types

from pg_mcp.config.settings import GeminiConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.models.query import ResultValidationResult
from pg_mcp.prompts.result_validation import (
    RESULT_VALIDATION_SYSTEM_PROMPT,
    build_validation_prompt,
)

if TYPE_CHECKING:
    pass


class ResultValidator:
    """Result validator using Google Gemini for query result verification.

    This class handles the interaction with Google's Gemini API to validate whether
    query results correctly answer the user's original question. It provides
    confidence scoring and suggestions for improvement.

    Example:
        >>> config = GeminiConfig(api_key="AIza...", model="gemini-3-flash-preview")
        >>> validation_config = ValidationConfig(confidence_threshold=70)
        >>> validator = ResultValidator(config, validation_config)
        >>> result = await validator.validate(
        ...     question="How many users registered today?",
        ...     sql="SELECT COUNT(*) FROM users WHERE created_at >= CURRENT_DATE",
        ...     results=[{"count": 42}],
        ...     row_count=1
        ... )
    """

    def __init__(
        self,
        gemini_config: GeminiConfig,
        validation_config: ValidationConfig,
    ) -> None:
        """Initialize result validator with Gemini and validation configuration.

        Args:
            gemini_config: Gemini configuration including API key and model settings.
            validation_config: Validation configuration including thresholds and timeouts.
        """
        self.gemini_config = gemini_config
        self.validation_config = validation_config
        self.client = genai.Client(api_key=gemini_config.api_key.get_secret_value())

    async def validate(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
    ) -> ResultValidationResult:
        """Validate query results against the user's original question.

        This method sends the question, SQL, and results to Gemini's API
        and gets a confidence assessment of whether the results correctly
        answer the question.

        Args:
            question: The user's original natural language question.
            sql: The SQL query that was executed.
            results: Query results (will be sampled if too large).
            row_count: Total number of rows in the complete result set.

        Returns:
            ResultValidationResult: Validation result including confidence score,
                explanation, and optional suggestions.

        Raises:
            LLMError: If validation fails or response is invalid.
            LLMTimeoutError: If the API request times out.
            LLMUnavailableError: If the API is unavailable or authentication fails.

        Example:
            >>> result = await validator.validate(
            ...     question="Count active users",
            ...     sql="SELECT COUNT(*) FROM users WHERE status = 'active'",
            ...     results=[{"count": 150}],
            ...     row_count=1
            ... )
            >>> print(f"Confidence: {result.confidence}%")
            >>> print(f"Acceptable: {result.is_acceptable}")
        """
        # If validation is disabled, return high confidence result
        if not self.validation_config.enabled:
            return ResultValidationResult(
                confidence=100,
                explanation="Validation is disabled in configuration",
                suggestion=None,
                is_acceptable=True,
            )

        # Sample results to avoid sending too much data to LLM
        sample_results = results[: self.validation_config.sample_rows]

        # Build the validation prompt
        prompt = build_validation_prompt(
            question=question,
            sql=sql,
            results=sample_results,
            row_count=row_count,
        )

        # Combine system prompt and user prompt
        full_prompt = f"{RESULT_VALIDATION_SYSTEM_PROMPT}\n\n{prompt}"

        try:
            # Call Gemini API with JSON output
            response = await self.client.aio.models.generate_content(
                model=self.gemini_config.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Use deterministic output for validation
                    max_output_tokens=500,
                    response_mime_type="application/json",
                ),
            )

            # Extract and parse the response
            if not response.text:
                raise LLMError(
                    message="Gemini returned empty response for result validation",
                    details={"response": str(response)},
                )

            content = response.text

            # Parse JSON response
            try:
                result_dict = json.loads(content)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return moderate confidence with error explanation
                return ResultValidationResult(
                    confidence=60,
                    explanation=f"Validation response parsing failed: {e!s}",
                    suggestion="Unable to parse LLM response, manual verification recommended",
                    is_acceptable=False,
                )

            # Extract fields from response
            confidence = result_dict.get("confidence", 50)
            explanation = result_dict.get("explanation", "No explanation provided")
            suggestion = result_dict.get("suggestion")

            # Validate confidence is within bounds
            if not isinstance(confidence, int) or not (0 <= confidence <= 100):
                confidence = (
                    max(0, min(100, int(confidence)))
                    if isinstance(confidence, (int, float))
                    else 50
                )

            # Determine if result is acceptable based on threshold
            is_acceptable = confidence >= self.validation_config.confidence_threshold

            return ResultValidationResult(
                confidence=confidence,
                explanation=explanation,
                suggestion=suggestion,
                is_acceptable=is_acceptable,
            )

        except TimeoutError as e:
            timeout = self.validation_config.timeout_seconds
            raise LLMTimeoutError(
                message=f"Result validation timed out after {timeout}s",
                details={"timeout": timeout},
            ) from e
        except json.JSONDecodeError as e:
            # This should be caught above, but kept for safety
            return ResultValidationResult(
                confidence=60,
                explanation=f"JSON parsing error: {e!s}",
                suggestion=None,
                is_acceptable=False,
            )
        except LLMError:
            # Re-raise LLM errors as-is
            raise
        except Exception as e:
            # Handle various Gemini errors
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise LLMUnavailableError(
                    message="Gemini API authentication failed - check API key",
                    details={"error": error_msg},
                ) from e
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                raise LLMUnavailableError(
                    message="Gemini API quota/rate limit exceeded",
                    details={"error": error_msg},
                ) from e
            raise LLMError(
                message=f"Result validation failed: {error_msg}",
                details={"error": error_msg},
            ) from e
