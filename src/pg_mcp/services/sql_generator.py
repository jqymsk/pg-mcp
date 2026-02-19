"""SQL generation service using OpenAI for natural language to SQL conversion.

This module provides the SQLGenerator class that uses OpenAI's LLM to convert
natural language questions into valid PostgreSQL SQL queries.
"""

import asyncio
import re
from typing import TYPE_CHECKING

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from pg_mcp.config.settings import GeminiConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.prompts.sql_generation import SQL_GENERATION_SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from pg_mcp.models.schema import DatabaseSchema


class SQLGenerator:
    """SQL generator using Gemini for natural language to SQL conversion.

    This class handles the interaction with Gemini's API to generate SQL queries
    from natural language questions. It includes robust error handling, SQL extraction
    from various response formats, and support for retry scenarios with error feedback.

    Example:
        >>> config = GeminiConfig(api_key="AIza...", model="gemini-3-flash-preview")
        >>> generator = SQLGenerator(config)
        >>> sql = await generator.generate(
        ...     question="How many users registered today?",
        ...     schema=db_schema
        ... )
    """

    def __init__(self, config: GeminiConfig) -> None:
        """Initialize SQL generator with Gemini configuration.

        Args:
            config: Gemini configuration including API key and model settings.
        """
        self.config = config
        genai.configure(api_key=config.api_key.get_secret_value())
        self.model = genai.GenerativeModel(
            model_name=config.model,
            system_instruction=SQL_GENERATION_SYSTEM_PROMPT,
        )

    async def generate(
        self,
        question: str,
        schema: "DatabaseSchema",
        context: str | None = None,
        previous_attempt: str | None = None,
        error_feedback: str | None = None,
    ) -> str:
        """Generate SQL statement from natural language question.

        This method sends the question and database schema to OpenAI's API
        and extracts the generated SQL query from the response. It supports
        retry scenarios by accepting previous failed attempts and error feedback.

        Args:
            question: User's natural language question.
            schema: Database schema information for context.
            context: Optional additional context to guide generation.
            previous_attempt: Previously generated SQL that failed (for retry).
            error_feedback: Error message from previous attempt (for retry).

        Returns:
            str: Generated SQL query (without trailing semicolon).

        Raises:
            LLMError: If generation fails or response is invalid.
            LLMTimeoutError: If the API request times out.
            LLMUnavailableError: If the API is unavailable or authentication fails.

        Example:
            >>> # Initial generation
            >>> sql = await generator.generate(
            ...     question="Count all active users",
            ...     schema=db_schema
            ... )
            >>> # Retry with error feedback
            >>> sql = await generator.generate(
            ...     question="Count all active users",
            ...     schema=db_schema,
            ...     previous_attempt="SELECT COUNT(*) FROM user",
            ...     error_feedback='relation "user" does not exist'
            ... )
        """
        user_prompt = build_user_prompt(
            question=question,
            schema=schema,
            context=context,
            previous_attempt=previous_attempt,
            error_feedback=error_feedback,
        )

        try:
            generation_config = GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            # Using wait_for to implement timeout since SDK might not support it natively in all versions
            response = await asyncio.wait_for(
                self.model.generate_content_async(
                    user_prompt,
                    generation_config=generation_config,
                ),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(
                message=f"Gemini API request timed out after {self.config.timeout}s",
                details={"timeout": self.config.timeout},
            ) from e
        except Exception as e:
            # Handle various Gemini errors
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "authentication" in error_msg.lower():
                raise LLMUnavailableError(
                    message="Gemini API authentication failed - check API key",
                    details={"error": error_msg},
                ) from e
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                raise LLMUnavailableError(
                    message="Gemini API rate limit exceeded",
                    details={"error": error_msg},
                ) from e
            raise LLMError(
                message=f"Gemini API request failed: {error_msg}",
                details={"error": error_msg},
            ) from e

        # Extract SQL from response
        if not response or not response.text:
            raise LLMError(
                message="Gemini returned empty response",
                details={"response": str(response)},
            )

        content = response.text
        sql = self._extract_sql(content)
        if not sql:
            raise LLMError(
                message="Failed to extract SQL from Gemini response",
                details={"content": content},
            )

        return sql

    def _extract_sql(self, content: str) -> str | None:
        """Extract SQL query from LLM response content.

        This method implements a multi-strategy approach to extract SQL from
        various response formats that LLMs might generate:

        1. Try to match ```sql ... ``` code blocks (preferred format)
        2. Try to match generic ``` ... ``` code blocks
        3. Try to find SELECT/WITH statements in plain text
        4. Check if entire content looks like SQL

        Args:
            content: Raw content from LLM response.

        Returns:
            str | None: Extracted SQL query, or None if extraction fails.

        Example:
            >>> generator._extract_sql("```sql\\nSELECT 1;\\n```")
            'SELECT 1;'
            >>> generator._extract_sql("SELECT * FROM users;")
            'SELECT * FROM users;'
        """
        if not content:
            return None

        content = content.strip()

        # Strategy 1: Match ```sql ... ``` or ``` ... ``` code blocks
        code_block_pattern = r"```(?:sql)?\s*\n?(.*?)\n?```"
        matches = re.findall(code_block_pattern, content, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
            # Remove trailing semicolon for consistency
            return sql.rstrip(";") + ";"

        # Strategy 2: Find SELECT/WITH statements in plain text
        sql_pattern = r"((?:WITH|SELECT)\s+.*?)(?:;|$)"
        matches = re.findall(sql_pattern, content, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
            return sql.rstrip(";") + ";"

        # Strategy 3: Check if entire content looks like SQL
        if content.upper().startswith(("SELECT", "WITH")):
            return content.rstrip(";") + ";"

        return None
