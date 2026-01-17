"""
Shared Gemini client with retry logic.

Provides a singleton client instance and retry-enabled generate function
to handle rate limits gracefully.
"""
import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Singleton client
_client = None

# Use the stable, recommended model (not the deprecated 2.0 version)
GEMINI_MODEL = "gemini-2.5-flash-lite"


def get_client():
    """
    Get shared Gemini client (singleton pattern).

    Returns the same client instance across all modules to avoid
    creating multiple connections that count against rate limits.
    """
    global _client
    if _client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_with_retry(
    prompt: str,
    max_retries: int = 3,
    initial_wait: float = 2.0
) -> str:
    """
    Generate content with exponential backoff retry for rate limits.

    Args:
        prompt: The prompt to send to Gemini
        max_retries: Maximum number of retry attempts (default: 3)
        initial_wait: Initial wait time in seconds (default: 2.0)

    Returns:
        Generated text response

    Raises:
        Exception: If all retries are exhausted or non-retryable error occurs
    """
    client = get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            last_error = e
            error_str = str(e)

            # Check if this is a retryable rate limit error
            is_rate_limit = (
                "RESOURCE_EXHAUSTED" in error_str or
                "429" in error_str or
                "rate" in error_str.lower()
            )

            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff: 2s, 4s, 8s
                wait_time = initial_wait * (2 ** attempt)
                print(f"[Gemini] Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue

            # Non-retryable error or max retries exceeded
            raise e

    # Should not reach here, but just in case
    raise last_error if last_error else Exception("Max retries exceeded")


def close_client():
    """Close the client connection (call on shutdown if needed)."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except:
            pass
        _client = None
