"""
Token counting utilities with provider-aware counting.

Supports both OpenAI (tiktoken) and Gemini (native API) token counting.
"""
import tiktoken

# Lazy-loaded encodings/clients
_tiktoken_encoding = None
_gemini_client = None


def _get_tiktoken_encoding():
    """Lazy load the tiktoken encoding for OpenAI."""
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoding


def _get_gemini_client():
    """Lazy load Gemini client for token counting."""
    global _gemini_client
    if _gemini_client is None:
        from utils.gemini_client import get_client
        _gemini_client = get_client()
    return _gemini_client


def count_tokens(text: str, provider: str = "openai") -> int:
    """
    Count tokens in text using the appropriate tokenizer.

    Args:
        text: Text to count tokens for
        provider: LLM provider ("gemini" or "openai")

    Returns:
        Number of tokens
    """
    if not text:
        return 0

    if provider == "gemini":
        return count_tokens_gemini(text)
    else:
        return count_tokens_openai(text)


def count_tokens_openai(text: str) -> int:
    """Count tokens using OpenAI's cl100k_base encoding."""
    if not text:
        return 0
    encoding = _get_tiktoken_encoding()
    return len(encoding.encode(text))


def count_tokens_gemini(text: str) -> int:
    """Count tokens using Gemini's native tokenizer."""
    if not text:
        return 0
    try:
        from utils.gemini_client import GEMINI_MODEL
        client = _get_gemini_client()
        result = client.models.count_tokens(
            model=GEMINI_MODEL,
            contents=text
        )
        return result.total_tokens
    except Exception:
        # Fallback to tiktoken if Gemini API fails
        return count_tokens_openai(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text
    """
    if not text:
        return text
    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
