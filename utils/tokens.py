"""
Token counting utilities using tiktoken.
"""
import tiktoken

# Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
_encoding = None


def _get_encoding():
    """Lazy load the tiktoken encoding."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """
    Count tokens in text using cl100k_base encoding.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    if not text:
        return 0
    encoding = _get_encoding()
    return len(encoding.encode(text))


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
