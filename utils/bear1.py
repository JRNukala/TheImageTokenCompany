"""
bear-1 compression wrapper for The Token Company API.
"""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class CompressionResult:
    """Result from bear-1 compression."""
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int

    @property
    def savings_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.compressed_tokens / self.original_tokens) * 100


# Lazy-loaded client
_client = None


def _get_client():
    """Lazy load the TokenClient."""
    global _client
    if _client is None:
        from tokenc import TokenClient
        api_key = os.getenv("TTC_API_KEY")
        if not api_key:
            raise ValueError("TTC_API_KEY environment variable not set")
        _client = TokenClient(api_key=api_key)
    return _client


def compress(text: str, aggressiveness: float = 0.7) -> CompressionResult:
    """
    Compress text using bear-1.

    Args:
        text: Text to compress
        aggressiveness: 0.0 (minimal) to 1.0 (maximum compression)

    Returns:
        CompressionResult with original and compressed text
    """
    from utils.tokens import count_tokens

    if not text or not text.strip():
        return CompressionResult(
            original=text,
            compressed=text,
            original_tokens=0,
            compressed_tokens=0
        )

    original_tokens = count_tokens(text)

    # Skip compression for very short text
    if original_tokens <= 3:
        return CompressionResult(
            original=text,
            compressed=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens
        )

    try:
        client = _get_client()
        result = client.compress_input(
            input=text,
            aggressiveness=aggressiveness
        )
        compressed = result.output
        compressed_tokens = count_tokens(compressed)

        return CompressionResult(
            original=text,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens
        )
    except Exception as e:
        # If compression fails, return original
        print(f"[bear-1] Compression failed: {e}")
        return CompressionResult(
            original=text,
            compressed=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens
        )
