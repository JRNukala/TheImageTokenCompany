"""
Step 2: Optional Prompt Compression.

Pre-compresses the user prompt to make it cleaner for CVspec generation.
"""
from utils.bear1 import compress, CompressionResult
from utils.tokens import count_tokens


def compress_prompt(prompt: str, aggressiveness: float = 0.7, provider: str = "gemini") -> tuple[str, bool, CompressionResult]:
    """
    Optionally compress the user prompt.

    Skips compression for short prompts (< 10 tokens).

    Args:
        prompt: User's original prompt
        aggressiveness: bear-1 aggressiveness level
        provider: LLM provider for accurate token counting ("gemini" or "openai")

    Returns:
        Tuple of (compressed_prompt, was_compressed, compression_result)
    """
    token_count = count_tokens(prompt, provider)

    # Skip compression for short prompts
    if token_count < 10:
        result = CompressionResult(
            original=prompt,
            compressed=prompt,
            original_tokens=token_count,
            compressed_tokens=token_count
        )
        return prompt, False, result

    # Compress with bear-1
    result = compress(prompt, aggressiveness=aggressiveness, provider=provider)

    # Only use compressed version if it's actually shorter
    if result.compressed_tokens < result.original_tokens:
        return result.compressed, True, result
    else:
        return prompt, False, result
