"""
Step 5: Final Compression with bear-1.

Compresses both vision output and user prompt.
"""
from dataclasses import dataclass

from utils.bear1 import compress, CompressionResult
from utils.tokens import count_tokens


@dataclass
class CompressionStats:
    """Statistics from compression step."""
    original_img_tokens: int
    compressed_img_tokens: int
    original_txt_tokens: int
    compressed_txt_tokens: int
    total_original: int
    total_compressed: int

    @property
    def savings_percent(self) -> float:
        if self.total_original == 0:
            return 0.0
        return (1 - self.total_compressed / self.total_original) * 100


def compress_outputs(
    img_descr: str,
    txt: str,
    txt_already_compressed: bool = False,
    aggressiveness: float = 0.7,
    provider: str = "gemini"
) -> tuple[str, str, CompressionStats]:
    """
    Compress vision output and prompt using bear-1.

    Args:
        img_descr: Vision output from Step 4
        txt: User's prompt (may already be compressed)
        txt_already_compressed: Whether prompt was compressed in Step 2
        aggressiveness: bear-1 aggressiveness (0.0-1.0)
        provider: LLM provider for accurate token counting ("gemini" or "openai")

    Returns:
        Tuple of (compressed_img_descr, compressed_txt, stats)
    """
    # Compress image description
    img_result = compress(img_descr, aggressiveness=aggressiveness, provider=provider)

    # Compress text only if not already compressed
    if txt_already_compressed:
        txt_result = CompressionResult(
            original=txt,
            compressed=txt,
            original_tokens=count_tokens(txt, provider),
            compressed_tokens=count_tokens(txt, provider)
        )
    else:
        txt_result = compress(txt, aggressiveness=aggressiveness, provider=provider)

    stats = CompressionStats(
        original_img_tokens=img_result.original_tokens,
        compressed_img_tokens=img_result.compressed_tokens,
        original_txt_tokens=txt_result.original_tokens,
        compressed_txt_tokens=txt_result.compressed_tokens,
        total_original=img_result.original_tokens + txt_result.original_tokens,
        total_compressed=img_result.compressed_tokens + txt_result.compressed_tokens
    )

    return img_result.compressed, txt_result.compressed, stats
