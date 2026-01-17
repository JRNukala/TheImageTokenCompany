"""
OCR module using EasyOCR.
"""
from typing import List, Optional
import numpy as np

# Lazy-loaded reader
_reader = None


def _get_reader():
    """Lazy load EasyOCR reader."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def extract_text(image: np.ndarray) -> str:
    """
    Extract text from image using OCR.

    Args:
        image: Image as numpy array (BGR or RGB)

    Returns:
        Compact string of extracted text, e.g. "text:STOP Welcome"
    """
    try:
        reader = _get_reader()
        results = reader.readtext(image)

        if not results:
            return ""

        # Extract just the text, join with spaces
        texts = [result[1] for result in results if result[2] > 0.3]  # confidence > 0.3

        if not texts:
            return ""

        combined = " ".join(texts)
        return f"text:{combined}"

    except Exception as e:
        print(f"[OCR] Failed: {e}")
        return ""


def extract_text_with_boxes(image: np.ndarray) -> List[dict]:
    """
    Extract text with bounding boxes (for layout analysis).

    Args:
        image: Image as numpy array

    Returns:
        List of dicts with 'text', 'box', 'confidence'
    """
    try:
        reader = _get_reader()
        results = reader.readtext(image)

        return [
            {
                "text": result[1],
                "box": result[0],
                "confidence": result[2]
            }
            for result in results
            if result[2] > 0.3
        ]

    except Exception as e:
        print(f"[OCR] Failed: {e}")
        return []
