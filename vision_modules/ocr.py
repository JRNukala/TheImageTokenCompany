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
    Extract text from image using OCR, preserving line structure.

    Args:
        image: Image as numpy array (BGR or RGB)

    Returns:
        Compact string of extracted text with lines separated by ' | '
        e.g. "text:Question 1 | A) Option A | B) Option B | C) Option C"
    """
    try:
        reader = _get_reader()
        results = reader.readtext(image)

        if not results:
            return ""

        # Filter by confidence
        filtered = [(result[0], result[1], result[2]) for result in results if result[2] > 0.3]

        if not filtered:
            return ""

        # Group text by vertical position (Y coordinate) to detect lines
        # Each result[0] is a list of 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Use the average Y of top-left and top-right corners as the line position
        line_items = []
        for box, text, conf in filtered:
            # Get average Y of top edge (top-left and top-right corners)
            avg_y = (box[0][1] + box[1][1]) / 2
            # Get X position for sorting within line
            x_pos = box[0][0]
            line_items.append((avg_y, x_pos, text))

        # Sort by Y position first, then X position
        line_items.sort(key=lambda x: (x[0], x[1]))

        # Group into lines based on Y proximity (threshold = 20 pixels)
        lines = []
        current_line = []
        current_y = None
        line_threshold = 20  # pixels

        for avg_y, x_pos, text in line_items:
            if current_y is None or abs(avg_y - current_y) <= line_threshold:
                current_line.append((x_pos, text))
                if current_y is None:
                    current_y = avg_y
                else:
                    current_y = (current_y + avg_y) / 2  # Running average
            else:
                # New line detected
                if current_line:
                    # Sort items in line by X position and join
                    current_line.sort(key=lambda x: x[0])
                    lines.append(" ".join(item[1] for item in current_line))
                current_line = [(x_pos, text)]
                current_y = avg_y

        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append(" ".join(item[1] for item in current_line))

        # Join lines with ' | ' separator
        combined = " | ".join(lines)
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
