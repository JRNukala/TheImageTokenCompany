"""
Document layout analysis module.
"""
from typing import List, Dict
import numpy as np


def analyze_layout(image: np.ndarray, ocr_results: List[Dict] = None) -> str:
    """
    Analyze document layout based on OCR bounding boxes.

    Args:
        image: Image as numpy array
        ocr_results: Optional pre-computed OCR results with boxes

    Returns:
        Compact string, e.g. "layout:headers,bullets"
    """
    try:
        # Get OCR results if not provided
        if ocr_results is None:
            from vision_modules.ocr import extract_text_with_boxes
            ocr_results = extract_text_with_boxes(image)

        if not ocr_results:
            return ""

        h, w = image.shape[:2]

        # Analyze layout features
        features = set()

        # Group text by vertical position (lines)
        lines = []
        current_line = []
        last_y = -1

        # Sort by y-coordinate
        sorted_results = sorted(ocr_results, key=lambda x: min(p[1] for p in x["box"]))

        for item in sorted_results:
            box = item["box"]
            y_center = sum(p[1] for p in box) / 4

            if last_y < 0 or abs(y_center - last_y) < h * 0.03:
                current_line.append(item)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [item]
            last_y = y_center

        if current_line:
            lines.append(current_line)

        # Detect headers (large text at top)
        if lines:
            first_line = lines[0]
            first_y = min(min(p[1] for p in item["box"]) for item in first_line)
            if first_y < h * 0.15:
                features.add("header")

        # Detect bullets/lists
        bullet_chars = {"•", "-", "*", "○", "►", "→"}
        for item in ocr_results:
            text = item["text"].strip()
            if text and text[0] in bullet_chars:
                features.add("bullets")
                break
            # Check for numbered lists
            if text and len(text) > 1 and text[0].isdigit() and text[1] in ".):":
                features.add("numbered")
                break

        # Detect table-like structure (aligned columns)
        if len(lines) >= 3:
            # Check if items are aligned across lines
            x_positions = []
            for line in lines[:5]:  # Check first 5 lines
                for item in line:
                    x_center = sum(p[0] for p in item["box"]) / 4
                    x_positions.append(x_center)

            if x_positions:
                # Check for column alignment
                x_positions.sort()
                column_gaps = []
                for i in range(len(x_positions) - 1):
                    gap = x_positions[i + 1] - x_positions[i]
                    if gap > w * 0.1:
                        column_gaps.append(gap)

                if len(column_gaps) >= 2:
                    features.add("table")

        # Detect paragraphs (multiple lines of similar width)
        if len(lines) >= 2:
            line_widths = []
            for line in lines:
                if line:
                    min_x = min(min(p[0] for p in item["box"]) for item in line)
                    max_x = max(max(p[0] for p in item["box"]) for item in line)
                    line_widths.append(max_x - min_x)

            if line_widths:
                avg_width = sum(line_widths) / len(line_widths)
                similar_widths = sum(1 for lw in line_widths if abs(lw - avg_width) < w * 0.1)
                if similar_widths >= len(lines) * 0.7:
                    features.add("paragraphs")

        if not features:
            features.add("text")

        return f"layout:{','.join(sorted(features))}"

    except Exception as e:
        print(f"[Layout] Analysis failed: {e}")
        return ""
