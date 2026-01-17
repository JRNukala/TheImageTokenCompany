"""
Color extraction module using K-means clustering.
"""
from typing import List, Tuple
import numpy as np
import cv2

# Color name mapping (RGB ranges to names)
COLOR_NAMES = {
    "red": [(150, 0, 0), (255, 100, 100)],
    "orange": [(200, 100, 0), (255, 180, 100)],
    "yellow": [(200, 200, 0), (255, 255, 150)],
    "green": [(0, 100, 0), (150, 255, 150)],
    "blue": [(0, 0, 150), (150, 150, 255)],
    "purple": [(100, 0, 100), (200, 100, 200)],
    "pink": [(200, 100, 150), (255, 200, 220)],
    "brown": [(100, 50, 0), (180, 120, 80)],
    "black": [(0, 0, 0), (50, 50, 50)],
    "white": [(200, 200, 200), (255, 255, 255)],
    "gray": [(80, 80, 80), (180, 180, 180)],
    "beige": [(180, 160, 120), (240, 220, 180)],
    "navy": [(0, 0, 80), (60, 60, 150)],
    "teal": [(0, 100, 100), (100, 180, 180)],
    "maroon": [(80, 0, 0), (150, 50, 50)],
}


def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Map RGB value to closest color name.

    Args:
        rgb: (R, G, B) tuple

    Returns:
        Color name string
    """
    r, g, b = rgb

    # Check each color range
    best_match = "unknown"
    best_dist = float("inf")

    for name, (low, high) in COLOR_NAMES.items():
        # Check if in range
        if (low[0] <= r <= high[0] and
            low[1] <= g <= high[1] and
            low[2] <= b <= high[2]):
            # Calculate distance to center of range
            center = ((low[0] + high[0]) / 2,
                      (low[1] + high[1]) / 2,
                      (low[2] + high[2]) / 2)
            dist = ((r - center[0])**2 + (g - center[1])**2 + (b - center[2])**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = name

    # If no exact match, find closest by Euclidean distance
    if best_match == "unknown":
        for name, (low, high) in COLOR_NAMES.items():
            center = ((low[0] + high[0]) / 2,
                      (low[1] + high[1]) / 2,
                      (low[2] + high[2]) / 2)
            dist = ((r - center[0])**2 + (g - center[1])**2 + (b - center[2])**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = name

    return best_match


def extract_dominant_colors(image: np.ndarray, n_colors: int = 3) -> str:
    """
    Extract dominant colors from image using K-means.

    Args:
        image: Image as numpy array (BGR)
        n_colors: Number of dominant colors to extract

    Returns:
        Compact string, e.g. "colors:blue,white,gray"
    """
    try:
        # Convert to RGB if BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Resize for faster processing
        small = cv2.resize(image_rgb, (100, 100))

        # Reshape to list of pixels
        pixels = small.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Sort by frequency
        label_counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(-label_counts)

        # Map to color names
        color_names = []
        for idx in sorted_indices:
            rgb = tuple(map(int, centers[idx]))
            name = rgb_to_color_name(rgb)
            if name not in color_names:  # avoid duplicates
                color_names.append(name)

        if not color_names:
            return ""

        return f"colors:{','.join(color_names)}"

    except Exception as e:
        print(f"[Colors] Extraction failed: {e}")
        return ""


def extract_region_color(image: np.ndarray, box: Tuple[int, int, int, int]) -> str:
    """
    Extract dominant color from a specific region.

    Args:
        image: Image as numpy array (BGR)
        box: Bounding box as (x1, y1, x2, y2)

    Returns:
        Color name string (e.g., "red")
    """
    try:
        x1, y1, x2, y2 = box
        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return "unknown"

        # Convert to RGB
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

        # Get average color (simpler than k-means for small regions)
        avg_color = tuple(map(int, np.mean(region_rgb.reshape(-1, 3), axis=0)))

        return rgb_to_color_name(avg_color)

    except Exception as e:
        print(f"[Colors] Region extraction failed: {e}")
        return "unknown"
