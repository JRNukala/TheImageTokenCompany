"""
Step 1: Input Handling.

Validates and loads image and prompt inputs.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class PipelineInput:
    """Validated pipeline inputs."""
    image: np.ndarray  # BGR format for OpenCV
    image_path: str
    prompt: str


def load_inputs(image_path: Union[str, Path], prompt: str) -> PipelineInput:
    """
    Validate and load pipeline inputs.

    Args:
        image_path: Path to image file
        prompt: User's question about the image

    Returns:
        PipelineInput with loaded image and validated prompt

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If prompt is empty or image can't be loaded
    """
    # Validate prompt
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    prompt = prompt.strip()

    # Validate image path
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Check file extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    # Load image with OpenCV (returns BGR)
    image = cv2.imread(str(image_path))

    if image is None:
        # Try with PIL as fallback
        try:
            pil_image = Image.open(image_path).convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    if image is None or image.size == 0:
        raise ValueError(f"Failed to load image: {image_path}")

    return PipelineInput(
        image=image,
        image_path=str(image_path),
        prompt=prompt
    )


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic image information.

    Args:
        image: Image as numpy array

    Returns:
        Dict with height, width, channels
    """
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) > 2 else 1

    return {
        "height": h,
        "width": w,
        "channels": c,
        "size": f"{w}x{h}"
    }
