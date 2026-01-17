"""
Person detection and attribute extraction module.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

# Lazy-loaded models
_face_mesh = None


def _get_face_mesh():
    """Lazy load MediaPipe face mesh."""
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            min_detection_confidence=0.5
        )
    return _face_mesh


# Simple clothing type classifier based on aspect ratio and position
CLOTHING_TYPES = ["t-shirt", "shirt", "dress", "jacket", "hoodie", "sweater"]


def detect_persons(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect persons in image and return bounding boxes.

    Args:
        image: Image as numpy array

    Returns:
        List of bounding boxes as (x1, y1, x2, y2)
    """
    from vision_modules.objects import detect_persons as yolo_detect_persons
    return yolo_detect_persons(image)


def count_persons(image: np.ndarray) -> str:
    """
    Count number of persons in image.

    Args:
        image: Image as numpy array

    Returns:
        Compact string, e.g. "people:3"
    """
    boxes = detect_persons(image)
    return f"people:{len(boxes)}"


def analyze_expression(image: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None) -> str:
    """
    Analyze facial expression.

    Args:
        image: Image as numpy array (RGB)
        face_box: Optional face bounding box

    Returns:
        Expression string (happy, sad, neutral, surprised, angry)
    """
    try:
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        face_mesh = _get_face_mesh()
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return "neutral"

        # Simple heuristic based on mouth and eyebrow positions
        landmarks = results.multi_face_landmarks[0].landmark

        # Get key points (simplified expression detection)
        # Mouth corners (61, 291)
        # Upper lip (13), Lower lip (14)
        # Eyebrows (70, 300)

        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        # Calculate mouth openness
        mouth_open = abs(upper_lip.y - lower_lip.y)

        # Calculate smile (mouth corner heights relative to center)
        mouth_center_y = (left_mouth.y + right_mouth.y) / 2
        mouth_width = abs(right_mouth.x - left_mouth.x)

        # Simple classification
        if mouth_open > 0.05:
            return "surprised"
        elif mouth_width > 0.15 and left_mouth.y < upper_lip.y and right_mouth.y < upper_lip.y:
            return "happy"
        elif left_mouth.y > upper_lip.y and right_mouth.y > upper_lip.y:
            return "sad"
        else:
            return "neutral"

    except Exception as e:
        print(f"[Person] Expression analysis failed: {e}")
        return "neutral"


def analyze_clothing(image: np.ndarray, person_box: Tuple[int, int, int, int]) -> Dict[str, str]:
    """
    Analyze clothing color and type for a person.

    Args:
        image: Image as numpy array (BGR)
        person_box: Person bounding box (x1, y1, x2, y2)

    Returns:
        Dict with 'color' and 'type' keys
    """
    from vision_modules.colors import extract_region_color

    try:
        x1, y1, x2, y2 = person_box
        height = y2 - y1
        width = x2 - x1

        # Upper body region (torso area - middle 40-80% of height)
        torso_y1 = y1 + int(height * 0.2)  # Below head
        torso_y2 = y1 + int(height * 0.6)  # Above legs
        torso_x1 = x1 + int(width * 0.1)
        torso_x2 = x2 - int(width * 0.1)

        torso_box = (torso_x1, torso_y1, torso_x2, torso_y2)

        # Extract color from torso region
        color = extract_region_color(image, torso_box)

        # Simple clothing type heuristic based on aspect ratio
        torso_height = torso_y2 - torso_y1
        torso_width = torso_x2 - torso_x1

        aspect = torso_height / max(torso_width, 1)

        # Guess clothing type (simplified)
        if aspect > 1.5:
            clothing_type = "dress"
        elif aspect > 1.0:
            clothing_type = "shirt"
        else:
            clothing_type = "t-shirt"

        return {"color": color, "type": clothing_type}

    except Exception as e:
        print(f"[Person] Clothing analysis failed: {e}")
        return {"color": "unknown", "type": "unknown"}


def analyze_persons(image: np.ndarray, detect: bool = True, count: bool = False,
                    expression: bool = False, clothing_color: bool = False,
                    clothing_type: bool = False) -> str:
    """
    Full person analysis based on CVspec requirements.

    Args:
        image: Image as numpy array
        detect: Whether to detect persons
        count: Whether to count persons
        expression: Whether to analyze expressions
        clothing_color: Whether to extract clothing colors
        clothing_type: Whether to identify clothing types

    Returns:
        Compact description string
    """
    if not detect and not count:
        return ""

    boxes = detect_persons(image)

    if not boxes:
        if count:
            return "people:0"
        return ""

    outputs = []

    # Count
    if count:
        outputs.append(f"people:{len(boxes)}")

    # Per-person analysis
    for i, box in enumerate(boxes, 1):
        person_parts = []

        if expression:
            # Crop face region (upper portion of person box)
            x1, y1, x2, y2 = box
            face_region = image[y1:y1 + (y2 - y1) // 3, x1:x2]
            if face_region.size > 0:
                expr = analyze_expression(face_region)
                person_parts.append(f"expression:{expr}")

        if clothing_color or clothing_type:
            clothing = analyze_clothing(image, box)
            if clothing_color:
                person_parts.append(clothing["color"])
            if clothing_type:
                person_parts.append(clothing["type"])

        if person_parts:
            outputs.append(f"person{i} {' '.join(person_parts)}")

    return " | ".join(outputs)
