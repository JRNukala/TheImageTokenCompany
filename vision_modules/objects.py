"""
Object detection module using YOLOv8.
"""
from typing import List, Tuple
import numpy as np

# Lazy-loaded model
_model = None


def _get_model():
    """Lazy load YOLOv8 model."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        # Use nano model for speed (CPU-friendly)
        _model = YOLO("yolov8n.pt")
    return _model


def detect_objects(image: np.ndarray, confidence: float = 0.3) -> str:
    """
    Detect objects in image.

    Args:
        image: Image as numpy array
        confidence: Minimum confidence threshold

    Returns:
        Compact string of objects, e.g. "objects:chair,table,laptop,cup"
    """
    try:
        model = _get_model()
        results = model(image, verbose=False, conf=confidence)

        if not results or len(results) == 0:
            return ""

        # Get unique class names
        names = results[0].names
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return ""

        # Collect unique object names
        detected = set()
        for box in boxes:
            cls_id = int(box.cls[0])
            detected.add(names[cls_id])

        if not detected:
            return ""

        return f"objects:{','.join(sorted(detected))}"

    except Exception as e:
        print(f"[Objects] Failed: {e}")
        return ""


def detect_objects_with_boxes(image: np.ndarray, confidence: float = 0.3) -> List[dict]:
    """
    Detect objects with bounding boxes.

    Args:
        image: Image as numpy array
        confidence: Minimum confidence threshold

    Returns:
        List of dicts with 'name', 'box', 'confidence'
    """
    try:
        model = _get_model()
        results = model(image, verbose=False, conf=confidence)

        if not results or len(results) == 0:
            return []

        names = results[0].names
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "name": names[cls_id],
                "box": xyxy,  # [x1, y1, x2, y2]
                "confidence": conf
            })

        return detections

    except Exception as e:
        print(f"[Objects] Failed: {e}")
        return []


def detect_persons(image: np.ndarray, confidence: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """
    Detect persons and return bounding boxes.

    Args:
        image: Image as numpy array
        confidence: Minimum confidence threshold

    Returns:
        List of bounding boxes as (x1, y1, x2, y2) tuples
    """
    try:
        model = _get_model()
        results = model(image, verbose=False, conf=confidence, classes=[0])  # class 0 = person

        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        return [tuple(map(int, box.xyxy[0].tolist())) for box in boxes]

    except Exception as e:
        print(f"[Objects] Person detection failed: {e}")
        return []
