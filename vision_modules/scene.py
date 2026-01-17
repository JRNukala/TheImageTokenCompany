"""
Scene classification module.

Uses a simple heuristic approach based on detected objects,
or CLIP zero-shot classification if available.
"""
from typing import Optional
import numpy as np

# Scene categories
SCENE_CATEGORIES = [
    "indoor", "outdoor", "office", "kitchen", "bedroom", "bathroom",
    "living room", "street", "park", "beach", "restaurant", "store",
    "vehicle", "sports field", "classroom", "library"
]

# Object to scene mapping heuristics
OBJECT_SCENE_MAP = {
    "car": "street",
    "truck": "street",
    "bus": "street",
    "bicycle": "outdoor",
    "motorcycle": "street",
    "traffic light": "street",
    "stop sign": "street",
    "parking meter": "street",
    "bench": "outdoor",
    "bird": "outdoor",
    "cat": "indoor",
    "dog": "outdoor",
    "horse": "outdoor",
    "cow": "outdoor",
    "elephant": "outdoor",
    "bear": "outdoor",
    "zebra": "outdoor",
    "giraffe": "outdoor",
    "backpack": "indoor",
    "umbrella": "outdoor",
    "handbag": "indoor",
    "tie": "office",
    "suitcase": "indoor",
    "frisbee": "park",
    "skis": "outdoor",
    "snowboard": "outdoor",
    "sports ball": "sports field",
    "kite": "outdoor",
    "baseball bat": "sports field",
    "baseball glove": "sports field",
    "skateboard": "outdoor",
    "surfboard": "beach",
    "tennis racket": "sports field",
    "bottle": "indoor",
    "wine glass": "restaurant",
    "cup": "indoor",
    "fork": "kitchen",
    "knife": "kitchen",
    "spoon": "kitchen",
    "bowl": "kitchen",
    "banana": "kitchen",
    "apple": "kitchen",
    "sandwich": "kitchen",
    "orange": "kitchen",
    "broccoli": "kitchen",
    "carrot": "kitchen",
    "hot dog": "kitchen",
    "pizza": "restaurant",
    "donut": "kitchen",
    "cake": "kitchen",
    "chair": "indoor",
    "couch": "living room",
    "potted plant": "indoor",
    "bed": "bedroom",
    "dining table": "kitchen",
    "toilet": "bathroom",
    "tv": "living room",
    "laptop": "office",
    "mouse": "office",
    "remote": "living room",
    "keyboard": "office",
    "cell phone": "indoor",
    "microwave": "kitchen",
    "oven": "kitchen",
    "toaster": "kitchen",
    "sink": "kitchen",
    "refrigerator": "kitchen",
    "book": "library",
    "clock": "indoor",
    "vase": "indoor",
    "scissors": "office",
    "teddy bear": "bedroom",
    "hair drier": "bathroom",
    "toothbrush": "bathroom",
}


def classify_scene(image: np.ndarray, detected_objects: Optional[list] = None) -> str:
    """
    Classify the scene type.

    Args:
        image: Image as numpy array
        detected_objects: Optional list of already-detected objects

    Returns:
        Compact string, e.g. "scene:office"
    """
    # If we have detected objects, use heuristics
    if detected_objects:
        scene_votes = {}
        for obj in detected_objects:
            obj_name = obj if isinstance(obj, str) else obj.get("name", "")
            if obj_name in OBJECT_SCENE_MAP:
                scene = OBJECT_SCENE_MAP[obj_name]
                scene_votes[scene] = scene_votes.get(scene, 0) + 1

        if scene_votes:
            best_scene = max(scene_votes, key=scene_votes.get)
            return f"scene:{best_scene}"

    # Fallback: try to detect objects and classify
    try:
        from vision_modules.objects import detect_objects_with_boxes
        detections = detect_objects_with_boxes(image)

        if detections:
            scene_votes = {}
            for det in detections:
                obj_name = det["name"]
                if obj_name in OBJECT_SCENE_MAP:
                    scene = OBJECT_SCENE_MAP[obj_name]
                    scene_votes[scene] = scene_votes.get(scene, 0) + 1

            if scene_votes:
                best_scene = max(scene_votes, key=scene_votes.get)
                return f"scene:{best_scene}"

        # If we have objects but no scene mapping, guess indoor/outdoor
        outdoor_objects = {"car", "truck", "bus", "bicycle", "traffic light", "bench", "bird"}
        detected_names = {d["name"] for d in detections}

        if detected_names & outdoor_objects:
            return "scene:outdoor"
        elif detections:
            return "scene:indoor"

    except Exception as e:
        print(f"[Scene] Classification failed: {e}")

    return "scene:unknown"
