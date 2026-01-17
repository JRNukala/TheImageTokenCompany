"""
Step 4: Vision Pipeline Dispatcher.

Executes ONLY the vision modules specified by CVspec.
"""
from typing import List
import numpy as np

from pipeline.step3_cvspec import CVSpec


def run_vision_pipeline(image: np.ndarray, cvspec: CVSpec) -> str:
    """
    Run vision analysis based on CVspec.

    Args:
        image: Image as numpy array (BGR format from cv2)
        cvspec: CVSpec indicating which modules to run

    Returns:
        Compact imgDescr string containing only relevant information
    """
    outputs: List[str] = []

    # OCR
    if cvspec.ocr:
        try:
            from vision_modules.ocr import extract_text
            result = extract_text(image)
            if result:
                outputs.append(result)
        except Exception as e:
            print(f"[Vision] OCR failed: {e}")

    # Objects
    if cvspec.objects:
        try:
            from vision_modules.objects import detect_objects
            result = detect_objects(image)
            if result:
                outputs.append(result)
        except Exception as e:
            print(f"[Vision] Objects failed: {e}")

    # Scene
    if cvspec.scene:
        try:
            from vision_modules.scene import classify_scene
            result = classify_scene(image)
            if result:
                outputs.append(result)
        except Exception as e:
            print(f"[Vision] Scene failed: {e}")

    # Person analysis
    person_spec = cvspec.person
    if person_spec.detect or person_spec.count or person_spec.expression or \
       person_spec.clothing_color or person_spec.clothing_type:
        try:
            from vision_modules.person import analyze_persons
            result = analyze_persons(
                image,
                detect=person_spec.detect or person_spec.expression or \
                       person_spec.clothing_color or person_spec.clothing_type,
                count=person_spec.count,
                expression=person_spec.expression,
                clothing_color=person_spec.clothing_color,
                clothing_type=person_spec.clothing_type
            )
            if result:
                outputs.append(result)
        except Exception as e:
            print(f"[Vision] Person analysis failed: {e}")

    # Colors
    if cvspec.colors:
        try:
            from vision_modules.colors import extract_dominant_colors
            result = extract_dominant_colors(image)
            if result:
                outputs.append(result)
        except Exception as e:
            print(f"[Vision] Colors failed: {e}")

    # Layout
    if cvspec.layout:
        try:
            from vision_modules.layout import analyze_layout
            result = analyze_layout(image)
            if result:
                outputs.append(result)
        except Exception as e:
            print(f"[Vision] Layout failed: {e}")

    # Join all outputs
    return " | ".join(outputs) if outputs else ""


def get_active_modules(cvspec: CVSpec) -> List[str]:
    """
    Get list of active module names for verbose output.

    Args:
        cvspec: CVSpec to check

    Returns:
        List of active module names
    """
    modules = []

    if cvspec.ocr:
        modules.append("ocr")
    if cvspec.objects:
        modules.append("objects")
    if cvspec.scene:
        modules.append("scene")
    if cvspec.person.detect:
        modules.append("person.detect")
    if cvspec.person.count:
        modules.append("person.count")
    if cvspec.person.expression:
        modules.append("person.expression")
    if cvspec.person.clothing_color:
        modules.append("person.clothing_color")
    if cvspec.person.clothing_type:
        modules.append("person.clothing_type")
    if cvspec.colors:
        modules.append("colors")
    if cvspec.layout:
        modules.append("layout")

    return modules
