"""
Step 4: Vision Pipeline Dispatcher.

Executes ONLY the vision modules specified by CVspec.
"""
from typing import List
import numpy as np

from pipeline.step3_cvspec import CVSpec


def run_vision_pipeline(image: np.ndarray, cvspec: CVSpec, user_prompt: str = "") -> str:
    """
    Run vision analysis based on CVspec.

    Args:
        image: Image as numpy array (BGR format from cv2)
        cvspec: CVSpec indicating which modules to run
        user_prompt: Original user prompt (fallback for SmolVLM if focus hint is empty)

    Returns:
        Compact imgDescr string containing only relevant information
    """
    outputs: List[str] = []

    # Debug: Print image info
    print(f"  [Debug] Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"  [Debug] CVSpec modules requested: ocr={cvspec.ocr}, objects={cvspec.objects}, scene={cvspec.scene}, person.detect={cvspec.person.detect}, colors={cvspec.colors}, layout={cvspec.layout}, understanding={cvspec.understanding}")

    # OCR (EasyOCR - local)
    if cvspec.ocr:
        print(f"  [Debug] Running OCR module (EasyOCR)...")
        try:
            from vision_modules.ocr import extract_text
            result = extract_text(image)
            print(f"  [OCR] Raw result: '{result}'")
            if result:
                print(f"  [OCR] → {result}")
                outputs.append(result)
            else:
                print(f"  [OCR] → No text detected")
        except Exception as e:
            print(f"[Vision] OCR failed: {e}")
            import traceback
            traceback.print_exc()

    # Objects (YOLO - local)
    if cvspec.objects:
        print(f"  [Debug] Running Objects module (YOLO)...")
        try:
            from vision_modules.objects import detect_objects
            result = detect_objects(image)
            print(f"  [Objects] Raw result: '{result}'")
            if result:
                print(f"  [Objects] → {result}")
                outputs.append(result)
            else:
                print(f"  [Objects] → No objects detected")
        except Exception as e:
            print(f"[Vision] Objects failed: {e}")
            import traceback
            traceback.print_exc()

    # Scene (local classifier)
    if cvspec.scene:
        print(f"  [Debug] Running Scene module...")
        try:
            from vision_modules.scene import classify_scene
            result = classify_scene(image)
            print(f"  [Scene] Raw result: '{result}'")
            if result:
                print(f"  [Scene] → {result}")
                outputs.append(result)
            else:
                print(f"  [Scene] → No scene classification")
        except Exception as e:
            print(f"[Vision] Scene failed: {e}")
            import traceback
            traceback.print_exc()

    # Person analysis (MediaPipe - local)
    person_spec = cvspec.person
    if person_spec.detect or person_spec.count or person_spec.expression or \
       person_spec.clothing_color or person_spec.clothing_type:
        print(f"  [Debug] Running Person module (MediaPipe)...")
        print(f"  [Debug] Person options: detect={person_spec.detect}, count={person_spec.count}, expression={person_spec.expression}, clothing_color={person_spec.clothing_color}, clothing_type={person_spec.clothing_type}")
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
            print(f"  [Person] Raw result: '{result}'")
            if result:
                print(f"  [Person] → {result}")
                outputs.append(result)
            else:
                print(f"  [Person] → No persons detected")
        except Exception as e:
            print(f"[Vision] Person analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Colors (local)
    if cvspec.colors:
        print(f"  [Debug] Running Colors module...")
        try:
            from vision_modules.colors import extract_dominant_colors
            result = extract_dominant_colors(image)
            print(f"  [Colors] Raw result: '{result}'")
            if result:
                print(f"  [Colors] → {result}")
                outputs.append(result)
            else:
                print(f"  [Colors] → No colors extracted")
        except Exception as e:
            print(f"[Vision] Colors failed: {e}")
            import traceback
            traceback.print_exc()

    # Layout (local)
    if cvspec.layout:
        print(f"  [Debug] Running Layout module...")
        try:
            from vision_modules.layout import analyze_layout
            result = analyze_layout(image)
            print(f"  [Layout] Raw result: '{result}'")
            if result:
                print(f"  [Layout] → {result}")
                outputs.append(result)
            else:
                print(f"  [Layout] → No layout detected")
        except Exception as e:
            print(f"[Vision] Layout failed: {e}")
            import traceback
            traceback.print_exc()

    # Scene Understanding (Local SmolVLM-256M)
    if cvspec.understanding:
        focus_hint = cvspec.understanding_focus if hasattr(cvspec, 'understanding_focus') else ""

        # If no focus hint from Gemini, use user prompt as fallback
        if not focus_hint and user_prompt:
            focus_hint = user_prompt
            print(f"  [Debug] Running Understanding module (SmolVLM-256M local) with user prompt as focus: '{focus_hint[:50]}...'")
        elif focus_hint:
            print(f"  [Debug] Running Understanding module (SmolVLM-256M local) with Gemini focus: '{focus_hint}'...")
        else:
            print(f"  [Debug] Running Understanding module (SmolVLM-256M local) with default prompt...")

        try:
            from vision_modules.scene_understanding import analyze_scene
            result = analyze_scene(image, focus_hint=focus_hint)
            print(f"  [Understanding] Raw result: '{result}'")
            if result:
                print(f"  [Understanding] → {result}")
                outputs.append(result)
            else:
                print(f"  [Understanding] → No understanding output")
        except Exception as e:
            print(f"[Vision] Scene understanding failed: {e}")
            import traceback
            traceback.print_exc()

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
    if cvspec.understanding:
        modules.append("understanding")

    return modules
