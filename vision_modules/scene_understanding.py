"""
Scene understanding using local SmolVLM-256M.

Provides semantic descriptions of actions, emotions, and context.
Runs 100% locally - no API calls.
"""
import numpy as np
from PIL import Image

_model = None
_processor = None


def _load_model():
    """Lazy load SmolVLM model."""
    global _model, _processor
    if _model is None:
        print("[Understanding] Loading SmolVLM-256M model (first time may take a moment)...")
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

        _processor = AutoProcessor.from_pretrained(model_id)
        _model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # CPU-friendly
            device_map="auto"
        )
        print("[Understanding] Model loaded.")
    return _model, _processor


def analyze_scene(image: np.ndarray, focus_hint: str = "") -> str:
    """
    Get semantic scene understanding using local SmolVLM.

    Args:
        image: Image as numpy array (BGR from cv2)
        focus_hint: Optional hint about what aspects to focus on

    Returns:
        Compact scene description (actions, emotions, context)
    """
    try:
        model, processor = _load_model()

        # Convert BGR numpy to RGB PIL
        pil_image = Image.fromarray(image[:, :, ::-1])

        # Create prompt with optional focus hint
        if focus_hint:
            prompt_text = f"Describe this scene, focusing specifically on: {focus_hint}. Be expressive but concise (30-40 words):"
        else:
            prompt_text = "Describe the key visual elements, actions, emotions, and atmosphere in this scene. Focus on important details that help understand what's happening. Be expressive but concise (30-40 words):"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # Process
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[pil_image], return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate (increased token limit for more expressive output)
        outputs = model.generate(**inputs, max_new_tokens=100)
        result = processor.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response (after the prompt)
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()

        return f"context:{result}"

    except Exception as e:
        print(f"[Understanding] Failed: {e}")
        return ""
