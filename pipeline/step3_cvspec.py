"""
Step 3: Generate CVspec via Small LLM.

Analyzes user prompt to determine which vision modules are needed.
Supports both Gemini (default) and OpenAI.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# OpenAI client (lazy-loaded)
_openai_client = None


def _get_openai_client():
    """Lazy load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


@dataclass
class PersonSpec:
    """Specification for person-related vision tasks."""
    detect: bool = False
    expression: bool = False
    clothing_color: bool = False
    clothing_type: bool = False
    count: bool = False


@dataclass
class CVSpec:
    """Computer Vision specification - which modules to run."""
    ocr: bool = False
    objects: bool = False
    scene: bool = False
    person: PersonSpec = field(default_factory=PersonSpec)
    colors: bool = False
    layout: bool = False
    understanding: bool = False  # Semantic scene understanding (actions, emotions, context)

    def to_dict(self) -> dict:
        return {
            "ocr": self.ocr,
            "objects": self.objects,
            "scene": self.scene,
            "person": {
                "detect": self.person.detect,
                "expression": self.person.expression,
                "clothing_color": self.person.clothing_color,
                "clothing_type": self.person.clothing_type,
                "count": self.person.count
            },
            "colors": self.colors,
            "layout": self.layout,
            "understanding": self.understanding
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CVSpec":
        person_data = data.get("person", {})
        person_spec = PersonSpec(
            detect=person_data.get("detect", False),
            expression=person_data.get("expression", False),
            clothing_color=person_data.get("clothing_color", False),
            clothing_type=person_data.get("clothing_type", False),
            count=person_data.get("count", False)
        )
        return cls(
            ocr=data.get("ocr", False),
            objects=data.get("objects", False),
            scene=data.get("scene", False),
            person=person_spec,
            colors=data.get("colors", False),
            layout=data.get("layout", False),
            understanding=data.get("understanding", False)
        )

    @classmethod
    def all_enabled(cls) -> "CVSpec":
        """Return CVSpec with all modules enabled (fallback)."""
        return cls(
            ocr=True,
            objects=True,
            scene=True,
            person=PersonSpec(
                detect=True,
                expression=True,
                clothing_color=True,
                clothing_type=True,
                count=True
            ),
            colors=True,
            layout=True,
            understanding=True
        )


SYSTEM_PROMPT = """You are a vision routing assistant. Given a user's question about an image, determine which computer vision modules are needed to answer it.

Output ONLY a JSON object with this schema (all values are booleans):
{
  "ocr": false,           // Need to extract text from image?
  "objects": false,       // Need to detect and list objects?
  "scene": false,         // Need to classify scene type (indoor/outdoor/etc)?
  "person": {
    "detect": false,      // Need to find people?
    "expression": false,  // Need facial expressions (happy/sad/etc)?
    "clothing_color": false, // Need clothing colors?
    "clothing_type": false,  // Need clothing types (shirt/dress/etc)?
    "count": false        // Need to count people?
  },
  "colors": false,        // Need overall dominant colors of image?
  "layout": false,        // Need document layout analysis?
  "understanding": false  // Need to understand actions/emotions/what's happening?
}

Examples:

User: "What does the sign say?"
{"ocr": true, "objects": false, "scene": false, "person": {"detect": false, "expression": false, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": false, "understanding": false}

User: "How many people are there?"
{"ocr": false, "objects": false, "scene": false, "person": {"detect": true, "expression": false, "clothing_color": false, "clothing_type": false, "count": true}, "colors": false, "layout": false, "understanding": false}

User: "What color is her dress?"
{"ocr": false, "objects": false, "scene": false, "person": {"detect": true, "expression": false, "clothing_color": true, "clothing_type": true, "count": false}, "colors": false, "layout": false, "understanding": false}

User: "Is this indoors or outdoors?"
{"ocr": false, "objects": false, "scene": true, "person": {"detect": false, "expression": false, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": false, "understanding": false}

User: "What objects are on the table?"
{"ocr": false, "objects": true, "scene": false, "person": {"detect": false, "expression": false, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": false, "understanding": false}

User: "Is he happy or sad?"
{"ocr": false, "objects": false, "scene": false, "person": {"detect": true, "expression": true, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": false, "understanding": false}

User: "What's the main color of the room?"
{"ocr": false, "objects": false, "scene": false, "person": {"detect": false, "expression": false, "clothing_color": false, "clothing_type": false, "count": false}, "colors": true, "layout": false, "understanding": false}

User: "Read the document"
{"ocr": true, "objects": false, "scene": false, "person": {"detect": false, "expression": false, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": true, "understanding": false}

User: "What are they doing?"
{"ocr": false, "objects": false, "scene": false, "person": {"detect": true, "expression": false, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": false, "understanding": true}

User: "Are they fighting or hugging?"
{"ocr": false, "objects": false, "scene": false, "person": {"detect": true, "expression": true, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": false, "understanding": true}

User: "Describe what's happening in the scene"
{"ocr": false, "objects": false, "scene": true, "person": {"detect": true, "expression": false, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": false, "understanding": true}

User: "What is going on here?"
{"ocr": false, "objects": true, "scene": true, "person": {"detect": true, "expression": false, "clothing_color": false, "clothing_type": false, "count": false}, "colors": false, "layout": false, "understanding": true}

Output ONLY the JSON object, no explanation."""


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    content = content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (``` markers)
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        content = content.strip()

    return json.loads(content)


def generate_cvspec_gemini(prompt: str) -> CVSpec:
    """Generate CVSpec using Gemini with shared client and retry logic."""
    try:
        # Use shared client with retry logic
        from utils.gemini_client import generate_with_retry, GEMINI_MODEL

        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: \"{prompt}\""
        content = generate_with_retry(full_prompt)

        data = _parse_json_response(content)
        return CVSpec.from_dict(data)

    except json.JSONDecodeError as e:
        print(f"[CVspec] JSON parsing failed: {e}")
        return CVSpec.all_enabled()
    except Exception as e:
        print(f"[CVspec] Gemini generation failed: {e}")
        return CVSpec.all_enabled()


def generate_cvspec_openai(prompt: str, model: str = "gpt-3.5-turbo") -> CVSpec:
    """Generate CVSpec using OpenAI."""
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        content = response.choices[0].message.content.strip()
        data = _parse_json_response(content)
        return CVSpec.from_dict(data)

    except json.JSONDecodeError as e:
        print(f"[CVspec] JSON parsing failed: {e}")
        return CVSpec.all_enabled()
    except Exception as e:
        print(f"[CVspec] OpenAI generation failed: {e}")
        return CVSpec.all_enabled()


def generate_cvspec(prompt: str, provider: str = "gemini", model: str = None) -> CVSpec:
    """
    Generate CVSpec from user prompt using LLM.

    Args:
        prompt: User's question about the image
        provider: LLM provider ("gemini" or "openai")
        model: Model name (optional, uses defaults)

    Returns:
        CVSpec indicating which vision modules to run
    """
    if provider == "gemini":
        return generate_cvspec_gemini(prompt)
    elif provider == "openai":
        return generate_cvspec_openai(prompt, model or "gpt-3.5-turbo")
    else:
        raise ValueError(f"Unknown provider: {provider}")
