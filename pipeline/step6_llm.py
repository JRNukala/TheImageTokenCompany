"""
Step 6: Final LLM Call.

Sends minimal prompt to LLM for final answer.
Supports both Gemini (default) and OpenAI.
"""
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Lazy-loaded clients
_gemini_model = None
_openai_client = None


def _get_gemini_model():
    """Lazy load Gemini model."""
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    return _gemini_model


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


PROMPT_TEMPLATE = """Image: {img_descr}
Question: {question}
Answer:"""

SYSTEM_INSTRUCTION = "Answer the question about the image based on the description provided. Be concise."


def get_answer_gemini(compressed_img_descr: str, compressed_txt: str) -> str:
    """Get answer using Gemini."""
    prompt = PROMPT_TEMPLATE.format(
        img_descr=compressed_img_descr,
        question=compressed_txt
    )

    try:
        model = _get_gemini_model()
        full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{prompt}"
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 150
            }
        )
        return response.text.strip()

    except Exception as e:
        return f"Error getting answer: {e}"


def get_answer_openai(compressed_img_descr: str, compressed_txt: str, model: str = "gpt-4o-mini") -> str:
    """Get answer using OpenAI."""
    prompt = PROMPT_TEMPLATE.format(
        img_descr=compressed_img_descr,
        question=compressed_txt
    )

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error getting answer: {e}"


def get_answer(
    compressed_img_descr: str,
    compressed_txt: str,
    provider: str = "gemini",
    model: str = None
) -> str:
    """
    Get final answer from LLM.

    Args:
        compressed_img_descr: Compressed vision output
        compressed_txt: Compressed user question
        provider: LLM provider ("gemini" or "openai")
        model: Model name (optional, uses defaults)

    Returns:
        LLM's answer string
    """
    if provider == "gemini":
        return get_answer_gemini(compressed_img_descr, compressed_txt)
    elif provider == "openai":
        return get_answer_openai(compressed_img_descr, compressed_txt, model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def build_final_prompt(compressed_img_descr: str, compressed_txt: str) -> str:
    """
    Build the final prompt for display/debugging.

    Args:
        compressed_img_descr: Compressed vision output
        compressed_txt: Compressed user question

    Returns:
        Final prompt string
    """
    return PROMPT_TEMPLATE.format(
        img_descr=compressed_img_descr,
        question=compressed_txt
    )
