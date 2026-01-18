"""
Step 6: Final LLM Call.

Sends minimal prompt to LLM for final answer.
Supports both Gemini (default) and OpenAI.
"""
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Lazy-loaded OpenAI client (Gemini uses shared client)
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


PROMPT_TEMPLATE = """Image: {img_descr}
Question: {question}
Answer:"""

SYSTEM_INSTRUCTION = "Answer the question about the image based on the description provided. Be concise."


def get_answer_gemini(compressed_img_descr: str, compressed_txt: str, original_prompt: str = None) -> str:
    """Get answer using Gemini with shared client and retry logic."""
    # If image description is empty or vision failed, use only the original prompt
    if not compressed_img_descr or compressed_img_descr.strip() == "":
        if original_prompt:
            print("[LLM] No image info available, using original prompt only")
            prompt = f"Question: {original_prompt}\nAnswer:"
        else:
            prompt = PROMPT_TEMPLATE.format(
                img_descr="[No image information available]",
                question=compressed_txt
            )
    else:
        prompt = PROMPT_TEMPLATE.format(
            img_descr=compressed_img_descr,
            question=compressed_txt
        )

    try:
        # Use shared client with retry logic
        from utils.gemini_client import generate_with_retry

        full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{prompt}"
        return generate_with_retry(full_prompt)

    except Exception as e:
        print(f"[LLM] Gemini failed: {e}")
        # Fallback: if we have original prompt, return a message
        if original_prompt:
            return f"Unable to process image. Original question was: {original_prompt}"
        return f"Error getting answer: {e}"


def get_answer_openai(compressed_img_descr: str, compressed_txt: str, original_prompt: str = None, model: str = "gpt-4o-mini") -> str:
    """Get answer using OpenAI."""
    # If image description is empty or vision failed, use only the original prompt
    if not compressed_img_descr or compressed_img_descr.strip() == "":
        if original_prompt:
            print("[LLM] No image info available, using original prompt only")
            prompt = f"Question: {original_prompt}\nAnswer:"
        else:
            prompt = PROMPT_TEMPLATE.format(
                img_descr="[No image information available]",
                question=compressed_txt
            )
    else:
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
        print(f"[LLM] OpenAI failed: {e}")
        # Fallback: if we have original prompt, return a message
        if original_prompt:
            return f"Unable to process image. Original question was: {original_prompt}"
        return f"Error getting answer: {e}"


def get_answer(
    compressed_img_descr: str,
    compressed_txt: str,
    provider: str = "gemini",
    model: str = None,
    original_prompt: str = None
) -> str:
    """
    Get final answer from LLM.

    Args:
        compressed_img_descr: Compressed vision output
        compressed_txt: Compressed user question
        provider: LLM provider ("gemini" or "openai")
        model: Model name (optional, uses defaults)
        original_prompt: Original user prompt (fallback if vision fails)

    Returns:
        LLM's answer string
    """
    if provider == "gemini":
        return get_answer_gemini(compressed_img_descr, compressed_txt, original_prompt)
    elif provider == "openai":
        return get_answer_openai(compressed_img_descr, compressed_txt, original_prompt, model or "gpt-4o-mini")
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
