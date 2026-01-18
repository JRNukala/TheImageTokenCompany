#!/usr/bin/env python3
"""
Multimodal Semantic Compression Pipeline

Query-aware visual extraction using bear-1 compression.
"""
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

app = typer.Typer(
    name="mscp",
    help="Multimodal Semantic Compression Pipeline - Query-aware visual extraction"
)
console = Console()


def run_pipeline(
    image_path: str,
    prompt: str,
    verbose: bool = False,
    provider: str = "gemini",
    model: str = None
) -> str:
    """
    Run the full compression pipeline.

    Args:
        image_path: Path to image file
        prompt: User's question about the image
        verbose: Whether to show detailed output
        provider: LLM provider ("gemini" or "openai")
        model: Model name (optional)

    Returns:
        Final answer string
    """
    from utils.tokens import count_tokens
    from pipeline.step1_input import load_inputs, get_image_info
    from pipeline.step2_compress_prompt import compress_prompt
    from pipeline.step3_cvspec import generate_cvspec
    from pipeline.step4_vision import run_vision_pipeline, get_active_modules
    from pipeline.step5_compress import compress_outputs
    from pipeline.step6_llm import get_answer, build_final_prompt

    # STEP 1: Load inputs
    if verbose:
        console.print("\n[bold blue]STEP 1:[/] Loading inputs...")

    inputs = load_inputs(image_path, prompt)

    if verbose:
        info = get_image_info(inputs.image)
        console.print(f"  Image: {inputs.image_path} ({info['size']})")
        console.print(f"  Prompt: \"{inputs.prompt}\"")
        console.print(f"  Prompt tokens: {count_tokens(inputs.prompt, provider)}")

    # STEP 2: Compress prompt (optional)
    if verbose:
        console.print("\n[bold blue]STEP 2:[/] Compressing prompt...")

    compressed_prompt, was_compressed, prompt_result = compress_prompt(inputs.prompt, provider=provider)

    if verbose:
        if was_compressed:
            console.print(f"  Original: \"{inputs.prompt}\" ({prompt_result.original_tokens} tokens)")
            console.print(f"  Compressed: \"{compressed_prompt}\" ({prompt_result.compressed_tokens} tokens)")
            console.print(f"  Savings: {prompt_result.savings_percent:.1f}%")
        else:
            console.print(f"  Skipped (prompt already short)")

    # STEP 3: Generate CVspec (use ORIGINAL prompt for better routing accuracy)
    if verbose:
        console.print(f"\n[bold blue]STEP 3:[/] Generating CVspec (using Gemini)...")

    cvspec = generate_cvspec(inputs.prompt, provider="gemini")

    # Fallback: If no modules selected, use understanding (SmolVLM) as default
    modules = get_active_modules(cvspec)
    if not modules:
        cvspec.understanding = True
        if verbose:
            console.print(f"  [yellow]No modules selected, falling back to understanding (SmolVLM)[/]")

    if verbose:
        modules = get_active_modules(cvspec)
        console.print(f"  Active modules: {', '.join(modules) if modules else 'none'}")
        # Debug: Show full CVSpec
        console.print(f"  [dim]CVSpec: {cvspec.to_dict()}[/]")

    # STEP 4: Run vision pipeline
    if verbose:
        console.print("\n[bold blue]STEP 4:[/] Running vision pipeline...")

    img_descr = run_vision_pipeline(inputs.image, cvspec, user_prompt=inputs.prompt)

    if verbose:
        console.print(f"  Raw output: \"{img_descr}\"")
        console.print(f"  Output tokens: {count_tokens(img_descr, provider)}")

    # STEP 5: Compress outputs
    if verbose:
        console.print("\n[bold blue]STEP 5:[/] Compressing with bear-1...")

    compressed_img, compressed_txt, stats = compress_outputs(
        img_descr, compressed_prompt, txt_already_compressed=was_compressed, provider=provider
    )

    if verbose:
        console.print(f"  Image desc: \"{compressed_img}\" ({stats.compressed_img_tokens} tokens)")
        console.print(f"  Question: \"{compressed_txt}\" ({stats.compressed_txt_tokens} tokens)")
        console.print(f"  Total tokens: {stats.total_compressed}")
        console.print(f"  Savings: {stats.savings_percent:.1f}%")

    # STEP 6: Get final answer
    if verbose:
        console.print(f"\n[bold blue]STEP 6:[/] Getting final answer from LLM...")
        final_prompt = build_final_prompt(compressed_img, compressed_txt)
        console.print(f"  Final prompt:\n    {final_prompt.replace(chr(10), chr(10) + '    ')}")
        console.print(f"  Final prompt tokens: {count_tokens(final_prompt, provider)}")

    answer = get_answer(compressed_img, compressed_txt, provider=provider, model=model, original_prompt=inputs.prompt)

    if verbose:
        console.print(f"\n[bold yellow]Debug Summary:[/]")
        console.print(f"  Vision pipeline raw output: \"{img_descr}\"")
        console.print(f"  Compressed image desc: \"{compressed_img}\"")
        console.print(f"  Compressed question: \"{compressed_txt}\"")

        # Show summary comparing to direct Gemini image call
        # Gemini charges ~258 tokens per image input (standard) or up to 560 tokens
        # We use 258 as the baseline for standard image processing
        gemini_image_tokens = 258
        prompt_tokens = count_tokens(inputs.prompt, provider)
        baseline_tokens = gemini_image_tokens + prompt_tokens  # Image + question sent to Gemini
        our_tokens = stats.total_compressed

        console.print("\n")
        table = Table(title="Token Comparison vs Direct Gemini")
        table.add_column("Approach", style="cyan", width=35)
        table.add_column("Input Tokens", justify="right")
        table.add_column("Savings", justify="right")
        table.add_row(
            "Direct Gemini (image + question)",
            f"{baseline_tokens} ({gemini_image_tokens} img + {prompt_tokens} txt)",
            "-"
        )
        table.add_row(
            "Our pipeline (text only)",
            str(our_tokens),
            f"[bold green]{((baseline_tokens - our_tokens) / baseline_tokens * 100):.0f}%[/]"
        )
        console.print(table)
        console.print(f"\n[dim]Note: Gemini charges ~258 tokens per image input[/]")

    return answer


@app.command()
def analyze(
    image: str = typer.Option(..., "--image", "-i", help="Path to image file"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Question about the image"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed pipeline output"),
    provider: str = typer.Option("gemini", "--provider", help="LLM provider: gemini or openai"),
    model: str = typer.Option(None, "--model", "-m", help="Model name (optional, uses defaults)")
):
    """
    Analyze an image with query-aware semantic compression.

    Example:
        python main.py analyze -i photo.jpg -p "What color is his shirt?" -v
        python main.py analyze -i photo.jpg -p "What color is his shirt?" --provider openai
    """
    try:
        answer = run_pipeline(image, prompt, verbose=verbose, provider=provider, model=model)

        if not verbose:
            console.print(f"\n[bold]Answer:[/] {answer}\n")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def demo():
    """
    Run a demo comparison showing token savings.
    """
    console.print(Panel.fit(
        "[bold]Multimodal Semantic Compression Pipeline Demo[/]\n\n"
        "This demo shows the token savings from query-aware visual extraction.\n\n"
        "[yellow]To run:[/] python main.py analyze -i <image> -p <prompt> -v",
        title="MSCP Demo"
    ))

    # Show example comparison
    console.print("\n[bold]Example Comparison:[/]\n")

    table = Table(title="Traditional vs Our Approach")
    table.add_column("Approach", style="cyan", width=20)
    table.add_column("Description", width=50)
    table.add_column("Tokens", justify="right")

    table.add_row(
        "Traditional",
        "\"This image shows a man standing in a park on a sunny day. "
        "He appears to be in his 30s with brown hair. He is wearing a "
        "red t-shirt and blue jeans...\"",
        "~100"
    )
    table.add_row(
        "Our Approach",
        "Image: person red t-shirt\nQuestion: color shirt",
        "~8"
    )
    console.print(table)

    console.print("\n[bold green]Token Savings: 92%[/]\n")


if __name__ == "__main__":
    app()
