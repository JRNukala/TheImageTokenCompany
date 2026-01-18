#!/usr/bin/env python3
"""
Web interface for Multimodal Semantic Compression Pipeline
"""
import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from utils.tokens import count_tokens
from pipeline.step1_input import load_inputs
from pipeline.step2_compress_prompt import compress_prompt
from pipeline.step3_cvspec import generate_cvspec
from pipeline.step4_vision import run_vision_pipeline, get_active_modules
from pipeline.step5_compress import compress_outputs
from pipeline.step6_llm import get_answer, build_final_prompt

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Process image and prompt, return results."""
    try:
        # Validate inputs
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        if 'prompt' not in request.form:
            return jsonify({'error': 'No prompt provided'}), 400

        file = request.files['image']
        prompt = request.form['prompt']
        provider = request.form.get('provider', 'gemini')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, webp'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run pipeline
        result = run_pipeline(filepath, prompt, provider)

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /analyze: {error_trace}")
        return jsonify({'error': str(e), 'trace': error_trace}), 500


def run_pipeline(image_path: str, prompt: str, provider: str = "gemini"):
    """
    Run the full compression pipeline and return detailed results.

    Returns:
        Dictionary with all pipeline steps and results
    """
    # STEP 1: Load inputs
    inputs = load_inputs(image_path, prompt)

    # STEP 2: Compress prompt
    compressed_prompt, was_compressed, prompt_result = compress_prompt(inputs.prompt, provider=provider)

    step2_result = {
        'original_prompt': inputs.prompt,
        'compressed_prompt': compressed_prompt,
        'was_compressed': was_compressed,
        'original_tokens': prompt_result.original_tokens if was_compressed else count_tokens(inputs.prompt, provider),
        'compressed_tokens': prompt_result.compressed_tokens if was_compressed else count_tokens(inputs.prompt, provider),
        'savings_percent': prompt_result.savings_percent if was_compressed else 0
    }

    # STEP 3: Generate CVspec
    cvspec = generate_cvspec(inputs.prompt, provider="gemini")

    # Fallback: If no modules selected, use understanding (SmolVLM) as default
    modules = get_active_modules(cvspec)
    if not modules:
        cvspec.understanding = True

    step3_result = {
        'cvspec': cvspec.to_dict(),
        'active_modules': get_active_modules(cvspec)
    }

    # STEP 4: Run vision pipeline
    img_descr = run_vision_pipeline(inputs.image, cvspec, user_prompt=inputs.prompt)

    step4_result = {
        'raw_output': img_descr,
        'output_tokens': count_tokens(img_descr, provider)
    }

    # STEP 5: Compress outputs
    compressed_img, compressed_txt, stats = compress_outputs(
        img_descr, compressed_prompt, txt_already_compressed=was_compressed, provider=provider
    )

    step5_result = {
        'compressed_image_desc': compressed_img,
        'compressed_question': compressed_txt,
        'compressed_img_tokens': stats.compressed_img_tokens,
        'compressed_txt_tokens': stats.compressed_txt_tokens,
        'total_compressed': stats.total_compressed,
        'savings_percent': stats.savings_percent
    }

    # STEP 6: Get final answer
    final_prompt = build_final_prompt(compressed_img, compressed_txt)
    answer = get_answer(compressed_img, compressed_txt, provider=provider, original_prompt=inputs.prompt)

    step6_result = {
        'final_prompt': final_prompt,
        'final_prompt_tokens': count_tokens(final_prompt, provider),
        'answer': answer
    }

    # Calculate token comparison
    gemini_image_tokens = 258
    prompt_tokens = count_tokens(inputs.prompt, provider)
    baseline_tokens = gemini_image_tokens + prompt_tokens
    our_tokens = stats.total_compressed

    comparison = {
        'baseline_tokens': baseline_tokens,
        'our_tokens': our_tokens,
        'savings_percent': ((baseline_tokens - our_tokens) / baseline_tokens * 100)
    }

    return {
        'success': True,
        'step2': step2_result,
        'step3': step3_result,
        'step4': step4_result,
        'step5': step5_result,
        'step6': step6_result,
        'comparison': comparison
    }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
