# Web Interface Guide

## Quick Start

### 1. Install Flask
```bash
pip install flask
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Set up API keys
Create a `.env` file with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

### 3. Run the web server
```bash
python app.py
```

The server will start at `http://localhost:5000`

### 4. Use the web interface
1. Open your browser to `http://localhost:5000`
2. Upload an image (png, jpg, jpeg, gif, bmp, webp)
3. Type your question about the image
4. Select your LLM provider (Gemini or OpenAI)
5. Click "Analyze Image"

## Features

The web interface shows:
- **Final Answer**: The LLM's response to your question
- **Token Savings**: Comparison showing how many tokens you saved vs direct Gemini image input
- **Pipeline Details**: Full breakdown of all 6 pipeline steps (collapsible)

## How It Works

1. **Upload Image**: Your image is temporarily uploaded to the server
2. **Process**: The full 6-step pipeline runs:
   - Step 1: Load inputs
   - Step 2: Compress prompt (bear-1)
   - Step 3: Generate CVspec (Gemini determines which vision modules to run)
   - Step 4: Run vision pipeline (OCR, objects, scene, person, colors, layout)
   - Step 5: Compress outputs (bear-1)
   - Step 6: Get final answer from LLM (Gemini/OpenAI)
3. **Display Results**: All results and intermediate steps are shown
4. **Cleanup**: Uploaded image is automatically deleted after processing

## API Endpoint

### POST /analyze

**Request:**
- `image`: Image file (multipart/form-data)
- `prompt`: Question about the image (form field)
- `provider`: LLM provider - "gemini" or "openai" (form field, optional, default: "gemini")

**Response:**
```json
{
  "success": true,
  "step2": { "original_prompt": "...", "compressed_prompt": "...", ... },
  "step3": { "cvspec": {...}, "active_modules": [...] },
  "step4": { "raw_output": "...", "output_tokens": 100 },
  "step5": { "compressed_image_desc": "...", "total_compressed": 50, ... },
  "step6": { "final_prompt": "...", "answer": "..." },
  "comparison": { "baseline_tokens": 300, "our_tokens": 50, "savings_percent": 83.3 }
}
```

## CLI Still Available

The CLI remains fully functional:
```bash
python main.py analyze -i person.jpg -p "What is this person feeling?" -v
```

## Troubleshooting

- **Port already in use**: Change the port in app.py: `app.run(debug=True, host='0.0.0.0', port=5001)`
- **Missing API keys**: Make sure `.env` file exists with valid API keys
- **Vision modules failing**: Some modules require model downloads on first run (YOLOv8, EasyOCR, SmolVLM)
