# TODO.md — Multimodal Semantic Compression Pipeline

## 🔥 Current Tasks

* [ ] Install Flask: `pip install flask`
* [ ] Test web interface: `python app.py`
* [ ] Test CLI: `python main.py analyze -i person.jpg -p "What is this person feeling?" -v`

## ✅ Completed

* [x] Create CLAUDE.md (hackathon execution rules)
* [x] Create TODO.md (task tracking)
* [x] Create project structure (pipeline/, vision_modules/, utils/)
* [x] Create requirements.txt
* [x] Create .env.example
* [x] Implement utils/bear1.py - bear-1 API wrapper
* [x] Implement utils/tokens.py - tiktoken counter
* [x] Implement pipeline/step3_cvspec.py - CVspec generation (core innovation)
* [x] Implement vision_modules/ocr.py - EasyOCR wrapper
* [x] Implement vision_modules/objects.py - YOLOv8 object detection
* [x] Implement vision_modules/scene.py - Scene classification
* [x] Implement vision_modules/colors.py - Dominant color extraction
* [x] Implement vision_modules/person.py - Person detection + attributes
* [x] Implement vision_modules/layout.py - Document layout analysis
* [x] Implement pipeline/step4_vision.py - Vision dispatcher
* [x] Implement pipeline/step5_compress.py - Final compression
* [x] Implement pipeline/step6_llm.py - Final LLM call
* [x] Implement pipeline/step1_input.py - Input validation
* [x] Implement pipeline/step2_compress_prompt.py - Prompt compression
* [x] Implement main.py CLI with typer/rich
* [x] Enhanced SmolVLM scene understanding with more expressive descriptions (30-40 words vs 15)
* [x] Set CVspec routing to use Gemini (most reliable for structured JSON output)
* [x] Added understanding_focus field to CVSpec for Gemini to guide SmolVLM
* [x] Re-enabled final LLM call with fallback to original prompt if vision fails
* [x] SmolVLM uses user prompt as focus hint when Gemini doesn't provide one
* [x] Created web interface with Flask (app.py)
* [x] Built frontend UI with image upload, prompt input, and results display
* [x] Added visual token comparison and pipeline details viewer

## 🧠 Notes / Decisions

* **Tech Stack**: Python + OpenCV + YOLOv8 + EasyOCR + MediaPipe + bear-1 + OpenAI + SmolVLM
* **LLM Provider**: Gemini for CVspec routing, OpenAI for final answer
* **Core Innovation**: CVspec - using cheap LLM (Gemini Flash) to route expensive vision operations
* **SmolVLM Integration**: Used for scene understanding module (local, 100% offline)
* **Target**: 85-95% token reduction vs verbose captions
* **Demo Strategy**: Side-by-side comparison showing token savings

## 📁 Project Structure

```
The Token Company Project/
├── main.py                      # CLI entry point
├── app.py                       # Web interface (Flask)
├── requirements.txt             # Dependencies
├── .env.example                 # API key template
├── CLAUDE.md                    # Hackathon rules
├── TODO.md                      # This file
├── templates/
│   └── index.html               # Web UI template
├── static/
│   ├── css/style.css            # Web UI styling
│   └── js/app.js                # Web UI JavaScript
├── pipeline/
│   ├── step1_input.py           # Input handling
│   ├── step2_compress_prompt.py # Optional prompt compression
│   ├── step3_cvspec.py          # CVspec generation (core!)
│   ├── step4_vision.py          # Vision dispatcher
│   ├── step5_compress.py        # Final compression
│   └── step6_llm.py             # Final LLM call
├── vision_modules/
│   ├── ocr.py                   # EasyOCR
│   ├── objects.py               # YOLOv8
│   ├── scene.py                 # Scene classification
│   ├── person.py                # Person detection
│   ├── colors.py                # Color extraction
│   └── layout.py                # Document layout
└── utils/
    ├── bear1.py                 # bear-1 wrapper
    └── tokens.py                # Token counting
```
