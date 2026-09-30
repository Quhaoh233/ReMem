# ReMem

**ReMem** is a novel recommender AI agent framework that combines OCR-based multimodal perception with time-evolving dynamic memory.





## Environment

We recommend using **Python 3.12** with the **UnSloth** environment.

Alternatively, you can install all required dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration

- Enter your **Hugging Face (HF) token** in `main.py` to access open-source models (e.g., Qwen3.5-9B and Qwen3-VL-8B).
- Enter your **OpenAI API key** in `code/.env` for the LLM-as-Judge evaluation.

## Quick Start

```bash
cd code
python main.py
```

Additional configuration options can be found in the argument parser defined in `main.py`.