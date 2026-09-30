"""Run DeepSeek OCR on a bounded number of PNG images."""

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main(argv=None):
    """Load OCR weights only when explicitly running this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", type=Path, default=DATA_DIR / "amazon_video_games_item_images_v3"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DATA_DIR / "games" / "ocr_results"
    )
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum PNG images to process"
    )
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    # Filter before limiting so non-images do not consume the processing budget.
    images = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )[: args.limit]
    if not images:
        print("No PNG images found.")
        return

    # Delay GPU setup until input validation has found actual work.
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        .eval()
        .cuda()
    )
    model.generation_config.repetition_penalty = 1.1
    for image_path in images:
        # Each image gets its own folder for Markdown and OCR visualizations.
        output_path = args.output_dir / image_path.stem
        output_path.mkdir(parents=True, exist_ok=True)
        model.infer(
            tokenizer,
            prompt="<image>\n<|grounding|>Convert the document to markdown. ",
            image_file=str(image_path),
            output_path=str(output_path),
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=True,
        )


if __name__ == "__main__":
    main()
