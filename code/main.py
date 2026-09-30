"""Command-line evaluation for recommendation and question-answering datasets."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path


def positive_int(value):
    """Reject invalid sizes before loading expensive models or datasets."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    """Parse evaluation settings; paths are independent of the working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct")  # select your model here
    parser.add_argument("--modal", choices=("text", "vision"), default="vision")
    parser.add_argument(
        "--save_name",
        default="local/YOUR_MODEL_NAME",
        help="Reserved training output name",
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN"),  # important for private models; set HF_TOKEN in your environment
        help="Optional Hugging Face token; defaults to HF_TOKEN",
    )
    parser.add_argument(
        "--batch", type=positive_int, default=2, help="Reserved training batch size"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--load_checkpoint", help="Model or adapter checkpoint to load")
    parser.add_argument(
        "--chunk_size",
        type=positive_int,
        default=3,
        help="History items per memory update",
    )
    parser.add_argument(
        "--reasoning", choices=("vanilla", "memory", "recurrent"), default="memory"
    )
    parser.add_argument("--task", choices=("searching", "judging"), default="judging")
    parser.add_argument(
        "--dataset",
        choices=(
            "games",
            "movietv",
            "books",
            "hotpotqa",
            "webwalkerqa_main",
        ),
        default="games",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run evaluation and stream one JSON record per sample to the result folder."""
    args = parse_args(argv)
    # Import Unsloth first for patching; defer all GPU imports until after --help.
    # isort: off
    from unsloth import FastLanguageModel, FastVisionModel
    import torch
    import dataset_utils
    import evaluation_utils
    import model_utils
    import reasoning_utils
    # isort: on

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    # Candidate selection uses Python randomness; generation uses PyTorch.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Each dataset adapter returns the same fields, so the loop below is shared.
    dataset = dataset_utils.load_dataset(args)
    if len(dataset) == 0:
        print("The dataset is empty; no samples to evaluate.")
        return
    model, tokenizer = model_utils.load_model(args)
    model_class = FastLanguageModel if args.modal == "text" else FastVisionModel
    model_class.for_inference(model)
    args.task = {
        "hotpotqa": "long-context reasoning",
        "webwalkerqa_main": "web search reasoning",
    }.get(args.dataset, args.task)

    result_dir = Path(__file__).resolve().parent.parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model.rstrip("/").split("/")[-1]
    log_path = (
        result_dir
        / f"{args.dataset}_{model_name}_reasoning_{args.reasoning}_{args.task}_logs.jsonl"
    )
    correct = sample_count = judge_errors = total_length = total_items = max_length = 0
    total_time = 0.0
    with log_path.open("w", encoding="utf-8") as log_file:
        for sample in dataset:
            # Candidates are already embedded in the question; do not count twice.
            length = len(sample["content"]) + len(sample["question"])
            total_length += length
            total_items += sample.get("seq_l", 0)
            max_length = max(max_length, length)
            sample_count += 1
            # memory groups history items; recurrent groups tokens; vanilla reads once.
            start = time.perf_counter()
            if args.reasoning == "memory":
                output = reasoning_utils.mem_reasoning(
                    sample, model, tokenizer, args.chunk_size
                )
            elif args.reasoning == "recurrent":
                output = reasoning_utils.recurrent_reasoning(
                    sample, model, tokenizer, args
                )
            else:
                output = reasoning_utils.vanilla_reasoning(
                    sample, model, tokenizer, args
                )
            # All strategies share these four results; memory also returns a trace.
            question, golden_answer, response_text, token_count = output[:4]
            elapsed = time.perf_counter() - start
            total_time += elapsed
            # Judge latency is excluded from the model inference time above.
            judgement = evaluation_utils.llm_as_judge(
                question, golden_answer, response_text
            )["judgement"]
            correct += judgement == "correct"
            judge_errors += judgement == "error"
            # An unavailable judge is not evidence that the model answered wrongly.
            evaluated = sample_count - judge_errors
            accuracy = correct / evaluated if evaluated else None
            log = {
                "idx": sample["idx"],
                "question": question,
                "golden_answer": golden_answer,
                "pred_answer": response_text,
                "judgement": judgement,
                "current_accuracy": accuracy,
                "judge_errors": judge_errors,
                "current_average_input_length": total_length / sample_count,
                "context_tokens": token_count,
                "inference_time": elapsed,
            }
            log_file.write(json.dumps(log, ensure_ascii=False) + "\n")
            # Preserve completed samples if a later generation fails.
            log_file.flush()
            accuracy_text = f"{accuracy:.4f}" if accuracy is not None else "N/A"
            print(
                f"Sample {sample['idx']}: {judgement}; accuracy: {accuracy_text}; inference: {elapsed:.2f}s"
            )

    print(f"Samples: {sample_count}; judge errors: {judge_errors}")
    print(f"Final accuracy (successful judgements): {accuracy_text}")
    print(
        f"Average input length (characters): {total_length / sample_count:.4f}; maximum: {max_length}"
    )
    print(f"Average sequence length: {total_items / sample_count:.4f}")
    print(f"Average inference time: {total_time / sample_count:.4f}s")
    print(f"Logs saved to {log_path}")


if __name__ == "__main__":
    main()
