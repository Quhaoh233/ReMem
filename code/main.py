# public
import time

from unsloth import FastLanguageModel, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
import torch
from datasets import load_dataset
from transformers import TextStreamer
import sys
import json
# private
import reasoning_utils
import dataset_utils
import prompt_utils
import model_utils
import evaluation_utils

# TODO: sequence length and token nums


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-VL-8B-Instruct", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--modal", type=str, default="vision", help="[text, vision] whether to run text-only or vision tasks")
    parser.add_argument("--save_name", type=str, default="local/YOUR_MODEL_NAME", help="model name saved in your deployment/model service endpoint")
    parser.add_argument("--hf_token", type=str, default="YOUR_HF_TOKEN", help="Your Hugging Face token here if loading from private repositories")
    parser.add_argument("--batch", type=int, default=2, help="training batch size")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for reproducibility")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="checkpoint path for resuming training")
    parser.add_argument("--chunk_size", type=int, default=3, help="chunk size for memory reasoning")    
    parser.add_argument("--reasoning", type=str, default="memory", help="[vanilla, memory, recurrent] reasoning method")
    parser.add_argument("--task", type=str, default="judging", help="task type for evaluation, [searching, ranking, judging]")
    parser.add_argument("--dataset", type=str, default="games", help="dataset name")  # webwalkerqa_main, instructrec_movietv, instructrec_books, instructrec_reads, hotpotqa
    args = parser.parse_args()

    # ------------------- load model and dataset --------------------
    dataset = dataset_utils.load_dataset(args) # Custom dataset example
    model, tokenizer = model_utils.load_model(args)
    if args.dataset == "hotpotqa":
        args.task = "long-context reasoning"
    elif args.dataset == "webwalkerqa_main":
        args.task = "web search reasoning"
    else:
        pass
    
    # ----------------- Evaluation -----------------
    accuracy = 0
    available = 0
    input_length = 0
    sample_n = 0
    seq_l = 0
    long_n = 0
    max_length = 0
    total_time = 0
    FastVisionModel.for_inference(model) # Enable for inference
    with open(f"../result/{args.dataset}_{args.model.split('/')[-1]}_reasoning_{args.reasoning}_{args.task}_logs.jsonl", "w") as f:
        for sample in dataset:
            length = len(sample['content'])+len(sample['question'])+len(sample['candidates'])

            input_length += length
            sample_n += 1
            seq_l += sample['seq_l']
            if length > max_length:
                max_length = length
            # print(f"current sample length: {length}, current average length: {input_length/sample_n:.4f}, current average seq_l: {seq_l/sample_n:.4f}")

            available += 1
            start = time.time()
            # reasoning
            if args.reasoning == "memory":
                question, golden_answer, response_text, token_num, history, candidates, memory_record = reasoning_utils.mem_reasoning(sample, model, tokenizer, chunk_size=args.chunk_size)
            elif args.reasoning == "recurrent":
                question, golden_answer, response_text, token_num = reasoning_utils.recurrent_reasoning(sample, model, tokenizer, args)
            elif args.reasoning == "vanilla":
                question, golden_answer, response_text, token_num = reasoning_utils.vanilla_reasoning(sample, model, tokenizer, args)
            else:
                raise ValueError(f"Unsupported reasoning method: {args.reasoning}")

            end = time.time()
            total_time += end - start

            # evaluate
            result = evaluation_utils.llm_as_judge(question, golden_answer, response_text)
            judgement_str = result['judgement']
            if judgement_str == "correct":
                accuracy += 1

            # save log to file
            log = {
                "idx": sample["idx"],
                "question": question,
                "golden_answer": golden_answer,
                "pred_answer": response_text,
                "judgement": judgement_str,
                "current_accuracy": accuracy / available,
                "current_average_input_length": input_length / available,
                # "memory_record": memory_record,
                # "history": history,
                # "candidates": candidates,
                "inference_time": end - start,
            }

            
            print(f"[Dataset: {args.dataset} | Task: {args.task} | Model: {args.model}] Sample {str(sample['idx'])}: {judgement_str}, current accuracy: {accuracy / (available):.4f}, current average input length: {input_length / (available):.4f}, average inference time: {(total_time)/available:.4f}s")
            f.write(json.dumps(log) + "\n")
            f.flush()
        
        print(f"final average length: {input_length/sample_n:.4f}, final average seq_l: {seq_l/sample_n:.4f}")
        print(f"max length: {max_length}")
        print(f"final average inference time: {total_time/sample_n:.4f}s")
    
    f.close()
    print(available)
    print(f"Final Accuracy: {accuracy / available:.4f}")    
    print(f"FinalAverage Input Length: {input_length / available:.4f}")
