"""Small SFT and GRPO training examples for the project's TRL 0.24 API."""

# Unsloth must be imported before TRL so its patches are applied.
# isort: off
from unsloth import FastLanguageModel, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
# isort: on


def sft_tuning(model, tokenizer, converted_dataset, output_dir="qwen_lora"):
    """Train on conversation examples, save the adapter, and return the trainer."""
    is_vision = hasattr(tokenizer, "tokenizer")
    model_class = FastVisionModel if is_vision else FastLanguageModel
    model_class.for_training(model)
    # Keep image fields intact for the vision collator instead of flattening text.
    vision_options = {}
    if is_vision:
        vision_options = dict(
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer)
        if is_vision
        else None,
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            # Accumulate four microbatches before each optimizer update.
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.0001,
            lr_scheduler_type="linear",
            seed=2026,
            output_dir=output_dir,
            report_to="none",
            max_length=2048,
            **vision_options,
        ),
    )
    # Constructing a trainer does not update weights; train before saving.
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


def grpo_tuning(
    model, tokenizer, converted_dataset, reward_funcs, output_dir="qwen_grpo_lora"
):
    """Train GRPO using explicit rewards and a dataset containing a `prompt` column.

    SFT `messages` examples must be converted to prompts first; exclude the target
    assistant answer. Supply callable rewards or reward model identifiers.
    """
    if not reward_funcs:
        raise ValueError("GRPO requires at least one reward function.")
    model_class = (
        FastVisionModel if hasattr(tokenizer, "tokenizer") else FastLanguageModel
    )
    model_class.for_training(model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=converted_dataset,
        args=GRPOConfig(
            per_device_train_batch_size=2,
            # Accumulate four microbatches before each optimizer update.
            gradient_accumulation_steps=4,
            # Compare rewards for two sampled completions of each prompt.
            num_generations=2,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.0001,
            lr_scheduler_type="linear",
            seed=2026,
            output_dir=output_dir,
            report_to="none",
            remove_unused_columns=False,
            max_completion_length=2048,
        ),
    )
    # Constructing a trainer does not update weights; train before saving.
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer



# format-related rewards, ref to https://github.com/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-GRPO.ipynb
import re

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def strict_format_reward_func(
    completions, **kwargs
) -> list[float]:
    pattern = (
        r"^<reasoning>\n.*?\n</reasoning>\n"
        r"<answer>\n.*?\n</answer>\n$"
    )
    responses = [
        completion[0]["content"]
        for completion in completions
    ]
    return [
        0.5 if re.match(pattern, response) else 0.0
        for response in responses
    ]


def soft_format_reward_func(
    completions, **kwargs
) -> list[float]:
    pattern = (
        r"<reasoning>.*?</reasoning>\s*"
        r"<answer>.*?</answer>"
    )
    responses = [
        completion[0]["content"]
        for completion in completions
    ]
    return [
        0.5 if re.match(pattern, response) else 0.0
        for response in responses
    ]


def count_xml(text: str) -> float:
    score = 0.0

    if text.count("<reasoning>\n") == 1:
        score += 0.125

    if text.count("\n</reasoning>\n") == 1:
        score += 0.125

    if text.count("\n<answer>\n") == 1:
        score += 0.125
        score -= len(
            text.split("\n</answer>\n")[-1]
        ) * 0.001

    if text.count("\n</answer>") == 1:
        score += 0.125
        score -= (
            len(text.split("\n</answer>")[-1]) - 1
        ) * 0.001

    return score


def xmlcount_reward_func(
    completions, **kwargs
) -> list[float]:
    return [
        count_xml(completion[0]["content"])
        for completion in completions
    ]


# correctness rewards
def correctness_reward_func(
    prompts, completions, answer, **kwargs
) -> list[float]:
    responses = [
        completion[0]["content"]
        for completion in completions
    ]
    predictions = [
        extract_xml_answer(response)
        for response in responses
    ]
    return [
        1.0 if prediction == target else 0.0
        for prediction, target in zip(predictions, answer)
    ]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [
        completion[0]["content"]
        for completion in completions
    ]
    predictions = [
        extract_xml_answer(response)
        for response in responses
    ]
    return [
        0.5 if prediction.isdigit() else 0.0
        for prediction in predictions
    ]