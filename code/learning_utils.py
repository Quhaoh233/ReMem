"""SFT and GRPO helpers for the project's Unsloth/TRL 0.24 stack.

GRPO datasets are prompt-only datasets: each row needs a ``prompt`` column and
may keep side columns such as ``answer`` for custom reward functions.  Use
``convert_to_grpo_example`` to adapt the common samples from ``dataset_utils``.
"""

import re
from collections.abc import Mapping

import prompt_utils

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
    model,
    tokenizer,
    converted_dataset,
    reward_funcs,
    output_dir="qwen_grpo_lora",
    config_overrides=None,
):
    """Train GRPO using rewards and a dataset containing a ``prompt`` column.

    SFT `messages` examples must be converted to prompts first; exclude the target
    assistant answer. Supply callable rewards or reward model identifiers. Values
    in ``config_overrides`` replace the small single-GPU example defaults below.
    """
    if not reward_funcs:
        raise ValueError("GRPO requires at least one reward function.")
    _validate_grpo_dataset(converted_dataset, reward_funcs)
    is_vision = hasattr(tokenizer, "tokenizer")
    model_class = FastVisionModel if is_vision else FastLanguageModel
    model_class.for_training(model)
    config_options = dict(
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
        # Keep columns such as `answer` available to custom rewards.
        remove_unused_columns=False,
        max_completion_length=2048,
        # Truncation can remove image tokens from VLM prompts.
        max_prompt_length=None if is_vision else prompt_utils.MAX_CONTEXT_LEN,
    )
    if config_overrides:
        config_options.update(config_overrides)
    save_dir = config_options["output_dir"]
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=converted_dataset,
        args=GRPOConfig(**config_options),
    )
    # Constructing a trainer does not update weights; train before saving.
    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return trainer


def multi_conversion_grpo_tuning(
    model,
    tokenizer,
    converted_datasets,
    reward_funcs,
    output_dir="qwen_multi_conversion_grpo_lora",
    config_overrides=None,
):
    """Train one GRPO policy over multiple prompt-conversion datasets.

    ``converted_datasets`` should map a short conversion name to a prompt-only
    dataset, for example ``{"vanilla": vanilla_rows, "recurrent": recurrent_rows}``.
    Every row receives a ``conversion`` column before the datasets are combined.
    Reward functions therefore receive ``conversion`` alongside other metadata
    and may return ``None`` for examples to which they do not apply.

    All component datasets must have compatible schemas. Python lists of row
    dictionaries and Hugging Face ``Dataset`` objects are both supported, but
    they should not be mixed in the same call.
    """
    combined_dataset = combine_grpo_conversions(converted_datasets, reward_funcs)
    return grpo_tuning(
        model=model,
        tokenizer=tokenizer,
        converted_dataset=combined_dataset,
        reward_funcs=reward_funcs,
        output_dir=output_dir,
        config_overrides=config_overrides,
    )


def combine_grpo_conversions(converted_datasets, reward_funcs):
    """Tag and combine compatible prompt-only datasets for GRPO training."""
    if not isinstance(converted_datasets, Mapping) or not converted_datasets:
        raise ValueError(
            "converted_datasets must be a non-empty mapping of conversion names "
            "to datasets."
        )

    tagged_datasets = []
    dataset_kinds = set()
    for name, dataset in converted_datasets.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every GRPO conversion needs a non-empty string name.")
        _validate_grpo_dataset(dataset, reward_funcs)
        columns = _dataset_columns(dataset)
        if columns is not None and "conversion" in columns:
            raise ValueError(
                f"GRPO conversion {name!r} already has a `conversion` column."
            )
        if isinstance(dataset, (list, tuple)):
            dataset_kinds.add("rows")
            if not dataset:
                raise ValueError(f"GRPO conversion {name!r} is empty.")
            tagged_datasets.append(
                [dict(row, conversion=name) for row in dataset]
            )
        elif hasattr(dataset, "column_names") and hasattr(dataset, "map"):
            dataset_kinds.add("huggingface")
            tagged_datasets.append(
                dataset.map(
                    lambda _, conversion_name=name: {
                        "conversion": conversion_name
                    }
                )
            )
        else:
            raise TypeError(
                f"Unsupported dataset type for GRPO conversion {name!r}: "
                f"{type(dataset).__name__}."
            )

    if len(dataset_kinds) != 1:
        raise TypeError(
            "Do not mix Python row lists and Hugging Face datasets in one "
            "multi-conversion run."
        )
    if "rows" in dataset_kinds:
        return [row for dataset in tagged_datasets for row in dataset]

    from datasets import concatenate_datasets

    try:
        return concatenate_datasets(tagged_datasets)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "GRPO conversion datasets must have compatible Hugging Face schemas."
        ) from error


GRPO_SYSTEM_PROMPT = """Return the response in exactly this XML structure:
<reasoning>
your concise reasoning
</reasoning>
<answer>
the final answer only
</answer>"""


def convert_to_grpo_example(sample, memory=None):
    """Convert one common project sample into a prompt-only GRPO example.

    ``memory`` can be a precomputed recurrent summary.  When omitted, the full
    sample content is used, matching vanilla reasoning.  The answer is retained
    only as reward metadata and is never included in the prompt.
    """
    missing = {"question", "content", "answer"}.difference(sample)
    if missing:
        raise ValueError(f"GRPO sample is missing fields: {', '.join(sorted(missing))}")
    candidates, _, question = prompt_utils.split_question(sample["question"])
    prompt_text = prompt_utils.TEMPLATE_FINAL.format(
        prompt=question + candidates,
        memory=sample["content"] if memory is None else memory,
    )
    image = sample.get("image")
    user_content = prompt_text
    if image is not None:
        # The image itself remains in its dataset column; the prompt holds a
        # placeholder so each processor can insert its own image tokens.
        user_content = [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]
    example = {
        "prompt": [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "answer": sample["answer"],
    }
    if image is not None:
        example["image"] = image
    return example


def _dataset_columns(dataset):
    """Return known columns without consuming iterable datasets."""
    columns = getattr(dataset, "column_names", None)
    if isinstance(columns, dict):
        columns = set().union(*columns.values())
    if columns is not None:
        return set(columns)
    if isinstance(dataset, (list, tuple)) and dataset:
        first = dataset[0]
        return set(first) if isinstance(first, dict) else set()
    return None


def _validate_grpo_dataset(dataset, reward_funcs):
    """Catch the two most common GRPO dataset mistakes before GPU setup."""
    columns = _dataset_columns(dataset)
    if columns is None:
        return
    if "prompt" not in columns:
        raise ValueError("GRPO train_dataset must contain a `prompt` column.")
    rewards = reward_funcs if isinstance(reward_funcs, (list, tuple)) else [reward_funcs]
    if correctness_reward_func in rewards and "answer" not in columns:
        raise ValueError("correctness_reward_func requires an `answer` dataset column.")


def _content_text(content):
    """Flatten text content from TRL's standard or conversational formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _content_text(content.get("content", content.get("text", "")))
    if isinstance(content, (list, tuple)):
        return "".join(_content_text(part) for part in content)
    return str(content)


def _completion_text(completion):
    """Extract generated text from either a string or assistant message list."""
    return _content_text(completion)


def _normalise_answer(answer):
    """Normalize harmless casing and whitespace differences for exact rewards."""
    return " ".join(str(answer).strip().casefold().split())


# Format-related rewards, adapted from Unsloth's GRPO notebook:
# https://github.com/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-GRPO.ipynb
def extract_xml_answer(text: str) -> str:
    """Return the first XML answer body, or an empty string when absent."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = (
        r"^<reasoning>\n.*?\n</reasoning>\n"
        r"<answer>\n.*?\n</answer>\n$"
    )
    responses = [_completion_text(completion) for completion in completions]
    return [
        0.5 if re.fullmatch(pattern, response, flags=re.DOTALL) else 0.0
        for response in responses
    ]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [_completion_text(completion) for completion in completions]
    return [
        0.5 if re.search(pattern, response, flags=re.DOTALL) else 0.0
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
        score -= len(text.split("\n</answer>\n")[-1]) * 0.001

    if text.count("\n</answer>") == 1:
        score += 0.125
        score -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001

    return score


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(_completion_text(completion)) for completion in completions]


# correctness rewards
def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    """Reward answers matching a gold string or any accepted answer alias."""
    responses = [_completion_text(completion) for completion in completions]
    predictions = [extract_xml_answer(response) for response in responses]
    rewards = []
    for prediction, target in zip(predictions, answer):
        targets = target if isinstance(target, (list, tuple, set)) else [target]
        accepted = {_normalise_answer(value) for value in targets}
        rewards.append(1.0 if _normalise_answer(prediction) in accepted else 0.0)
    return rewards


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [_completion_text(completion) for completion in completions]
    predictions = [extract_xml_answer(response) for response in responses]
    return [0.5 if prediction.isdigit() else 0.0 for prediction in predictions]
