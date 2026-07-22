from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import FastLanguageModel, FastVisionModel


# supervised finetuning example
def sft_tuning(model, tokenizer, converted_dataset):
    # ----------------- Finetuning Example -----------------
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = converted_dataset,
        args = SFTConfig(
            per_device_train_batch_size = 2,    # batch size per device during training
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 30,
            # num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.0001,
            lr_scheduler_type = "linear",
            seed = 2026,
            output_dir = "outputs",
            report_to = "none",     # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        ),
    )
    model.save_pretrained("qwen_lora")  # Local saving
    tokenizer.save_pretrained("qwen_lora")
    # model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
    # tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving


# GRPO tuning example
def grpo_tuning(model, tokenizer, converted_dataset):
    # ----------------- GRPO Tuning Example -----------------
    FastVisionModel.for_training(model)

    trainer = GRPOTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = converted_dataset,
        args = GRPOConfig(
            per_device_train_batch_size = 2,    # batch size per device during training
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 30,
            # num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.0001,
            lr_scheduler_type = "linear",
            seed = 2026,
            output_dir = "outputs",
            report_to = "none",     # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        ),
    )
    model.save_pretrained("qwen_grpo_lora")  # Local saving
    tokenizer.save_pretrained("qwen_grpo_lora")
    # model.push_to_hub("your_name/qwen_grpo_lora", token = "YOUR_HF_TOKEN") # Online saving
    # tokenizer.push_to_hub("your_name/qwen_grpo_lora", token = "YOUR_HF_TOKEN") # Online saving