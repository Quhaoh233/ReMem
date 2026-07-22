from unsloth import FastLanguageModel, FastVisionModel
import prompt_utils


def load_model(args):
    # load the checkpoint if needed
    if args.load_checkpoint is not None:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name = "qwen_lora", # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = True, # Set to False for 16bit LoRA
        )
        
    else:
    # initialize the model and tokenizer
        if args.modal == "text":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=prompt_utils.MAX_CONTEXT_LEN,
                load_in_4bit=True,
                use_gradient_checkpointing="unsloth",
                token=args.hf_token,  # Put Your Hugging Face token here if loading from private repositories
            )

            # peft
            model = FastLanguageModel.get_peft_model(
                model,
                # finetune_vision_layers     = True, # False if not finetuning vision layers
                finetune_language_layers   = True, # False if not finetuning language layers
                finetune_attention_modules = True, # False if not finetuning attention layers
                finetune_mlp_modules       = True, # False if not finetuning MLP layers

                r = 16,           # The larger, the higher the accuracy, but might overfit
                lora_alpha = 16,  # Recommended alpha == r at least
                lora_dropout = 0,
                bias = "none",
                random_state = args.seed,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
                # target_modules = "all-linear", # Optional now! Can specify a list if needed
            )

        elif args.modal == "vision":
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=args.model,
                max_seq_length=prompt_utils.MAX_CONTEXT_LEN,
                load_in_4bit=True,
                use_gradient_checkpointing="unsloth",
                token=args.hf_token,
            )

            # peft
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers     = True, # False if not finetuning vision layers
                finetune_language_layers   = True, # False if not finetuning language layers
                finetune_attention_modules = True, # False if not finetuning attention layers
                finetune_mlp_modules       = True, # False if not finetuning MLP layers

                r = 16,           # The larger, the higher the accuracy, but might overfit
                lora_alpha = 16,  # Recommended alpha == r at least
                lora_dropout = 0,
                bias = "none",
                random_state = args.seed,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
                # target_modules = "all-linear", # Optional now! Can specify a list if needed
            )
        else:
            raise ValueError(f"Unsupported modality: {args.modal}")

    return model, tokenizer