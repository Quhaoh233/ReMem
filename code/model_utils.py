"""Load quantized text or vision models and attach trainable LoRA adapters."""

import prompt_utils
from unsloth import FastLanguageModel, FastVisionModel


def load_model(args):
    """Load the requested checkpoint, or initialize a model with fresh adapters."""
    if args.modal not in {"text", "vision"}:
        raise ValueError(f"Unsupported modality: {args.modal}")
    model_class = FastLanguageModel if args.modal == "text" else FastVisionModel
    # Four-bit base weights reduce GPU memory; adapters hold trainable updates.
    model, tokenizer = model_class.from_pretrained(
        model_name=args.load_checkpoint or args.model,
        max_seq_length=prompt_utils.MAX_CONTEXT_LEN,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        token=args.hf_token,
    )
    # Reuse the loaded checkpoint without attaching a fresh set of adapters.
    if args.load_checkpoint:
        return model, tokenizer

    # LoRA learns low-rank weight updates: r sets rank, alpha controls scaling.
    adapter_options = dict(
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    if args.modal == "vision":
        adapter_options.update(
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
    else:
        # Attach text adapters to attention projections and feed-forward layers.
        adapter_options["target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    model = model_class.get_peft_model(model, **adapter_options)
    return model, tokenizer
