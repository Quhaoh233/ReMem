"""Answer questions directly or by accumulating memory across context chunks."""

import prompt_utils


def _text_tokenizer(tokenizer):
    """Accept either a text tokenizer or a vision processor wrapping one."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def _context_tokens(content, tokenizer):
    """Keep the beginning and end of long contexts; report the original length."""
    tokens = _text_tokenizer(tokenizer).encode(content, add_special_tokens=False)
    original_length = len(tokens)
    # This cap covers source context only, not prompts or generated answers.
    limit = prompt_utils.MAX_CONTEXT_LEN
    # Retain initial context and the latest information when dropping the middle.
    if original_length > limit:
        head_length = limit // 2
        tokens = tokens[:head_length] + tokens[-(limit - head_length) :]
    return tokens, original_length


def generate_response(prompt, image, model, tokenizer):
    """Generate only the continuation, omitting image placeholders for text input."""
    # Vision processors accept typed content blocks; text tokenizers use a string.
    is_processor = hasattr(tokenizer, "tokenizer")
    if image is not None and not is_processor:
        raise ValueError("Image input requires a vision processor.")
    if is_processor:
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image", "image": image})
    else:
        content = prompt
    messages = [{"role": "user", "content": content}]
    # Render the model-specific chat format before converting it into tensors.
    # The generation prompt marks where the assistant should start answering.
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs = {"text": input_text, "add_special_tokens": False, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = image
    inputs = tokenizer(**kwargs).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    responses = model.generate(
        **inputs,
        max_new_tokens=prompt_utils.RECURRENT_MAX_NEW,
        use_cache=True,
        do_sample=True,
        temperature=0.5,
        min_p=0.1,
    )
    # Decoder-only output contains the input prefix followed by the new answer.
    return _text_tokenizer(tokenizer).decode(
        responses[0][input_length:], skip_special_tokens=True
    )


def vanilla_reasoning(sample, model, tokenizer, args):
    """Answer from the context in one generation (args retained for callers)."""
    candidates, _, question = prompt_utils.split_question(sample["question"])
    tokens, token_count = _context_tokens(sample["content"], tokenizer)
    content = sample["content"]
    if token_count > len(tokens):
        content = _text_tokenizer(tokenizer).decode(tokens, skip_special_tokens=True)
    prompt = prompt_utils.TEMPLATE_FINAL.format(
        prompt=question + candidates, memory=content
    )
    response = generate_response(prompt, sample.get("image"), model, tokenizer)
    return sample["question"], sample["answer"], response, token_count


def mem_reasoning(sample, model, tokenizer, chunk_size):
    """Accumulate memory over groups of history items, then answer the question."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    candidates, memory_question, final_question = prompt_utils.split_question(
        sample["question"]
    )
    tokens, token_count = _context_tokens(sample["content"], tokenizer)
    content = sample["content"]
    if token_count > len(tokens):
        content = _text_tokenizer(tokenizer).decode(tokens, skip_special_tokens=True)

    # Recommendation content starts with a persona/preamble, then history items.
    sections = content.split(prompt_utils.SECTION_SEPARATOR)
    if len(sections) > 1:
        preamble, history = sections[0], sections[1:]
    else:
        # Plain QA contexts have no item delimiters but still need to be read.
        preamble, history = "", sections
    memory = prompt_utils.NO_MEMORY
    memory_records = []
    for start in range(0, len(history), chunk_size):
        # Repeat the persona so each memory update has the same user context.
        chunk = preamble + "".join(history[start : start + chunk_size])
        prompt = prompt_utils.TEMPLATE.format(
            prompt=memory_question, chunk=chunk, memory=memory
        )
        # The new summary replaces memory and becomes input to the next chunk.
        memory = generate_response(prompt, sample.get("image"), model, tokenizer)
        memory_records.append(f"Memory {len(memory_records) + 1}: {memory}\n")

    # Answer using the accumulated summary and candidates, not the full history.
    prompt = prompt_utils.TEMPLATE_FINAL.format(
        prompt=final_question + candidates, memory=memory
    )
    response = generate_response(prompt, sample.get("image"), model, tokenizer)
    return (
        sample["question"],
        sample["answer"],
        response,
        token_count,
        history,
        candidates,
        "".join(memory_records),
    )


def recurrent_memory(sample, model, tokenizer):
    """Build memory from token chunks, shared by inference and SFT conversion."""
    _, question, _ = prompt_utils.split_question(sample["question"])
    tokens, token_count = _context_tokens(sample["content"], tokenizer)
    memory = prompt_utils.NO_MEMORY
    # Token chunks work for plain documents without item separators.
    for start in range(0, len(tokens), prompt_utils.RECURRENT_CHUNK_SIZE):
        chunk = _text_tokenizer(tokenizer).decode(
            tokens[start : start + prompt_utils.RECURRENT_CHUNK_SIZE],
            skip_special_tokens=True,
        )
        prompt = prompt_utils.TEMPLATE.format(
            prompt=question, chunk=chunk, memory=memory
        )
        # The new summary replaces memory and becomes input to the next chunk.
        memory = generate_response(prompt, sample.get("image"), model, tokenizer)
    return memory, token_count


def recurrent_reasoning(sample, model, tokenizer, args):
    """Answer after accumulating memory over fixed-size token chunks."""
    candidates, _, question = prompt_utils.split_question(sample["question"])
    memory, token_count = recurrent_memory(sample, model, tokenizer)
    prompt = prompt_utils.TEMPLATE_FINAL.format(
        prompt=question + candidates, memory=memory
    )
    response = generate_response(prompt, sample.get("image"), model, tokenizer)
    return sample["question"], sample["answer"], response, token_count
