import prompt_utils
import sys


def vanilla_reasoning(sample, model, tokenizer, args):
    # read the sample
    content = sample["content"]
    image = sample["image"]
    question = sample["question"]
    golden_answer = sample["answer"]

    if args.dataset in ["movietv", "books", "games"]:
        question = question.split("---<split>---")
        candidates = question[0]
        memory_question = question[1]
        final_question = question[2]
    else:
        final_question = question
        candidates = ""

    # head-trail truncation for long contexts
    tokens = tokenizer.tokenizer.encode(content, add_special_tokens=False)
    token_num = len(tokens)
    if len(tokens) > prompt_utils.MAX_CONTEXT_LEN:
        tokens = tokens[: prompt_utils.MAX_CONTEXT_LEN // 2] + tokens[-prompt_utils.MAX_CONTEXT_LEN // 2 :]
        content = tokenizer.tokenizer.decode(tokens, skip_special_tokens=True)

    # template the input for the model
    msg = prompt_utils.TEMPLATE_FINAL.format(prompt=final_question+candidates, memory=content)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": msg}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    input_length = inputs.input_ids.shape[1]

    # generate the answer with the model
    responses = model.generate(
        **inputs, 
        max_new_tokens = prompt_utils.RECURRENT_MAX_NEW,
        use_cache = True, 
        temperature = 0.5, 
        min_p = 0.1
        )
    response_text = tokenizer.decode(responses[0][input_length:], skip_special_tokens = True)
    return question, golden_answer, response_text, token_num


# chunked by item number
def mem_reasoning(sample, model, tokenizer, chunk_size):
    # read the sample
    content = sample["content"]
    image = sample["image"]
    question = sample["question"]
    golden_answer = sample["answer"]

    # head-trail truncation for long contexts
    tokens = tokenizer.tokenizer.encode(content, add_special_tokens=False)
    token_num = len(tokens)
    if len(tokens) > prompt_utils.MAX_CONTEXT_LEN:
        tokens = tokens[: prompt_utils.MAX_CONTEXT_LEN // 2] + tokens[-prompt_utils.MAX_CONTEXT_LEN // 2 :]
        content = tokenizer.tokenizer.decode(tokens, skip_special_tokens=True)
    
    # accumulate memory by iterating through chunks of the context
    history = content.split("---<split>---")
    pre_prompt = history[0]
    history = history[1:]

    question = question.split("---<split>---")
    candidates = question[0]
    memory_question = question[1]
    final_question = question[2]

    memory_record = ""
    memory = prompt_utils.NO_MEMORY
    for i in range(0, len(history), chunk_size):
        if i + chunk_size > len(history):
            chunk = pre_prompt + "".join(history[i:])
        else:
            chunk = pre_prompt + "".join(history[i:i+chunk_size])
        msg = prompt_utils.TEMPLATE.format(prompt=memory_question, chunk=chunk, memory=memory)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": msg}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        input_length = inputs.input_ids.shape[1]

        # generate the answer with the model
        responses = model.generate(
            **inputs, 
            max_new_tokens = prompt_utils.RECURRENT_MAX_NEW,
            use_cache = True, 
            temperature = 0.5, 
            min_p = 0.1
            )
        memory = tokenizer.decode(responses[0][input_length:], skip_special_tokens = True)
        memory_record += f"the {i}-th memory = {memory}\n"

    # final answer
    msg = prompt_utils.TEMPLATE_FINAL.format(prompt=final_question+candidates, memory=memory)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": msg}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    input_length = inputs.input_ids.shape[1]

    # generate the answer with the model
    responses = model.generate(
        **inputs, 
        max_new_tokens = prompt_utils.RECURRENT_MAX_NEW,
        use_cache = True, 
        temperature = 0.5, 
        min_p = 0.1
        )
    response_text = tokenizer.decode(responses[0][input_length:], skip_special_tokens = True)

    return question, golden_answer, response_text, token_num, history, candidates, memory_record


# chunked by token length
def recurrent_reasoning(sample, model, tokenizer, args):
    # read the sample
    content = sample["content"]
    image = sample["image"]
    question = sample["question"]
    golden_answer = sample["answer"]

    if args.dataset in ["games", "reads", "movietvs", "books"]:
        question = question.split("---<split>---")
        candidates = question[0]
        memory_question = question[1]
        final_question = question[2]        
    else:
        memory_question = question
        final_question = question
        candidates = ""

    # head-trail truncation for long contexts
    tokens = tokenizer.tokenizer.encode(content, add_special_tokens=False)
    token_num = len(tokens)
    if len(tokens) > prompt_utils.MAX_CONTEXT_LEN:
        tokens = tokens[: prompt_utils.MAX_CONTEXT_LEN // 2] + tokens[-prompt_utils.MAX_CONTEXT_LEN // 2 :]
        content = tokenizer.tokenizer.decode(tokens, skip_special_tokens=True)
    
    # accumulate memory by iterating through chunks of the context
    memory = prompt_utils.NO_MEMORY
    for i in range(0, len(tokens), prompt_utils.RECURRENT_CHUNK_SIZE):
        chunk = tokens[i : i + prompt_utils.RECURRENT_CHUNK_SIZE]
        msg = prompt_utils.TEMPLATE.format(prompt=memory_question, chunk=tokenizer.decode(chunk), memory=memory)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": msg}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        input_length = inputs.input_ids.shape[1]

        # generate the answer with the model
        responses = model.generate(
            **inputs, 
            max_new_tokens = prompt_utils.RECURRENT_MAX_NEW,
            use_cache = True, 
            temperature = 0.5, 
            min_p = 0.1
            )
        memory = tokenizer.decode(responses[0][input_length:], skip_special_tokens = True)
    
    # final answer
    msg = prompt_utils.TEMPLATE_FINAL.format(prompt=final_question+candidates, memory=memory)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": msg}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    input_length = inputs.input_ids.shape[1]

    # generate the answer with the model
    responses = model.generate(
        **inputs, 
        max_new_tokens = prompt_utils.RECURRENT_MAX_NEW,
        use_cache = True, 
        temperature = 0.5, 
        min_p = 0.1
        )
    response_text = tokenizer.decode(responses[0][input_length:], skip_special_tokens = True)

    return question, golden_answer, response_text, token_num

