import sys

TEMPLATE = """
### You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

### Updated memory:
"""

TEMPLATE_FINAL = """
### You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

### Your Answer:
"""

NO_MEMORY = "No previous memory"

MAX_INPUT_LEN = 120000
MAX_CONTEXT_LEN = 100000
RECURRENT_CHUNK_SIZE = 5000
RECURRENT_MAX_NEW = 5000

SEARCHING_QUESTION = "Now, what would be the next possible item for the user from the candidates based on the user's interaction history and instruction? Please just give the title of the item as the answer; no explanation is needed."

JUDGING_QUESTION = "Now, is the user likely to interact with the given item? Please answer with a single word: 'Yes' or 'No'. No explanation is needed."


def convert_to_conversation_vanilla(sample, model, tokenizer):
    content = sample["content"]
    image = sample["image"]
    question = sample["question"]
    golden_answer = sample["answer"]

    components = question.split("---<split>---")
    candidates = components[0]
    memory_question = components[1]
    final_question = components[2]

    msg = TEMPLATE_FINAL.format(prompt=final_question+candidates, memory=content)
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : msg},
            {"type" : "image", "image" : image} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : golden_answer} ]
        },
    ]
    return { "messages" : conversation }


def convert_to_conversation_recurrent(sample, model, tokenizer):
    # ------------------ Recurrent Reasoning for Long Contexts ------------------
    content = sample["content"]
    image = sample["image"]
    question = sample["question"]
    golden_answer = sample["answer"]
    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": content}
    ]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
        ).to("cuda")
    input_ids = inputs["input_ids"][0].tolist()

    # acculate memory!
    memory = NO_MEMORY
    for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
        chunk = input_ids[i : i + RECURRENT_CHUNK_SIZE]
        msg = TEMPLATE.format(prompt=question, chunk=tokenizer.decode(chunk), memory=memory)
        msg = [
            {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": msg}
            ]},]
        msg = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
            ).to("cuda")
        input_length = inputs.input_ids.shape[1]
        try:
            response = model.generate(**inputs,  # **inputs
                                      # streamer = text_streamer, 
                                      max_new_tokens = RECURRENT_MAX_NEW, 
                                      use_cache = True, 
                                      temperature = 1.5, 
                                      min_p = 0.1
                                      )  # output_ids
            memory = tokenizer.decode(response[0][input_length:], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Error during generation: {e}")
            pass

    
    # ---------- Final Answer Generation Based on Accumulated Memory ---------
    msg = TEMPLATE_FINAL.format(prompt=question, memory=memory)
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : msg},
            {"type" : "image", "image" : image} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : golden_answer} ]
        },
    ]

    return { "messages" : conversation }


