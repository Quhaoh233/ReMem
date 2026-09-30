"""Shared prompt templates and supervised conversation builders."""

SECTION_SEPARATOR = "---<split>---"

# Memory updates read one new chunk together with the previous summary.
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

# Final answering reads memory only; doubled braces survive str.format().
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

# Token counts, not character counts. MAX_INPUT_LEN is reserved for callers.
MAX_INPUT_LEN = 120000
MAX_CONTEXT_LEN = 100000
RECURRENT_CHUNK_SIZE = 5000
RECURRENT_MAX_NEW = 5000

SEARCHING_QUESTION = "Now, what would be the next possible item for the user from the candidates based on the user's interaction history and instruction? Please just give the title of the item as the answer; no explanation is needed."

JUDGING_QUESTION = "Now, is the user likely to interact with the given item? Please answer with a single word: 'Yes' or 'No'. No explanation is needed."


def split_question(question):
    """Return candidates, memory instruction, and final question.

    Recommendation datasets separate these three fields with SECTION_SEPARATOR.
    Ordinary QA datasets use the same question for memory and final answering.
    """
    parts = question.split(SECTION_SEPARATOR, 2)
    if len(parts) == 1:
        return "", question, question
    if len(parts) != 3:
        raise ValueError(
            "Expected a plain question or three separated question fields."
        )
    return tuple(parts)


def _conversation(sample, memory, tokenizer):
    """Build a supervised example without inserting an absent image."""
    candidates, _, question = split_question(sample["question"])
    prompt = TEMPLATE_FINAL.format(prompt=question + candidates, memory=memory)
    answer = sample["answer"]
    # QA may supply answer aliases; an SFT example needs one target response.
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else ""
    if hasattr(tokenizer, "tokenizer"):
        content = [{"type": "text", "text": prompt}]
        if sample.get("image") is not None:
            content.append({"type": "image", "image": sample["image"]})
        answer = [{"type": "text", "text": str(answer)}]
    else:
        if sample.get("image") is not None:
            raise ValueError("Image input requires a vision processor.")
        content = prompt
        answer = str(answer)
    # The assistant turn is the supervised target, not part of the user prompt.
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},
        ]
    }


def convert_to_conversation_vanilla(sample, model, tokenizer):
    """Create an SFT example using the full context as memory."""
    return _conversation(sample, sample["content"], tokenizer)


def convert_to_conversation_recurrent(sample, model, tokenizer):
    """Generate memory from successive chunks before creating an SFT example.

    The caller must put the model in inference mode before converting examples.
    Generation failures propagate so incomplete memory is not silently trained on.
    """
    from reasoning_utils import recurrent_memory

    memory, _ = recurrent_memory(sample, model, tokenizer)
    return _conversation(sample, memory, tokenizer)
