"""Evaluate model answers while keeping judge failures distinct from wrong answers."""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# Locate configuration even when launched from the project root.
load_dotenv(Path(__file__).with_name(".env"))


# LLM as judge metric: https://github.com/bingreeky/MemEvolve
def llm_as_judge(
    question, golden_answer, pred_answer, model="gpt-5-mini"
):  # gpt-4.1-nano-2025-04-14, gpt-5-mini
    """Return correct, incorrect, or error; malformed judge output is an error."""
    if isinstance(pred_answer, dict):
        pred_answer = pred_answer.get("answer", pred_answer)

    # An empty model response can be marked wrong without calling the judge API.
    if not pred_answer or (isinstance(pred_answer, str) and pred_answer.strip() == ""):
        return {
            "question": question,
            "judgement": "incorrect",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }

    prompt = f"""You are a general AI assistant. Based on the [Correct Answer] provided below, determine whether the [Response] to the [Original Question] is correct.

[Original Question]: {question}

[Correct Answer]: {golden_answer}

[Response]: {pred_answer}

Your judgment must follow this standard:
- Focus only on whether there are substantial differences between the [Response] and the [Correct Answer]
- Do not comment on the background of the question
- Do not attempt to resolve the problem again
- Only focus on judging whether the answers are consistent
- If the [Response] is consistent with the [Correct Answer], or within an acceptable small margin of error for numerical questions, judge as "correct"
- Otherwise (i.e., in cases of any inconsistency, ambiguity, non-equivalence, or incorrectly extracted answer), judge as "incorrect"

Output JSON format:
{{
  "judgement": "correct" or "incorrect"
}}"""

    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fair judge for web navigation tasks. Focus on core answer correctness, not formatting.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        result_text = (response.choices[0].message.content or "").strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback: try to extract judgement from text
            import json_repair

            try:
                result = json_repair.loads(result_text)
            except Exception:
                result = {"judgement": "error"}

        # Parsing valid JSON is not enough: require a recognized verdict as well.
        judgement = result.get("judgement") if isinstance(result, dict) else None
        if isinstance(judgement, str):
            judgement = judgement.strip().lower()
        if judgement not in {"correct", "incorrect"}:
            logger.warning("Invalid judge response: %r", result)
            judgement = "error"

        return {
            "question": question,
            "judgement": judgement,
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }

    except Exception:
        logger.exception("Error judging answer")
        return {
            "question": question,
            "judgement": "error",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }


@lru_cache(maxsize=2)
def _similarity_model(model_name):
    """Avoid loading sentence-transformer weights for every comparison."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def sentence_similarity_judge(
    text1, text2, model_name="all-MiniLM-L6-v2", threshold=0.8
):
    """Compare two strings using cosine similarity of sentence embeddings."""
    try:
        from sentence_transformers import util

        model = _similarity_model(model_name)
        golden_embedding = model.encode(text1, convert_to_tensor=True)
        pred_embedding = model.encode(text2, convert_to_tensor=True)
        # Compare embedding directions, then apply the caller-selected cutoff.
        cosine_sim = util.cos_sim(golden_embedding, pred_embedding).item()

        judgement = "correct" if cosine_sim >= threshold else "incorrect"

        return {
            "judgement": judgement,
            "golden_answer": text1,
            "pred_answer": text2,
            "similarity_score": cosine_sim,
        }
    except Exception:
        logger.exception("Error in sentence similarity judge")
        return {
            "judgement": "error",
            "golden_answer": text1,
            "pred_answer": text2,
            "similarity_score": None,
        }
