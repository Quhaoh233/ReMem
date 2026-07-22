import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# reads the .env file in the current directory and loads the environment variables into the process's environment. This allows you to access the variables using os.getenv() or directly from the environment.
load_dotenv()

def llm_as_judge(question, golden_answer, pred_answer, model="gpt-5-mini"):  # gpt-4.1-nano-2025-04-14, gpt-5-mini
    
    try:
        if isinstance(pred_answer, dict):
            pred_answer = pred_answer.get("answer", pred_answer)
    except Exception:
        pass
    
    if not pred_answer or (isinstance(pred_answer, str) and pred_answer.strip() == ''):
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
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        # openai_client.chat.completions.create
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a fair judge for web navigation tasks. Focus on core answer correctness, not formatting."
                },
                {"role": "user", "content": prompt}
            ],
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback: try to extract judgement from text
            import json_repair
            try:
                result = json_repair.loads(result_text)
            except Exception:
                result = {"judgement": "error"}
        
        judgement = result.get('judgement', '').strip().lower()
        if judgement not in ['correct', 'incorrect']:
            logger.warning(f"Invalid judgement value: {judgement}, marking as 'incorrect'")
            judgement = 'incorrect'
        
        return {
            "question": question,
            "judgement": judgement,
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }
        
    except Exception as e:
        logger.error(f"Error judging answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "question": question,
            "judgement": "error",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }


def sentence_similarity_judge(text1, text2, model_name="all-MiniLM-L6-v2", threshold=0.8):
    try:
        model = SentenceTransformer(model_name)
        golden_embedding = model.encode(text1, convert_to_tensor=True)
        pred_embedding = model.encode(text2, convert_to_tensor=True)
        cosine_sim = util.cos_sim(golden_embedding, pred_embedding).item()
        
        judgement = "correct" if cosine_sim >= threshold else "incorrect"
        
        return {
            "judgement": judgement,
            "golden_answer": text1,
            "pred_answer": text2,
            "similarity_score": cosine_sim
        }
    except Exception as e:
        logger.error(f"Error in sentence similarity judge: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "judgement": "error",
            "golden_answer": text1,
            "pred_answer": text2,
            "similarity_score": None
        }
