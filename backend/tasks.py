import json
import logging
import ssl

from datetime import datetime
from typing import Dict

import redis

from celery import Celery

from groq import Groq

from pydantic import (
    BaseModel,
    ValidationError,
)

from config import settings


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# =========================================================
# REDIS CLIENT
# =========================================================

redis_client = redis.Redis.from_url(
    settings.redis_url,
    decode_responses=True,
    ssl_cert_reqs=ssl.CERT_NONE
)


# =========================================================
# CELERY CONFIGURATION
# =========================================================

celery_app = Celery(
    "tasks",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(

    broker_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    },

    redis_backend_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    },

    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    timezone="UTC",
    enable_utc=True,

    task_track_started=True,
    task_time_limit=300,

    worker_prefetch_multiplier=1,

    result_expires=3600,
)


# =========================================================
# GROQ CLIENT
# =========================================================

groq_client = Groq(
    api_key=settings.GROQ_API_KEY
)


# =========================================================
# PYDANTIC VALIDATION MODEL
# =========================================================

class EvaluationResult(BaseModel):

    accuracy: int
    clarity: int
    relevance: int
    overall_score: float


# =========================================================
# JUDGE SYSTEM PROMPT
# =========================================================

JUDGE_SYSTEM_PROMPT = """
You are a strict AI quality evaluator.

You MUST return ONLY valid JSON.

Evaluation Rubric:

1. accuracy
- factual correctness
- score 1-10

2. clarity
- readability and explanation quality
- score 1-10

3. relevance
- relevance to user prompt
- score 1-10

Compute:
overall_score = average of all scores

IMPORTANT:
- Return ONLY JSON
- No markdown
- No explanations
- No extra text

Required JSON format:

{
  "accuracy": 8,
  "clarity": 9,
  "relevance": 8,
  "overall_score": 8.3
}
"""


# =========================================================
# HELPER FUNCTION
# =========================================================

def extract_json(
    raw_text: str
) -> Dict:
    """
    Robust JSON extraction helper.
    """

    try:

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1

        json_text = raw_text[start:end]

        return json.loads(json_text)

    except Exception as e:

        logger.exception(
            "JSON extraction failed"
        )

        raise ValueError(
            "Invalid JSON response"
        ) from e


# =========================================================
# EVALUATION TASK
# =========================================================

@celery_app.task(name="evaluate_response")
def evaluate_response(
    prompt: str,
    response: str
) -> Dict:
    """
    Production-grade async evaluation task.
    """

    try:

        logger.info(
            "Starting async evaluation"
        )

        # =================================================
        # BUILD EVALUATION PROMPT
        # =================================================

        evaluation_prompt = f"""
        USER PROMPT:
        {prompt}

        MODEL RESPONSE:
        {response}
        """

        # =================================================
        # CALL JUDGE LLM
        # =================================================

        completion = (
            groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            JUDGE_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            evaluation_prompt
                        ),
                    },
                ],
                temperature=0,
                max_tokens=256,
            )
        )

        # =================================================
        # RAW OUTPUT
        # =================================================

        raw_result = (
            completion
            .choices[0]
            .message
            .content
        )

        logger.info(
            f"Judge raw output: "
            f"{raw_result}"
        )

        # =================================================
        # EXTRACT JSON
        # =================================================

        parsed_json = extract_json(
            raw_result
        )

        # =================================================
        # VALIDATE SCHEMA
        # =================================================

        validated_result = (
            EvaluationResult(
                **parsed_json
            )
        )

        # =================================================
        # FINAL STRUCTURED RECORD
        # =================================================

        evaluation_record = {
            "prompt": prompt,
            "response": response,
            "accuracy": (
                validated_result.accuracy
            ),
            "clarity": (
                validated_result.clarity
            ),
            "relevance": (
                validated_result.relevance
            ),
            "overall_score": (
                validated_result
                .overall_score
            ),
            "timestamp": (
                datetime.utcnow()
                .isoformat()
            ),
        }

        # =================================================
        # STORE IN REDIS
        # =================================================

        redis_client.rpush(
            "llm_evaluations",
            json.dumps(
                evaluation_record
            )
        )

        logger.info(
            "Evaluation stored successfully"
        )

        return evaluation_record

    except ValidationError as e:

        logger.exception(
            "Schema validation failed"
        )

        return {
            "error": "Schema validation failed",
            "details": str(e)
        }

    except Exception as e:

        logger.exception(
            "Evaluation task failed"
        )

        return {
            "error": str(e)
        }