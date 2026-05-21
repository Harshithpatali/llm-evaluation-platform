"""
Production-grade Celery evaluation workers.

Responsibilities:
- async evaluation
- Redis queue integration
- judge LLM scoring
- structured JSON parsing
- Redis persistence
"""

from celery import Celery

import logging
import time
import json

from typing import (
    Dict,
    Any
)

import redis

from groq import Groq

from config import settings


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)


# =====================================================
# REDIS URL
# =====================================================

REDIS_URL = (
    f"rediss://:"
    f"{settings.REDIS_PASSWORD}"
    f"@{settings.REDIS_HOST}:"
    f"{settings.REDIS_PORT}/0"
)


# =====================================================
# REDIS CLIENT
# =====================================================

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True,
    ssl=True
)


# =====================================================
# CELERY CONFIGURATION
# =====================================================

celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)


# =====================================================
# GROQ CLIENT
# =====================================================

groq_client = Groq(
    api_key=settings.GROQ_API_KEY
)


# =====================================================
# VALIDATION FUNCTION
# =====================================================

def validate_evaluation_output(
    data: Dict[str, Any]
) -> bool:
    """
    Validate structured judge output.
    """

    required_fields = [
        "overall_score",
        "accuracy",
        "clarity",
        "reasoning",
        "helpfulness",
        "explanation"
    ]

    for field in required_fields:

        if field not in data:
            return False

    numeric_fields = [
        "overall_score",
        "accuracy",
        "clarity",
        "reasoning",
        "helpfulness"
    ]

    for field in numeric_fields:

        value = data[field]

        if not isinstance(
            value,
            (int, float)
        ):
            return False

        if value < 1 or value > 10:
            return False

    return True


# =====================================================
# MAIN EVALUATION TASK
# =====================================================

@celery_app.task
def evaluate_response(
    prompt: str,
    response: str
) -> Dict[str, Any]:
    """
    Async judge evaluation task.
    """

    logger.info(
        "Starting evaluation task"
    )

    start_time = time.time()

    try:

        # ==========================================
        # JUDGE PROMPT
        # ==========================================

        judge_prompt = f"""
You are an expert AI evaluator.

Evaluate the response according
to the following rubric:

1. Accuracy
2. Clarity
3. Reasoning
4. Helpfulness

Return ONLY valid JSON.

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}

JSON FORMAT:

{{
  "overall_score": number,
  "accuracy": number,
  "clarity": number,
  "reasoning": number,
  "helpfulness": number,
  "explanation": "brief explanation"
}}
"""

        logger.info(
            "Sending evaluation request "
            "to judge LLM"
        )

        # ==========================================
        # GROQ JUDGE REQUEST
        # ==========================================

        completion = (
            groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": judge_prompt
                    }
                ],
                temperature=0,
                max_tokens=300
            )
        )

        raw_output = (
            completion.choices[0]
            .message.content
        )

        logger.info(
            "Judge response received"
        )

        # ==========================================
        # PARSE JSON
        # ==========================================

        try:

            parsed_output = json.loads(
                raw_output
            )

        except json.JSONDecodeError:

            logger.exception(
                "Invalid JSON from judge"
            )

            return {
                "status": "failed",
                "error": (
                    "Judge returned invalid JSON"
                ),
                "raw_output": raw_output
            }

        # ==========================================
        # VALIDATE OUTPUT
        # ==========================================

        is_valid = (
            validate_evaluation_output(
                parsed_output
            )
        )

        if not is_valid:

            logger.error(
                "Evaluation validation failed"
            )

            return {
                "status": "failed",
                "error": (
                    "Invalid evaluation schema"
                )
            }

        # ==========================================
        # BUILD RESULT
        # ==========================================

        evaluation_result = {
            "prompt": prompt,
            "response": response,
            "overall_score": (
                parsed_output[
                    "overall_score"
                ]
            ),
            "accuracy": (
                parsed_output[
                    "accuracy"
                ]
            ),
            "clarity": (
                parsed_output[
                    "clarity"
                ]
            ),
            "reasoning": (
                parsed_output[
                    "reasoning"
                ]
            ),
            "helpfulness": (
                parsed_output[
                    "helpfulness"
                ]
            ),
            "explanation": (
                parsed_output[
                    "explanation"
                ]
            ),
            "evaluation_latency": round(
                time.time() - start_time,
                3
            ),
            "timestamp": time.time()
        }

        # ==========================================
        # STORE IN REDIS
        # ==========================================

        redis_client.rpush(
            "evaluation_results",
            json.dumps(
                evaluation_result
            )
        )

        logger.info(
            "Evaluation stored successfully"
        )

        return {
            "status": "success",
            "evaluation": (
                evaluation_result
            )
        }

    except Exception as error:

        logger.exception(
            "Evaluation task failed"
        )

        return {
            "status": "failed",
            "error": str(error)
        }