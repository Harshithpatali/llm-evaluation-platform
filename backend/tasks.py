import json
import time
from datetime import datetime
from typing import Dict, Any

import redis
import structlog

from celery import Celery
from groq import Groq
from jsonschema import validate, ValidationError

from config import settings


# =========================================================
# STRUCTURED LOGGER
# =========================================================

logger = structlog.get_logger()


# =========================================================
# CELERY APPLICATION
# =========================================================

celery_app = Celery(
    "llm_evaluation_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)


# =========================================================
# CELERY CONFIGURATION
# =========================================================

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    task_acks_late=True,
    worker_prefetch_multiplier=1,

    task_default_retry_delay=5,
    task_max_retries=3,
)


# =========================================================
# REDIS CLIENT
# =========================================================

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    ssl=True,
    decode_responses=True,
)


# =========================================================
# GROQ CLIENT
# =========================================================

groq_client = Groq(
    api_key=settings.GROQ_API_KEY
)


# =========================================================
# EVALUATION SCHEMA
# =========================================================

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
        },
        "clarity": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
        },
        "helpfulness": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
        },
        "safety": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
        },
        "professionalism": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
        },
        "overall_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
        },
    },
    "required": [
        "correctness",
        "clarity",
        "helpfulness",
        "safety",
        "professionalism",
        "overall_score",
    ],
}


# =========================================================
# JUDGE PROMPT
# =========================================================

JUDGE_SYSTEM_PROMPT = """
You are a highly reliable enterprise AI evaluation system.

You MUST:
- score objectively
- return ONLY valid JSON
- NEVER explain scores
- NEVER include markdown

Return EXACT JSON:

{
    "correctness": 0,
    "clarity": 0,
    "helpfulness": 0,
    "safety": 0,
    "professionalism": 0,
    "overall_score": 0
}
"""


# =========================================================
# JSON EXTRACTION
# =========================================================

def extract_json(content: str) -> Dict[str, Any]:

    start = content.find("{")
    end = content.rfind("}") + 1

    json_str = content[start:end]

    return json.loads(json_str)


# =========================================================
# VALIDATION
# =========================================================

def validate_evaluation(
    evaluation_data: Dict[str, Any]
) -> None:

    validate(
        instance=evaluation_data,
        schema=EVALUATION_SCHEMA,
    )


# =========================================================
# TELEMETRY STORAGE
# =========================================================

def store_telemetry(
    telemetry: Dict[str, Any]
) -> None:

    redis_client.rpush(
        "llm:evaluations",
        json.dumps(telemetry),
    )


# =========================================================
# BACKGROUND TASK
# =========================================================

@celery_app.task(bind=True)
def evaluate_llm_output(
    self,
    request_id: str,
    prompt: str,
    response: str,
) -> Dict[str, Any]:

    start_time = time.perf_counter()

    logger.info(
        "evaluation_started",
        request_id=request_id,
    )

    try:

        evaluation_prompt = f"""
PROMPT:
{prompt}

ASSISTANT RESPONSE:
{response}

Evaluate the response.
"""

        completion = (
            groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt,
                    },
                ],
            )
        )

        raw_output = (
            completion.choices[0]
            .message
            .content
        )

        logger.info(
            "judge_raw_output",
            request_id=request_id,
            raw_output=raw_output,
        )

        evaluation_data = extract_json(
            raw_output
        )

        validate_evaluation(
            evaluation_data
        )

        telemetry = {
            "request_id": request_id,
            "timestamp": (
                datetime.utcnow().isoformat()
            ),
            "prompt": prompt,
            "response": response,
            "evaluation": evaluation_data,
        }

        store_telemetry(telemetry)

        latency = (
            time.perf_counter() - start_time
        )

        logger.info(
            "evaluation_completed",
            request_id=request_id,
            latency=latency,
        )

        return {
            "status": "success",
            "request_id": request_id,
            "evaluation": evaluation_data,
            "latency_seconds": round(
                latency,
                3,
            ),
        }

    except ValidationError as e:

        logger.error(
            "schema_validation_failed",
            request_id=request_id,
            error=str(e),
        )

        raise self.retry(exc=e)

    except Exception as e:

        logger.error(
            "evaluation_pipeline_failed",
            request_id=request_id,
            error=str(e),
        )

        raise self.retry(exc=e)