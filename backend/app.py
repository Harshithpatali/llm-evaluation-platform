import json
import logging
import ssl
import time

from contextlib import asynccontextmanager
from typing import Dict

import redis

from tasks import evaluate_response
from statistical_engine import analyze_drift

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from groq import AsyncGroq

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
)

from fastapi.responses import Response

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
# PROMETHEUS METRICS
# =========================================================

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests"
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Inference latency"
)


# =========================================================
# GROQ CLIENT
# =========================================================

groq_client = AsyncGroq(
    api_key=settings.GROQ_API_KEY
)


# =========================================================
# REDIS CLIENT
# =========================================================

redis_client = redis.Redis.from_url(
    settings.redis_url,
    decode_responses=True,
    ssl_cert_reqs=ssl.CERT_NONE
)


# =========================================================
# PYDANTIC REQUEST MODEL
# =========================================================

class InferenceRequest(BaseModel):
    """
    Input request schema.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User prompt"
    )


# =========================================================
# PYDANTIC RESPONSE MODEL
# =========================================================

class InferenceResponse(BaseModel):
    """
    API response schema.
    """

    response: str


# =========================================================
# APPLICATION LIFECYCLE
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting AI Reliability Platform Backend..."
    )

    yield

    logger.info(
        "Shutting down backend..."
    )


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan
)


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """

    return {
        "status": "healthy"
    }


# =========================================================
# METRICS ENDPOINT
# =========================================================

@app.get("/metrics")
async def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )


# =========================================================
# EVALUATION HISTORY
# =========================================================

@app.get("/api/v1/evaluations")
async def get_evaluations():
    """
    Fetch evaluation history.
    """

    try:

        raw_records = redis_client.lrange(
            "llm_evaluations",
            0,
            -1
        )

        parsed_records = [
            json.loads(record)
            for record in raw_records
        ]

        return {
            "evaluations": parsed_records
        }

    except Exception as e:

        logger.exception(
            "Failed to fetch evaluations"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# DRIFT ANALYSIS ENDPOINT
# =========================================================

@app.get("/api/v1/drift")
async def drift_analysis():
    """
    Run statistical drift analysis.
    """

    try:

        result = analyze_drift()

        return result

    except Exception as e:

        logger.exception(
            "Drift analysis failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# INFERENCE ENDPOINT
# =========================================================

@app.post(
    "/api/v1/inference",
    response_model=InferenceResponse
)
async def run_inference(
    request: InferenceRequest
) -> InferenceResponse:
    """
    Main async inference endpoint.
    """

    REQUEST_COUNT.inc()

    start_time = time.time()

    try:

        logger.info(
            "Received inference request"
        )

        completion = await groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            temperature=0.3,
            max_tokens=512,
        )

        response_text = (
            completion.choices[0]
            .message
            .content
        )

        latency = time.time() - start_time

        REQUEST_LATENCY.observe(latency)

        logger.info(
            f"Inference completed in "
            f"{latency:.2f} seconds"
        )

        # =============================================
        # TRIGGER ASYNC EVALUATION
        # =============================================

        evaluate_response.delay(
            request.prompt,
            response_text
        )

        logger.info(
            "Async evaluation task queued"
        )

        return InferenceResponse(
            response=response_text
        )

    except Exception as e:

        logger.exception(
            "Inference failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )