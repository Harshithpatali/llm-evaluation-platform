"""
Main FastAPI application.

Production-Grade Asynchronous
LLM Evaluation & Drift Detection Platform

Responsibilities:
- API routing
- async inference handling
- Celery task triggering
- Prometheus metrics
- drift monitoring
- health checks
- production logging
"""

from typing import Dict, Any
import logging
import time

from fastapi import (
    FastAPI,
    HTTPException
)

from pydantic import (
    BaseModel,
    Field
)

from groq import Groq

from prometheus_fastapi_instrumentator import (
    Instrumentator
)

from config import settings

from tasks import evaluate_response

from statistical_engine import (
    run_drift_detection
)


# =====================================================
# LOGGING CONFIGURATION
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
# FASTAPI APPLICATION INITIALIZATION
# =====================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    description=(
        "Production-Grade Asynchronous "
        "LLM Evaluation & Drift Detection Platform"
    )
)


# =====================================================
# PROMETHEUS METRICS
# =====================================================

Instrumentator().instrument(app).expose(app)


# =====================================================
# GROQ CLIENT
# =====================================================

groq_client = Groq(
    api_key=settings.GROQ_API_KEY
)


# =====================================================
# REQUEST SCHEMA
# =====================================================

class InferenceRequest(BaseModel):
    """
    Request schema for inference API.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User prompt"
    )


# =====================================================
# RESPONSE SCHEMA
# =====================================================

class InferenceResponse(BaseModel):
    """
    Response schema for inference API.
    """

    response: str

    latency_seconds: float

    evaluation_status: str


# =====================================================
# ROOT ENDPOINT
# =====================================================

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint.
    """

    return {
        "message": (
            "Production LLM Evaluation "
            "Platform Running"
        )
    }


# =====================================================
# HEALTH ENDPOINT
# =====================================================

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Used by:
    - Docker
    - Render
    - monitoring systems
    """

    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT
    }


# =====================================================
# TEST ENDPOINT
# =====================================================

@app.get("/api/v1/test")
async def test_endpoint() -> Dict[str, str]:
    """
    Basic API test endpoint.
    """

    logger.info(
        "Test endpoint called"
    )

    return {
        "message": (
            "Backend operational"
        )
    }


# =====================================================
# MAIN INFERENCE ENDPOINT
# =====================================================

@app.post(
    "/api/v1/inference",
    response_model=InferenceResponse
)
async def run_inference(
    request: InferenceRequest
) -> InferenceResponse:
    """
    Main inference endpoint.

    Flow:
    -----
    1. Validate request
    2. Call Groq LLM
    3. Return response immediately
    4. Trigger async evaluation
    """

    logger.info(
        "Received inference request"
    )

    logger.info(
        f"Prompt length: "
        f"{len(request.prompt)}"
    )

    start_time = time.time()

    try:

        # ==========================================
        # LLM API REQUEST
        # ==========================================

        completion = (
            groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ],
                temperature=0.3,
                max_tokens=512
            )
        )

        # ==========================================
        # RESPONSE EXTRACTION
        # ==========================================

        generated_text = (
            completion.choices[0]
            .message.content
        )

        # ==========================================
        # LATENCY CALCULATION
        # ==========================================

        latency = (
            time.time() - start_time
        )

        logger.info(
            f"Inference completed "
            f"in {latency:.3f} seconds"
        )

        # ==========================================
        # TRIGGER ASYNC EVALUATION
        # ==========================================

        try:

            evaluate_response.delay(
                request.prompt,
                generated_text
            )

            logger.info(
                "Async evaluation task queued"
            )

            evaluation_status = "queued"

        except Exception as celery_error:

            logger.exception(
                "Failed to queue evaluation task"
            )

            evaluation_status = (
                f"queue_failed: "
                f"{str(celery_error)}"
            )

        # ==========================================
        # IMMEDIATE RESPONSE
        # ==========================================

        return InferenceResponse(
            response=generated_text,
            latency_seconds=round(
                latency,
                3
            ),
            evaluation_status=(
                evaluation_status
            )
        )

    except Exception as error:

        logger.exception(
            "Inference pipeline failed"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Inference request failed"
                ),
                "error": str(error)
            }
        )


# =====================================================
# DRIFT DETECTION ENDPOINT
# =====================================================

@app.get("/api/v1/drift")
async def drift_analysis() -> Dict[str, Any]:
    """
    Run statistical drift detection.

    Compares:
    - historical evaluation scores
    - recent evaluation scores
    """

    logger.info(
        "Drift analysis endpoint called"
    )

    result = run_drift_detection()

    return result


# =====================================================
# RECENT EVALUATIONS ENDPOINT
# =====================================================

@app.get("/api/v1/recent-evaluations")
async def recent_evaluations():
    """
    Retrieve latest evaluation results.
    """

    import redis
    import json

    try:

        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )

        raw_results = redis_client.lrange(
            "evaluation_results",
            -10,
            -1
        )

        parsed_results = []

        for item in raw_results:

            try:

                parsed_results.append(
                    json.loads(item)
                )

            except Exception:

                continue

        return {
            "count": len(parsed_results),
            "results": parsed_results
        }

    except Exception as error:

        logger.exception(
            "Failed to fetch evaluations"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =====================================================
# STARTUP EVENT
# =====================================================

@app.on_event("startup")
async def startup_event():
    """
    Startup initialization.
    """

    logger.info(
        "Starting Production LLM "
        "Evaluation Platform"
    )

    logger.info(
        f"Environment: "
        f"{settings.ENVIRONMENT}"
    )


# =====================================================
# SHUTDOWN EVENT
# =====================================================

@app.on_event("shutdown")
async def shutdown_event():
    """
    Graceful shutdown event.
    """

    logger.info(
        "Shutting down platform"
    )