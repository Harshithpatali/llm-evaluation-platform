import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import redis
import structlog

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.responses import (
    JSONResponse,
    Response,
)

from pydantic import (
    BaseModel,
    Field,
)

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from groq import Groq

from config import settings
from tasks import evaluate_llm_output
from statistical_engine import analyze_drift


# =========================================================
# STRUCTURED LOGGING
# =========================================================

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(
            fmt="iso"
        ),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


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
# PROMETHEUS METRICS
# =========================================================

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Inference request latency",
)

INFERENCE_FAILURES = Counter(
    "inference_failures_total",
    "Total inference failures",
)


# =========================================================
# FASTAPI LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "application_startup",
        environment=settings.ENVIRONMENT,
    )

    yield

    logger.info(
        "application_shutdown"
    )


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="LLM Reliability Platform",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# PYDANTIC MODELS
# =========================================================

class InferenceRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    model: str = Field(
        default="llama-3.1-8b-instant"
    )


class InferenceResponse(BaseModel):

    request_id: str

    response: str

    latency_seconds: float


# =========================================================
# GROQ CLIENT
# =========================================================

groq_client = Groq(
    api_key=settings.GROQ_API_KEY
)


# =========================================================
# GLOBAL EXCEPTION HANDLER
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    INFERENCE_FAILURES.inc()

    logger.error(
        "unhandled_exception",
        error=str(exc),
        path=request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error"
        },
    )


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
    }


# =========================================================
# METRICS ENDPOINT
# =========================================================

@app.get("/metrics")
async def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# =========================================================
# DRIFT ANALYSIS ENDPOINT
# =========================================================

@app.get("/api/v1/drift")
async def drift_analysis():

    try:

        analysis = analyze_drift()

        return analysis

    except Exception as e:

        logger.error(
            "drift_endpoint_failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail="Drift analysis failed",
        )


# =========================================================
# EVALUATION HISTORY ENDPOINT
# =========================================================

@app.get("/api/v1/evaluations")
async def get_evaluation_history():

    """
    Historical evaluation telemetry endpoint.
    """

    try:

        raw_data = redis_client.lrange(
            "llm:evaluations",
            0,
            -1,
        )

        evaluations = []

        for item in raw_data:

            try:

                evaluations.append(
                    json.loads(item)
                )

            except Exception as e:

                logger.error(
                    "evaluation_parse_failed",
                    error=str(e),
                )

        return {
            "count": len(evaluations),
            "evaluations": evaluations,
        }

    except Exception as e:

        logger.error(
            "evaluation_history_failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to load evaluations",
        )


# =========================================================
# MAIN INFERENCE ENDPOINT
# =========================================================

@app.post(
    "/api/v1/inference",
    response_model=InferenceResponse,
)
async def run_inference(
    payload: InferenceRequest
):

    REQUEST_COUNT.inc()

    request_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    logger.info(
        "inference_started",
        request_id=request_id,
        model=payload.model,
    )

    try:

        # =========================================
        # PRIMARY LLM INFERENCE
        # =========================================

        completion = (
            groq_client.chat.completions.create(
                model=payload.model,
                messages=[
                    {
                        "role": "user",
                        "content": payload.prompt,
                    }
                ],
                temperature=0.2,
            )
        )

        model_response = (
            completion.choices[0]
            .message
            .content
        )

        # =========================================
        # ASYNC BACKGROUND EVALUATION
        # =========================================

        evaluate_llm_output.delay(
            request_id=request_id,
            prompt=payload.prompt,
            response=model_response,
        )

        latency = (
            time.perf_counter() - start_time
        )

        REQUEST_LATENCY.observe(
            latency
        )

        logger.info(
            "inference_completed",
            request_id=request_id,
            latency=latency,
        )

        return InferenceResponse(
            request_id=request_id,
            response=model_response,
            latency_seconds=round(
                latency,
                3,
            ),
        )

    except Exception as e:

        INFERENCE_FAILURES.inc()

        logger.error(
            "inference_failed",
            request_id=request_id,
            error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail="Inference failed",
        )