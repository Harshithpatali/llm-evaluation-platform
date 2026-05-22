import json
from typing import Dict, List, Any

import redis
import numpy as np
import structlog

from scipy.stats import (
    ks_2samp,
    wasserstein_distance,
)

from config import settings


logger = structlog.get_logger()


redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    ssl=True,
    decode_responses=True,
)


ROLLING_WINDOW_SIZE = 50

BASELINE_WINDOW_SIZE = 200

DRIFT_PVALUE_THRESHOLD = 0.05

WASSERSTEIN_ALERT_THRESHOLD = 1.5


def load_evaluation_telemetry() -> List[Dict[str, Any]]:

    raw_data = redis_client.lrange(
        "llm:evaluations",
        0,
        -1,
    )

    telemetry = []

    for item in raw_data:

        try:
            telemetry.append(json.loads(item))

        except Exception as e:

            logger.error(
                "telemetry_parse_failed",
                error=str(e),
            )

    return telemetry


def extract_scores(
    telemetry: List[Dict[str, Any]]
) -> List[float]:

    scores = []

    for item in telemetry:

        try:

            score = (
                item["evaluation"]
                ["overall_score"]
            )

            scores.append(float(score))

        except Exception as e:

            logger.error(
                "score_extraction_failed",
                error=str(e),
            )

    return scores


def generate_windows(
    scores: List[float]
):

    if len(scores) < (
        BASELINE_WINDOW_SIZE
        + ROLLING_WINDOW_SIZE
    ):
        return None, None

    baseline = scores[
        -(
            BASELINE_WINDOW_SIZE
            + ROLLING_WINDOW_SIZE
        ):-ROLLING_WINDOW_SIZE
    ]

    recent = scores[
        -ROLLING_WINDOW_SIZE:
    ]

    return baseline, recent


def compute_ks_test(
    baseline: List[float],
    recent: List[float],
) -> Dict[str, float]:

    statistic, p_value = ks_2samp(
        baseline,
        recent,
    )

    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
    }


def compute_wasserstein(
    baseline: List[float],
    recent: List[float],
) -> float:

    distance = wasserstein_distance(
        baseline,
        recent,
    )

    return float(distance)


def analyze_drift() -> Dict[str, Any]:

    logger.info(
        "drift_analysis_started"
    )

    telemetry = (
        load_evaluation_telemetry()
    )

    scores = extract_scores(
        telemetry
    )

    baseline, recent = (
        generate_windows(scores)
    )

    if baseline is None:

        return {
            "status": "insufficient_data",
            "message": (
                "Need more telemetry"
            ),
        }

    ks_results = compute_ks_test(
        baseline,
        recent,
    )

    wasserstein = (
        compute_wasserstein(
            baseline,
            recent,
        )
    )

    baseline_mean = float(
        np.mean(baseline)
    )

    recent_mean = float(
        np.mean(recent)
    )

    drift_detected = (
        ks_results["p_value"]
        < DRIFT_PVALUE_THRESHOLD
    )

    severe_drift = (
        wasserstein
        > WASSERSTEIN_ALERT_THRESHOLD
    )

    analysis = {
        "status": "success",

        "telemetry_count": len(scores),

        "baseline_mean": round(
            baseline_mean,
            3,
        ),

        "recent_mean": round(
            recent_mean,
            3,
        ),

        "ks_statistic": round(
            ks_results["ks_statistic"],
            4,
        ),

        "p_value": round(
            ks_results["p_value"],
            6,
        ),

        "wasserstein_distance": round(
            wasserstein,
            4,
        ),

        "drift_detected": drift_detected,

        "severe_drift": severe_drift,
    }

    logger.info(
        "drift_analysis_completed",
        analysis=analysis,
    )

    return analysis