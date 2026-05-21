"""
Production-grade statistical drift engine.

Responsibilities:
- rolling score windows
- Wasserstein distance
- KS statistical testing
- drift monitoring
"""

import json
import logging

from typing import (
    Dict,
    List,
    Any
)

import redis
import numpy as np

from scipy.stats import (
    ks_2samp,
    wasserstein_distance
)

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
# DRIFT CONFIGURATION
# =====================================================

BASELINE_WINDOW_SIZE = 5

CURRENT_WINDOW_SIZE = 3

KS_PVALUE_THRESHOLD = 0.05

WASSERSTEIN_THRESHOLD = 1.5


# =====================================================
# FETCH SCORES
# =====================================================

def fetch_evaluation_scores() -> List[float]:
    """
    Fetch evaluation scores from Redis.
    """

    raw_results = redis_client.lrange(
        "evaluation_results",
        0,
        -1
    )

    scores = []

    for item in raw_results:

        try:

            parsed = json.loads(item)

            scores.append(
                parsed["overall_score"]
            )

        except Exception:

            continue

    return scores


# =====================================================
# CREATE WINDOWS
# =====================================================

def create_score_windows(
    scores: List[float]
) -> Dict[str, List[float]]:
    """
    Create baseline/current windows.
    """

    minimum_required = (
        BASELINE_WINDOW_SIZE +
        CURRENT_WINDOW_SIZE
    )

    if len(scores) < minimum_required:

        raise ValueError(
            "Not enough evaluation data "
            "for drift analysis"
        )

    baseline_window = scores[
        -(minimum_required):
        -CURRENT_WINDOW_SIZE
    ]

    current_window = scores[
        -CURRENT_WINDOW_SIZE:
    ]

    return {
        "baseline": baseline_window,
        "current": current_window
    }


# =====================================================
# DRIFT DETECTION
# =====================================================

def run_drift_detection() -> Dict[str, Any]:
    """
    Run statistical drift analysis.
    """

    logger.info(
        "Starting drift detection"
    )

    try:

        # ==========================================
        # FETCH SCORES
        # ==========================================

        scores = (
            fetch_evaluation_scores()
        )

        logger.info(
            f"Loaded {len(scores)} scores"
        )

        # ==========================================
        # CREATE WINDOWS
        # ==========================================

        windows = (
            create_score_windows(
                scores
            )
        )

        baseline_scores = (
            windows["baseline"]
        )

        current_scores = (
            windows["current"]
        )

        # ==========================================
        # WASSERSTEIN DISTANCE
        # ==========================================

        wasserstein_value = (
            wasserstein_distance(
                baseline_scores,
                current_scores
            )
        )

        # ==========================================
        # KS TEST
        # ==========================================

        ks_statistic, p_value = (
            ks_2samp(
                baseline_scores,
                current_scores
            )
        )

        # ==========================================
        # DRIFT DECISION
        # ==========================================

        drift_detected = (
            p_value < KS_PVALUE_THRESHOLD
            or
            wasserstein_value >
            WASSERSTEIN_THRESHOLD
        )

        # ==========================================
        # SUMMARY STATS
        # ==========================================

        baseline_mean = float(
            np.mean(
                baseline_scores
            )
        )

        current_mean = float(
            np.mean(
                current_scores
            )
        )

        result = {
            "drift_detected": (
                drift_detected
            ),
            "wasserstein_distance": round(
                wasserstein_value,
                4
            ),
            "ks_statistic": round(
                float(ks_statistic),
                4
            ),
            "p_value": round(
                float(p_value),
                6
            ),
            "baseline_mean": round(
                baseline_mean,
                3
            ),
            "current_mean": round(
                current_mean,
                3
            ),
            "baseline_window_size": (
                len(baseline_scores)
            ),
            "current_window_size": (
                len(current_scores)
            )
        }

        logger.info(
            "Drift detection completed"
        )

        return result

    except Exception as error:

        logger.exception(
            "Drift detection failed"
        )

        return {
            "status": "failed",
            "error": str(error)
        }