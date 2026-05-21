import json
import logging
import ssl

from typing import Dict, List

import redis
import numpy as np

from scipy.stats import (
    ks_2samp,
    wasserstein_distance,
)

from config import settings


# =========================================================
# LOGGING
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
# CONFIGURATION
# =========================================================

BASELINE_WINDOW_SIZE = 50
CURRENT_WINDOW_SIZE = 50

DRIFT_THRESHOLD = 0.20

P_VALUE_THRESHOLD = 0.05


# =========================================================
# FETCH EVALUATIONS
# =========================================================

def fetch_evaluations() -> List[Dict]:
    """
    Fetch all evaluation records from Redis.
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

        logger.info(
            f"Fetched "
            f"{len(parsed_records)} "
            f"evaluations"
        )

        return parsed_records

    except Exception as e:

        logger.exception(
            "Failed to fetch evaluations"
        )

        return []


# =========================================================
# EXTRACT SCORES
# =========================================================

def extract_scores(
    evaluations: List[Dict]
) -> List[float]:
    """
    Extract overall scores.
    """

    scores = []

    for evaluation in evaluations:

        if "overall_score" in evaluation:

            scores.append(
                float(
                    evaluation[
                        "overall_score"
                    ]
                )
            )

    return scores


# =========================================================
# SPLIT WINDOWS
# =========================================================

def split_windows(
    scores: List[float]
):
    """
    Split into baseline/current windows.
    """

    baseline_window = scores[
        -(
            BASELINE_WINDOW_SIZE
            +
            CURRENT_WINDOW_SIZE
        ):-CURRENT_WINDOW_SIZE
    ]

    current_window = scores[
        -CURRENT_WINDOW_SIZE:
    ]

    return (
        baseline_window,
        current_window
    )


# =========================================================
# DRIFT ANALYSIS
# =========================================================

def analyze_drift() -> Dict:
    """
    Main statistical drift analysis.
    """

    try:

        # =============================================
        # FETCH DATA
        # =============================================

        evaluations = fetch_evaluations()

        scores = extract_scores(
            evaluations
        )

        # =============================================
        # CHECK DATA SUFFICIENCY
        # =============================================

        minimum_required = (
            BASELINE_WINDOW_SIZE
            +
            CURRENT_WINDOW_SIZE
        )

        if len(scores) < minimum_required:

            return {
                "status": (
                    "insufficient_data"
                ),
                "required": (
                    minimum_required
                ),
                "current": len(scores),
            }

        # =============================================
        # SPLIT WINDOWS
        # =============================================

        (
            baseline_window,
            current_window
        ) = split_windows(scores)

        # =============================================
        # NUMPY ARRAYS
        # =============================================

        baseline_array = np.array(
            baseline_window
        )

        current_array = np.array(
            current_window
        )

        # =============================================
        # KS TEST
        # =============================================

        ks_statistic, p_value = (
            ks_2samp(
                baseline_array,
                current_array
            )
        )

        # =============================================
        # WASSERSTEIN DISTANCE
        # =============================================

        wasserstein_score = (
            wasserstein_distance(
                baseline_array,
                current_array
            )
        )

        # =============================================
        # MEAN COMPARISON
        # =============================================

        baseline_mean = float(
            np.mean(baseline_array)
        )

        current_mean = float(
            np.mean(current_array)
        )

        # =============================================
        # DRIFT LOGIC
        # =============================================

        drift_detected = (

            wasserstein_score
            >
            DRIFT_THRESHOLD

            or

            p_value
            <
            P_VALUE_THRESHOLD
        )

        # =============================================
        # FINAL RESULT
        # =============================================

        result = {

            "drift_detected":
                drift_detected,

            "ks_statistic":
                float(ks_statistic),

            "p_value":
                float(p_value),

            "wasserstein_distance":
                float(wasserstein_score),

            "baseline_mean":
                baseline_mean,

            "current_mean":
                current_mean,

            "baseline_size":
                len(baseline_window),

            "current_size":
                len(current_window),
        }

        logger.info(
            f"Drift analysis result: "
            f"{result}"
        )

        return result

    except Exception as e:

        logger.exception(
            "Drift analysis failed"
        )

        return {
            "error": str(e)
        }


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    result = analyze_drift()

    print(
        json.dumps(
            result,
            indent=2
        )
    )