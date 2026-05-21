"""
Production Streamlit Frontend

Responsibilities:
- prompt submission
- API communication
- response visualization
- drift monitoring
- evaluation dashboard
"""

import requests
import streamlit as st
import pandas as pd


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title=(
        "LLM Evaluation Platform"
    ),
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# BACKEND CONFIG
# =====================================================

BACKEND_URL = (
    "http://backend:8000"
)


# =====================================================
# TITLE
# =====================================================

st.title(
    "Production LLM Evaluation Platform"
)

st.markdown(
    """
AI Reliability Engineering Dashboard

Features:
- LLM inference
- async evaluation
- drift detection
- monitoring analytics
"""
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "System Monitoring"
)

st.sidebar.success(
    "Frontend Operational"
)


# =====================================================
# PROMPT INPUT
# =====================================================

st.header("LLM Inference")

user_prompt = st.text_area(
    "Enter your prompt",
    height=150,
    placeholder=(
        "Ask something..."
    )
)


# =====================================================
# RUN INFERENCE
# =====================================================

if st.button(
    "Generate Response"
):

    if not user_prompt.strip():

        st.warning(
            "Please enter a prompt."
        )

    else:

        try:

            with st.spinner(
                "Generating response..."
            ):

                response = requests.post(
                    f"{BACKEND_URL}"
                    "/api/v1/inference",
                    json={
                        "prompt": user_prompt
                    },
                    timeout=60
                )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    "Inference completed"
                )

                # ==============================
                # RESPONSE DISPLAY
                # ==============================

                st.subheader(
                    "Generated Response"
                )

                st.write(
                    result["response"]
                )

                # ==============================
                # METRICS
                # ==============================

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Latency (s)",
                        result[
                            "latency_seconds"
                        ]
                    )

                with col2:

                    st.metric(
                        "Evaluation Status",
                        result[
                            "evaluation_status"
                        ]
                    )

            else:

                st.error(
                    f"Backend Error: "
                    f"{response.text}"
                )

        except Exception as error:

            st.exception(error)


# =====================================================
# DRIFT DETECTION SECTION
# =====================================================

st.header(
    "Drift Detection"
)

if st.button(
    "Run Drift Analysis"
):

    try:

        with st.spinner(
            "Running drift analysis..."
        ):

            response = requests.get(
                f"{BACKEND_URL}"
                "/api/v1/drift",
                timeout=30
            )

        if response.status_code == 200:

            drift_result = (
                response.json()
            )

            if (
                "status" in drift_result
                and
                drift_result["status"]
                == "failed"
            ):

                st.warning(
                    drift_result[
                        "error"
                    ]
                )

            else:

                st.success(
                    "Drift analysis completed"
                )

                # ==============================
                # DRIFT METRICS
                # ==============================

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Wasserstein Distance",
                        drift_result[
                            "wasserstein_distance"
                        ]
                    )

                    st.metric(
                        "KS Statistic",
                        drift_result[
                            "ks_statistic"
                        ]
                    )

                with col2:

                    st.metric(
                        "P-Value",
                        drift_result[
                            "p_value"
                        ]
                    )

                    st.metric(
                        "Drift Detected",
                        str(
                            drift_result[
                                "drift_detected"
                            ]
                        )
                    )

                # ==============================
                # MEAN SCORES
                # ==============================

                st.subheader(
                    "Score Distribution"
                )

                mean_df = pd.DataFrame({
                    "Window": [
                        "Baseline",
                        "Current"
                    ],
                    "Mean Score": [
                        drift_result[
                            "baseline_mean"
                        ],
                        drift_result[
                            "current_mean"
                        ]
                    ]
                })

                st.bar_chart(
                    mean_df.set_index(
                        "Window"
                    )
                )

        else:

            st.error(
                f"Drift API Error: "
                f"{response.text}"
            )

    except Exception as error:

        st.exception(error)


# =====================================================
# RECENT EVALUATIONS
# =====================================================

st.header(
    "Recent Evaluations"
)

if st.button(
    "Load Recent Evaluations"
):

    try:

        with st.spinner(
            "Loading evaluations..."
        ):

            response = requests.get(
                f"{BACKEND_URL}"
                "/api/v1/recent-evaluations",
                timeout=30
            )

        if response.status_code == 200:

            data = response.json()

            evaluations = (
                data["results"]
            )

            if len(evaluations) == 0:

                st.warning(
                    "No evaluations found"
                )

            else:

                df = pd.DataFrame(
                    evaluations
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

        else:

            st.error(
                response.text
            )

    except Exception as error:

        st.exception(error)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Production-Grade "
    "Asynchronous LLM "
    "Evaluation Platform"
)